"""
implementation of the MoTEF federated learning algorithm (https://arxiv.org/pdf/2405.20114)
compare with document MoTEF_Alg.pdf
part of the course work for optmization in machine learning
"""
from model.resnet8 import *
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from compression_func.Top import top_k_compress

import time

model = ResNet8()

import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # set backend manually for now to avoid problems with shared GPU
    backend = 'gloo'
    # backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def communicate_with_neighbors(rank, world_size, q_h_i, q_g_i):
    device = q_h_i.device
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # Create Buffers for receiving q_h_i and q_g_i
    recv_left_q_h_i = torch.zeros_like(q_h_i)
    recv_right_q_h_i = torch.zeros_like(q_h_i)
    recv_left_q_g_i = torch.zeros_like(q_g_i)
    recv_right_q_g_i = torch.zeros_like(q_g_i)

    # send q_h_i message to left neighbor async and receive from right neighbor sync
    req_send_left_q_h_i = dist.isend(q_h_i, dst=left_neighbor)
    # print(f"{rank} send {q_h_i} to {left_neighbor}")
    req_recv_right_q_h_i = dist.recv(recv_right_q_h_i, src=right_neighbor)
#     print(f"{rank} received {recv_right_q_h_i} from {right_neighbor}")
    req_send_left_q_h_i.wait()

    # send q_h_i message to right neighbor async and receive from left neighbor sync
    req_send_right_q_h_i = dist.isend(q_h_i, dst=right_neighbor)
#     print(f"{rank} send {q_h_i} to {right_neighbor}")
    req_recv_left_q_h_i = dist.recv(recv_left_q_h_i, src=left_neighbor)
#     print(f"{rank} received {recv_left_q_h_i} from {left_neighbor}")
    req_send_right_q_h_i.wait()

    # send q_g_i message to left neighbor async and receive from right neighbor sync
    req_send_left_q_g_i = dist.isend(q_g_i, dst=left_neighbor)
#     print(f"{rank} send {q_g_i} to {left_neighbor}")
    req_recv_right_q_g_i = dist.recv(recv_right_q_g_i, src=right_neighbor)
#     print(f"{rank} received {recv_right_q_g_i} from {right_neighbor}")
    req_send_left_q_g_i.wait()

    # send q_g_i message to right neighbor async and receive from left neighbor sync
    req_send_right_q_g_i = dist.isend(q_g_i, dst=right_neighbor)
#     print(f"{rank} send {q_g_i} to {right_neighbor}")
    req_recv_left_q_g_i = dist.recv(recv_left_q_g_i, src=left_neighbor)
#     print(f"{rank} received {recv_left_q_g_i} from {left_neighbor}")
    req_send_right_q_g_i.wait()

    dist.barrier()
    return (recv_left_q_h_i,
            recv_right_q_h_i), (recv_left_q_g_i, recv_right_q_g_i)


def motef_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, lambda_, com_ratio):
    setup(rank, world_size)

    # num_gpus = torch.cuda.device_count()

    # avoid cuda for now for simplification
    device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")  # All workers use the same GPU
    #     torch.cuda.set_device(device)
    # else:
    #     device = torch.device("cpu")
    model = model.to(device)
    (data, target) = next(iter(train_loader))
    criterion = nn.CrossEntropyLoss()
    # Compute loss
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)

    # # Add L2 penalty (weight decay)
    # weight_decay = 0.0001
    # l2_reg = torch.tensor(0., device=device)
    # for param in model.parameters():
    #     l2_reg += torch.norm(param) ** 2
    # loss += weight_decay * l2_reg / 2
    loss.backward()

    initGrad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

    # Initialize local states
    x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))
    x = torch.randn_like(x) * 0.01
    h = x.clone()
    g = initGrad.clone()
    v = initGrad.clone()
    m = initGrad.clone()

    q_h_i = torch.zeros_like(x)
    q_g_i = torch.zeros_like(x)

    # Define neighbors (assuming a ring topology for simplicity)
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size
    idxs_n = [left_neighbor, right_neighbor]
    neighborStates = {left_neighbor:
                          {"h": torch.zeros_like(x), "g": torch.zeros_like(x)},
                      right_neighbor:
                          {"h": torch.zeros_like(x), "g": torch.zeros_like(x)}}
    weights = torch.tensor([[0.0, 0.5, 0.0, 0.5],
                            [0.5, 0.0, 0.5, 0.0],
                            [0.0, 0.5, 0.0, 0.5],
                            [0.5, 0.0, 0.5, 0.0]])

    # time the experiment
    start_time = time.time()
    ##############################################################
    #                   start training routine                   #
    ##############################################################

    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if batch_idx != 0:
                # Receive q_h and q_g from neighbors
                (q_h_j_left, q_h_j_right), (q_g_j_left, q_g_j_right) = communicate_with_neighbors(rank, world_size, q_h_i,
                                                                                                        q_g_i)

                # update local neighbor states
                neighborStates[left_neighbor]["h"] += q_h_j_left
                neighborStates[left_neighbor]["g"] += q_g_j_left

                neighborStates[right_neighbor]["h"] += q_h_j_right
                neighborStates[right_neighbor]["g"] += q_g_j_right

            # Update x
            weighted_diffs = [weights[rank][x] * (neighborStates[x]["h"] - h) for x in idxs_n]
            mixing = sum(weighted_diffs)
            x += gamma * mixing - eta * v

            # Update model parameters with x
            param_shapes = [p.shape for p in model.parameters()]
            param_numels = [p.numel() for p in model.parameters()]

            with torch.no_grad():
                x_split = x.split(param_numels)
                for param, x_i, shape in zip(model.parameters(), x_split, param_shapes):
                    param.data = x_i.view(shape)
            # Compute q_h
            q_h_i = top_k_compress((x - h), com_ratio)
            h += q_h_i

            # Compute gradient

            # Compute loss
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # # Add L2 penalty (weight decay)
            # weight_decay = 0.0001
            # l2_reg = torch.tensor(0., device=device)
            # for param in model.parameters():
            #     l2_reg += torch.norm(param) ** 2
            # loss += weight_decay * l2_reg / 2
            loss.backward()

            grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

            # print(f"Unclipped Gradient norm: {grad.norm().item()}")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            # Print gradient statistics
            # print(f"Clipped Gradient norm : {grad.norm().item()}")

            # Update m and v
            m_old = m.clone()
            m = (1 - lambda_) * m + lambda_ * grad
            weighted_diffs_glob_grad = [weights[rank][x] * (neighborStates[x]["g"] - g) for x in idxs_n]
            mixing_glob_grad = sum(weighted_diffs_glob_grad)
            v += gamma * mixing_glob_grad + m - m_old

            # Compute q_g
            q_g_i = top_k_compress((v - g), com_ratio)
            g += q_g_i

            # print stats
            print(f"Rank {rank}, Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            print(f"x norm: {x.norm().item()}, v norm: {v.norm().item()}")
            print(f"h norm: {h.norm().item()}, g norm: {g.norm().item()}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time
        print(
            f"Rank {rank}, Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")

        start_time = time.time()

    cleanup()


def worker_fn(rank, world_size, model, trainset, valset, epochs, gamma, eta, lambda_, com_ratio):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(trainset, batch_size=128,
                              num_workers=2, sampler=train_sampler)

    val_loader = DataLoader(valset, batch_size=128, shuffle=False)

    motef_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, lambda_, com_ratio)


def run_motef(world_size, epochs, gamma, eta, lambda_, com_ratio):
    # transform_train = transforms.Compose([
    #     # transforms.RandomCrop(32, padding=4),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = ResNet8(10)
    # model = ResNet8().cpu()

    model.share_memory()

    mp.spawn(
        worker_fn,
        args=(world_size, model, train_set, val_set, epochs, gamma, eta, lambda_, com_ratio),
        nprocs=world_size
    )


if __name__ == "__main__":
    world_size = 4  # Number of nodes
    start_time = time.time()
    ep = 5
    gam = 0.05
    et = 0.0001
    lbd = 0.9
    com = 0.2
    print(f"gamma={gam}, eta={et}, lambda_={lbd}, com_ratio={com}")

    run_motef(world_size=world_size, epochs=ep, gamma=gam, eta=et, lambda_=lbd, com_ratio=com)
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")

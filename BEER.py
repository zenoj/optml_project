"""
Implementation of the BEER federated learning algorithm (https://openreview.net/pdf?id=I47eFCKa1f3)
Part of the course work for optimization in machine learning
"""
# from model.resnet18 import *
from model.simpleMLP import *
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from compression_func.Top import top_k_compress
from torch import nn
import time
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
    req_recv_right_q_h_i = dist.recv(recv_right_q_h_i, src=right_neighbor)
    req_send_left_q_h_i.wait()

    # send q_h_i message to right neighbor async and receive from left neighbor sync
    req_send_right_q_h_i = dist.isend(q_h_i, dst=right_neighbor)
    req_recv_left_q_h_i = dist.recv(recv_left_q_h_i, src=left_neighbor)
    req_send_right_q_h_i.wait()

    # send q_g_i message to left neighbor async and receive from right neighbor sync
    req_send_left_q_g_i = dist.isend(q_g_i, dst=left_neighbor)
    req_recv_right_q_g_i = dist.recv(recv_right_q_g_i, src=right_neighbor)
    req_send_left_q_g_i.wait()

    # send q_g_i message to right neighbor async and receive from left neighbor sync
    req_send_right_q_g_i = dist.isend(q_g_i, dst=right_neighbor)
    req_recv_left_q_g_i = dist.recv(recv_left_q_g_i, src=left_neighbor)
    req_send_right_q_g_i.wait()

    dist.barrier()
    return (recv_left_q_h_i, recv_right_q_h_i), (recv_left_q_g_i, recv_right_q_g_i)

def beer_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, com_ratio):
    setup(rank, world_size)

    # avoid cuda for now for simplification
    device = torch.device('cpu')
    model = model.to(device)
    (data, target) = next(iter(train_loader))
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    initGrad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

    # Initialize local states
    x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))
    x = torch.randn_like(x) * 0.01
    h = x.clone()
    g = initGrad.clone()
    v = initGrad.clone()

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
        old_grad = initGrad.clone()

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

            # Compute q_h
            q_h_i = top_k_compress((x - h), com_ratio)
            h += q_h_i

            # Compute gradient
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            old_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            old_grad = old_grad.clone()
            # Update model parameters with x
            param_shapes = [p.shape for p in model.parameters()]
            param_numels = [p.numel() for p in model.parameters()]

            with torch.no_grad():
                x_split = x.split(param_numels)
                for param, x_i, shape in zip(model.parameters(), x_split, param_shapes):
                    param.data = x_i.view(shape)

            # Compute gradient
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()


            new_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

            # Print gradient statistics
            # print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Weights norm: {[p.data.norm().item() for p in model.parameters()]}")


            # Update v
            weighted_diffs_grad = [weights[rank][x] * (neighborStates[x]["g"] - g) for x in idxs_n]
            grad_mixing = sum(weighted_diffs_grad)
            v += gamma * grad_mixing + new_grad - old_grad

            # Compute q_g
            q_g_i = top_k_compress((v - g), com_ratio)
            g += q_g_i

            # print stats
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                print(f"x norm: {x.norm().item()}, updateSize: {v.norm().item() * eta}")
                print(f"h norm: {h.norm().item()}, g norm: {g.norm().item()}")

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        print(f'Rank {rank}, Epoch {epoch}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%')

    cleanup()

    # calculate time and return the value
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Rank {rank}, Total training time: {elapsed_time:.2f} seconds")

def worker_fn(rank, world_size, model, trainset, valset, epochs, gamma, eta, com_ratio):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(trainset, batch_size=128,
                              num_workers=2, sampler=train_sampler)

    val_loader = DataLoader(valset, batch_size=128, shuffle=False)

    beer_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, com_ratio)

def run_beer(world_size, epochs, gamma, eta, com_ratio):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = MLP()

    model.share_memory()

    mp.spawn(
        worker_fn,
        args=(world_size, model, train_set, val_set, epochs, gamma, eta, com_ratio),
        nprocs=world_size
    )

if __name__ == "__main__":
    world_size = 4  # Number of nodes
    start_time = time.time()
    ep = 10
    gam = 0.001
    et = 0.03
    com = 0.2
    print(f"gamma={gam}, eta={et}, com_ratio={com}")

    run_beer(world_size=world_size, epochs=ep, gamma=gam, eta=et, com_ratio=com)

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")

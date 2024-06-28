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

model = ResNet8()


def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def motef_worker(rank, world_size, model, data_loader, epochs, gamma, eta, lambda_, C_alpha, weight):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Initialize local states
    x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))
    h = torch.zeros_like(x)
    g = torch.zeros_like(x)
    v = torch.zeros_like(x)
    m = torch.zeros_like(x)
    q_h_i = torch.zeros_like(x)
    q_g_i = torch.zeros_like(x)

    # Define neighbors (assuming a ring topology for simplicity)
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size
    idxs_n = [left_neighbor, right_neighbor]
    neighborStates = {left_neighbor:
                          {"h": {torch.zeros_like(x)}, "g": {torch.zeros_like(x)}},
                      right_neighbor:
                          {"h": {torch.zeros_like(x)}, "g": {torch.zeros_like(x)}}}
    weights = torch.empty(world_size, world_size)
    nn.init.zeros_(weights)
    # TODO: fill weights with 1 ones for neighbors and keep 0 for the rest

    ##############################################################
    #                   start training routine                   #
    ##############################################################

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # Receive q_h and q_g from neighbors
            for n in neighborStates.keys():
                # send messages to neighbor
                dist.send(q_h_i, dst=n)
                dist.send(q_g_i, dst=n)

                # receive message from neighbor
                q_h_temp = torch.zeros_like(x)
                q_g_temp = torch.zeros_like(x)
                dist.recv(q_h_temp, src=n)
                dist.recv(q_g_temp, src=n)

                # update local neighbor states
                neighborStates[n]["h"] += q_h_temp
                neighborStates[n]["g"] += q_g_temp

            # Update x
            weighted_diffs = [weights[rank][x] * (neighborStates[x]["h"] - h) for x in idxs_n]
            mixing = sum(weighted_diffs)
            x += gamma * mixing - eta * v
            
            # Compute q_h
            q_h = C_alpha * (x - h)
            h += q_h

            # Compute gradient
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

            # Update m and v
            m_old = m.clone()
            m = (1 - lambda_) * m + lambda_ * grad
            weighted_diffs_glob_grad = [weights[rank][x] * (neighborStates[x]["g"] - g) for x in idxs_n]
            mixing_glob_grad = sum(weighted_diffs_glob_grad)
            v += gamma * mixing_glob_grad + m - m_old

            # Compute q_g
            q_g = C_alpha * (v - g)
            g += q_g

            if batch_idx % 50 == 0:
                print(f"Rank {rank}, Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.3f}")

            # Update model parameters with x
            with torch.no_grad():
                for param, x_i in zip(model.parameters(), x.split(param.numel())):
                    param.data = x_i.view(param.shape)

    cleanup()


def run_motef(world_size, epochs, gamma, eta, lambda_, C_alpha):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    model = ResNet8()
    model.share_memory()

    processes = []
    for rank in range(world_size):
        data_loader = DataLoader(trainset, batch_size=64, shuffle=True,
                                 num_workers=2, sampler=torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=world_size, rank=rank))

        p = mp.Process(target=motef_worker,
                       args=(rank, world_size, model, data_loader, epochs, gamma, eta, lambda_, C_alpha))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    world_size = 4  # Number of nodes
    run_motef(world_size=world_size, epochs=10, gamma=0.1, eta=0.01, lambda_=0.9, C_alpha=0.5)

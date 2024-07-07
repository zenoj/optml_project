"""
Implementation of the MoTEF federated learning algorithm (https://arxiv.org/pdf/2405.20114)
compare with document MoTEF_Alg.pdf
part of the course work for optimization in machine learning
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os
import networkx as nx
import numpy as np
from model.resnet8 import ResNet8
from compression_func.Top import top_k_compress

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_neighbors(rank, world_size, topology, erdos_renyi_prob=0.5):
    neighbors = []
    if topology == 'ring':
        neighbors = [(rank - 1) % world_size, (rank + 1) % world_size]
        return neighbors
    elif topology == 'star':
        if rank == 0:
            neighbors = list(range(1, world_size))
        else:
            neighbors = [0]
        return neighbors
    elif topology == 'grid':
        side = int(world_size ** 0.5)
        row, col = divmod(rank, side)
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= row + i < side and 0 <= col + j < side:
                neighbors.append((row + i) * side + (col + j)
        return neighbors
    elif topology == 'fully-connected':
        neighbors = list(range(world_size))
        neighbors.remove(rank)
        return neighbors
    elif topology == 'erdos-renyi':
        G = nx.erdos_renyi_graph(world_size, erdos_renyi_prob, seed=42)
        neighbors = list(G.neighbors(rank))
        return neighbors
    else:
        raise ValueError("Unknown topology type")

def communicate_with_neighbors(rank, world_size, q_h_i, q_g_i, neighbors):
    device = q_h_i.device
    neighbor_states = {n: {'h': torch.zeros_like(q_h_i), 'g': torch.zeros_like(q_g_i)} for n in neighbors}

    for n in neighbors:
        dist.send(tensor=q_h_i, dst=n)
        dist.recv(tensor=neighbor_states[n]['h'], src=n)
        dist.send(tensor=q_g_i, dst=n)
        dist.recv(tensor=neighbor_states[n]['g'], src=n)

    return neighbor_states

def motef_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, lambda_, com_ratio, topology, p):
    setup(rank, world_size)
    device = torch.device('cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))
    x = torch.randn_like(x) * 0.01
    h = torch.randn_like(x) * 0.01
    g = torch.randn_like(x) * 0.01
    v = torch.randn_like(x) * 0.01
    m = torch.randn_like(x) * 0.01

    q_h_i = torch.zeros_like(x)
    q_g_i = torch.zeros_like(x)

    neighbors = get_neighbors(rank, world_size, topology, p)
    weights = torch.zeros(world_size, world_size)
    for neighbor in neighbors:
        weights[rank][neighbor] = 1.0 / len(neighbors)
    weights[rank][rank] = 1.0 / len(neighbors)

    start_time = time.time()
    
    ##############################################################
    #                   start training routine                   #
    ##############################################################

    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            neighbor_states = communicate_with_neighbors(rank, world_size, q_h_i, q_g_i, neighbors)

            weighted_diffs = [weights[rank][n] * (neighbor_states[n]['h'] - h) for n in neighbors]
            mixing = sum(weighted_diffs)
            x += gamma * mixing - eta * v

            q_h_i = top_k_compress((x - h), com_ratio)
            h += q_h_i

            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            m_old = m.clone()
            m = (1 - lambda_) * m + lambda_ * grad
            weighted_diffs_glob_grad = [weights[rank][n] * (neighbor_states[n]['g'] - g) for n in neighbors]
            mixing_glob_grad = sum(weighted_diffs_glob_grad)
            v += gamma * mixing_glob_grad + m - m_old

            q_g_i = top_k_compress((v - g), com_ratio)
            g += q_g_i

            param_shapes = [p.shape for p in model.parameters()]
            param_numels = [p.numel() for p in model.parameters()]

            with torch.no_grad():
                x_split = x.split(param_numels)
                for param, x_i, shape in zip(model.parameters(), x_split, param_shapes):
                    param.data = x_i.view(shape)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time
        print(f"Rank {rank}, Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")

        start_time = time.time()

    cleanup()

def worker_fn(rank, world_size, model, trainset, valset, epochs, gamma, eta, lambda_, com_ratio, topology, p):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(trainset, batch_size=64, num_workers=2, sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False)

    motef_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, lambda_, com_ratio, topology, p)

def run_motef(world_size, epochs, gamma, eta, lambda_, com_ratio, topology, p=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    model = ResNet8()
    model.share_memory()

    mp.spawn(
        worker_fn,
        args=(world_size, model, trainset, valset, epochs, gamma, eta, lambda_, com_ratio, topology, p),
        nprocs=world_size
    )

if __name__ == "__main__":
    world_size = 3
    start_time = time.time()
    run_motef(world_size=world_size, epochs=10, gamma=0.01, eta=0.01, lambda_=0.9, com_ratio=0.2, topology='ring', p=0.5)
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")

"""
implementation of the MoTEF federated learning algorithm (https://arxiv.org/pdf/2405.20114)
compare with document MoTEF_Alg.pdf
part of the course work for optmization in machine learning
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
from torchvision.models import resnet18
from torch import nn
import time
import os
import networkx as nx


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def communicate_with_neighbors(rank, world_size, q_h_i, q_g_i, adjacency_matrix):
    neighbors = [i for i in range(world_size) if adjacency_matrix[rank][i] != 0]
    device = q_h_i.device

    recv_q_h_i = {neighbor: torch.zeros_like(q_h_i) for neighbor in neighbors}
    recv_q_g_i = {neighbor: torch.zeros_like(q_g_i) for neighbor in neighbors}

    print(f"Rank {rank} communicating with neighbors: {neighbors}")

    send_requests = []
    for neighbor in neighbors:
        send_requests.append(dist.isend(q_h_i, dst=neighbor))
        print(f"Rank {rank} sent q_h_i to {neighbor}")
    
    for neighbor in neighbors:
        dist.recv(recv_q_h_i[neighbor], src=neighbor)
        print(f"Rank {rank} received q_h_i from {neighbor}")

    for request in send_requests:
        request.wait()  # Ensure all sends are completed

    send_requests = []
    for neighbor in neighbors:
        send_requests.append(dist.isend(q_g_i, dst=neighbor))
        print(f"Rank {rank} sent q_g_i to {neighbor}")

    for neighbor in neighbors:
        dist.recv(recv_q_g_i[neighbor], src=neighbor)
        print(f"Rank {rank} received q_g_i from {neighbor}")

    for request in send_requests:
        request.wait()  # Ensure all sends are completed

    dist.barrier()
    print(f"Rank {rank} passed the barrier")

    return {neighbor: recv_q_h_i[neighbor] for neighbor in neighbors}, {neighbor: recv_q_g_i[neighbor] for neighbor in neighbors}


def create_adjacency_matrix(topology, world_size, prob=0.1):
    if topology == 'ring':
        G = nx.cycle_graph(world_size)
    elif topology == 'fully-connected':
        G = nx.complete_graph(world_size)
    elif topology == 'star':
        G = nx.star_graph(world_size - 1)
    elif topology == 'grid':
        size = int(world_size ** 0.5)
        G = nx.grid_2d_graph(size, size)
        G = nx.convert_node_labels_to_integers(G)
    elif topology == 'erdos-renyi':
        G = nx.erdos_renyi_graph(world_size, prob)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    return adjacency_matrix


def motef_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, lambda_, com_ratio, adjacency_matrix):
    setup(rank, world_size)
    device = torch.device('cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()

    data_iter = iter(train_loader)
    data, target = next(data_iter)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    initGrad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

    x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))
    x = torch.randn_like(x) * 0.01
    h = x.clone()
    g = initGrad.clone()
    v = initGrad.clone()
    m = initGrad.clone()

    q_h_i = torch.zeros_like(x)
    q_g_i = torch.zeros_like(x)

    neighborStates = {neighbor: {"h": torch.zeros_like(x), "g": torch.zeros_like(x)} for neighbor in range(world_size) if adjacency_matrix[rank][neighbor] != 0}
    weights = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)

    start_time = time.time()
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if batch_idx != 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx} - starting communication")
                q_h_j, q_g_j = communicate_with_neighbors(rank, world_size, q_h_i, q_g_i, adjacency_matrix)
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx} - completed communication")

                for neighbor in neighborStates:
                    neighborStates[neighbor]["h"] += q_h_j[neighbor]
                    neighborStates[neighbor]["g"] += q_g_j[neighbor]

            weighted_diffs = sum([weights[rank][neighbor] * (neighborStates[neighbor]["h"] - h) for neighbor in neighborStates])
            x += gamma * weighted_diffs - eta * v

            param_shapes = [p.shape for p in model.parameters()]
            param_numels = [p.numel() for p in model.parameters()]

            with torch.no_grad():
                x_split = x.split(param_numels)
                for param, x_i, shape in zip(model.parameters(), x_split, param_shapes):
                    param.data = x_i.view(shape)

            q_h_i = top_k_compress((x - h), com_ratio)
            h += q_h_i

            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx} - Unclipped Gradient norm: {grad.norm().item()}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

            m_old = m.clone()
            m = (1 - lambda_) * m + lambda_ * grad
            weighted_diffs_glob_grad = sum([weights[rank][neighbor] * (neighborStates[neighbor]["g"] - g) for neighbor in neighborStates])
            v += gamma * weighted_diffs_glob_grad + m - m_old

            q_g_i = top_k_compress((v - g), com_ratio)
            g += q_g_i

            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            print(f"x norm: {x.norm().item()}, v norm: {v.norm().item()}")
            print(f"h norm: {h.norm().item()}, g norm: {g.norm().item()}")

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


def worker_fn(rank, world_size, model, trainset, valset, epochs, gamma, eta, lambda_, com_ratio, adjacency_matrix):
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(trainset, batch_size=128, num_workers=2, sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False)
    motef_worker(rank, world_size, model, train_loader, val_loader, epochs, gamma, eta, lambda_, com_ratio, adjacency_matrix)


def run_motef(world_size, epochs, gamma, eta, lambda_, com_ratio, topology, prob=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = MLP()
    model.share_memory()

    adjacency_matrix = create_adjacency_matrix(topology, world_size, prob)

    mp.spawn(worker_fn, args=(world_size, model, train_set, val_set, epochs, gamma, eta, lambda_, com_ratio, adjacency_matrix), nprocs=world_size)


if __name__ == "__main__":
    world_size = 4
    start_time = time.time()
    ep = 10
    gam = 0.001
    et = 0.03
    lbd = 0.9
    com = 0.2
    topology = 'grid'  # 'ring', 'fully-connected', 'star', 'grid', 'erdos-renyi'
    prob = 0.1  # Only used for erdos-renyi topology

    if topology == 'erdos-renyi': 
        print(f"gamma={gam}, eta={et}, lambda_={lbd}, com_ratio={com}, topology={topology}, prob={prob}")
    else:
        print(f"gamma={gam}, eta={et}, lambda_={lbd}, com_ratio={com}, topology={topology}")    
    run_motef(world_size=world_size, epochs=ep, gamma=gam, eta=et, lambda_=lbd, com_ratio=com, topology=topology, prob=prob)
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")

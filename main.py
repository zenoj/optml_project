"""
implementation of the MoTEF federated learning algorithm (https://arxiv.org/pdf/2405.20114)
compare with document MoTEF_Alg.pdf
part of the course work for optmization in machine learning
"""
import itertools
import os
import time

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import early_stopping
from compression_ops import *
# from model.resnet18 import *
from model.simpleMLP import *
from optimizers.Beer import BEER
from optimizers.Motef import MoTEF
from world import create_adjacency_matrix
from early_stopping import early_stopping_factory

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def motef_worker(rank, world_size, model, optimizer, train_loader, val_loader, epochs, gamma, eta, lambda_, comp_func, com_ratio,
                 adjacency_matrix):
    setup(rank, world_size)
    device = torch.device('cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    early_stopping = early_stopping_factory()

    # initialize optimizer
    optim = optimizer(world_size, rank, model, train_loader, adjacency_matrix, gamma, eta, lambda_, comp_func, com_ratio)

    start_time = time.time()
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # go one iteration of the optimizer and return current loss
            loss = optim.step(data, target)
            # if batch_idx % 2 == 0:
            #     print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            #     print(f"x norm: {optim.x.norm().item()}, v norm: {optim.v.norm().item()}")
            #     print(f"h norm: {optim.h.norm().item()}, g norm: {optim.g.norm().item()}")

            # do validation every 1/5 of an epoch
            # if (batch_idx + 1) % int(3000 / data.size(0)) == 0:
            #     model.eval()
            #     val_loss = 0
            #     val_correct = 0
            #     val_total = 0
            #     with torch.no_grad():
            #         for data, target in val_loader:
            #             data, target = data.to(device), target.to(device)
            #             output = model(data)
            #             val_loss += criterion(output, target).item()
            #             _, predicted = torch.max(output.data, 1)
            #             val_total += target.size(0)
            #             val_correct += (predicted == target).sum().item()
            #
            #     val_loss /= len(val_loader)
            #     val_acc = val_correct / val_total
            #     epoch_time = time.time() - start_time
            #     print(
            #         f"Rank {rank}, Epoch {epoch + 1}, Batch_idx:{batch_idx}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")
            #
            #     start_time = time.time()
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
        print(
            f"Rank {rank}, Epoch {epoch + 1}, Batch_idx:{batch_idx}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")

        start_time = time.time()
    cleanup()


def worker_fn(rank, optimizer, world_size, model, trainset, valset, epochs, gamma, eta, lambda_, comp_func, com_ratio,
              adjacency_matrix, batch_size):
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank,
                                                                    shuffle=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=2, sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    print(f"rank:{rank}")
    motef_worker(rank, world_size, model, optimizer, train_loader, val_loader, epochs, gamma, eta, lambda_, comp_func, com_ratio,
                 adjacency_matrix)


def run_motef(optimizer, world_size, epochs, gamma, eta, lambda_, comp_func, com_ratio, topology, prob=0.1, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = MLP()
    model.share_memory()

    adjacency_matrix = create_adjacency_matrix(topology, world_size, prob)

    mp.spawn(worker_fn, args=(
        optimizer, world_size, model, train_set, val_set, epochs, gamma, eta, lambda_, comp_func, com_ratio, adjacency_matrix, batch_size),
             nprocs=world_size)


if __name__ == "__main__":
    # for beer: eta=0.01, gamma=0.002, com_ratio=0.2, comp_func=top_k, topology=grid,
    # for motef: eta=0.03, gamma=0.001, com_ratio=0.2, comp_func=top_k, topology=grid

    world_size = 4
    start_time = time.time()
    ep = 10
    batch_sizes = [128]
    comp_ratios = [8]
    gammas = [0.001]
    etas = [0.03]
    lbds = [0.1]

    # topologies = ['grid', "ring", 'fully-connected', 'star', 'erdos-renyi']
    topologies = ['ring']
    prob = 0.1  # Only used for erdos-renyi topology
    comp_funcs = [uniform_quantize]
    optimizers = [MoTEF]
    beer_baseline = {"eta": 0.01, "gamma": 0.002, "comp_func": top_k, "comp_ratio": 0.2, "topology": "ring", "prob": prob}
    # for beer: eta=0.01, gamma=0.002, com_ratio=0.2, comp_func=top_k, topology=grid,
    # for motef: eta=0.03, gamma=0.001, com_ratio=0.2, comp_func=top_k, topology=grid
    for optimizer, gam, et, lbd, comp_func, comp_ratio, topology, batch_size in itertools.product(optimizers, gammas, etas, lbds, comp_funcs, comp_ratios, topologies, batch_sizes):
        if topology == 'erdos-renyi':
            print(f"optimizer:{optimizer}, gamma={gam}, eta={et}, lambda_={lbd}, comp_func: {comp_func} comp_ratio={comp_ratio}, topology={topology}, prob={prob}")
        else:
            print(f"optimizer:{optimizer}, gamma={gam}, eta={et}, lambda_={lbd}, comp_func: {comp_func}, comp_ratio={comp_ratio}, topology={topology}, batch_size={batch_size}")

        run_motef(optimizer, world_size=world_size, epochs=ep, gamma=gam, eta=et, lambda_=lbd, comp_func=comp_func, com_ratio=comp_ratio,
                  topology=topology, prob=prob, batch_size=batch_size)

        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f}s")

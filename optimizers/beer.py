"""
implementation of the MoTEF federated learning algorithm (https://arxiv.org/pdf/2405.20114)
compare with document MoTEF_Alg.pdf
part of the course work for optmization in machine learning
"""

from compression_ops import *
from torch import nn
from world import communicate_with_neighbors

torch.manual_seed(42)

class BEER():
    def __init__(self, world_size, rank, model, train_loader, adjacency_matrix, gamma, eta, lbd, comp_func, com_ratio):
        # model parameters
        self.c = 0
        self.model = model
        self.train_loader = train_loader

        device = torch.device('cpu')
        self.model = self.model.to(device)

        # initialize g,v,m as gradient from first batch
        criterion = nn.CrossEntropyLoss()
        self.model.zero_grad()
        data_iter = iter(train_loader)
        data, target = next(data_iter)
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        init_grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])

        # init x and h as the same random tensor and q_h_i/q_g_i as zeros
        x = torch.zeros_like(torch.cat([p.data.view(-1) for p in self.model.parameters()]))

        #  final initialization of optimizer variables
        self.g, self.v, self.m = init_grad.clone(), init_grad.clone(), init_grad.clone()
        self.x = torch.randn_like(x) * 0.01
        self.h = x.clone()
        self.q_h_i, self.q_g_i = torch.zeros_like(x), torch.zeros_like(x)

        # world variables
        self.neighbor_states = {neighbor: {"h": torch.zeros_like(x), "g": torch.zeros_like(x)} for neighbor in
                                range(world_size)
                                if adjacency_matrix[rank][neighbor] != 0}
        self.weights = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)

        self.world_size, self.adjacency_matrix, self.rank = world_size, adjacency_matrix, rank

        # hyperparameters
        self.eta, self.lbd, self.gamma = eta, lbd, gamma
        self.comp_func, self.com_ratio = comp_func, com_ratio

    def step(self, data, target):
        criterion = nn.CrossEntropyLoss()
        if self.c != 0:
            q_h_j, q_g_j = communicate_with_neighbors(self.rank, self.world_size, self.q_h_i, self.q_g_i,
                                                      self.adjacency_matrix)
            for n in self.neighbor_states:
                self.neighbor_states[n]["h"] += q_h_j[n]
                self.neighbor_states[n]["g"] += q_g_j[n]

        # Compute gradient
        self.model.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        old_grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])

        weighted_diffs = sum(
            [self.weights[self.rank][n] * (self.neighbor_states[n]["h"] - self.h) for n in self.neighbor_states])
        self.x += self.gamma * weighted_diffs - self.eta * self.v

        param_shapes = [p.shape for p in self.model.parameters()]
        param_numels = [p.numel() for p in self.model.parameters()]

        with torch.no_grad():
            x_split = self.x.split(param_numels)
            for param, x_i, shape in zip(self.model.parameters(), x_split, param_shapes):
                param.data = x_i.view(shape)

        self.q_h_i = self.comp_func((self.x - self.h), self.com_ratio)
        self.h += self.q_h_i

        self.model.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()

        new_grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])

        # Update v
        weighted_diffs_grad = sum(
            [self.weights[self.rank][n] * (self.neighbor_states[n]["g"] - self.g) for n in self.neighbor_states])
        self.v += self.gamma * weighted_diffs_grad + new_grad - old_grad
        self.q_g_i = self.comp_func((self.v - self.g), self.com_ratio)
        self.g += self.q_g_i
        self.c = 1
        return loss

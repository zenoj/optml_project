"""
implementation of the MoTEF federated learning algorithm (https://arxiv.org/pdf/2405.20114)
compare with document MoTEF_Alg.pdf
part of the course work for optmization in machine learning
"""

from compression_ops import *
from torch import nn
from world import communicate_with_neighbors

class MoTEF():
    def __init__(self, world_size, rank, model, train_loader, adjacency_matrix, gamma, eta, lbd, comp_func, com_ratio):
        # model parameters
        self.c = 0
        self.model = model
        self.train_loader = train_loader

        device = torch.device('cpu')
        model = model.to(device)

        # initialize g,v,m as gradient from first batch
        criterion = nn.CrossEntropyLoss()
        model.zero_grad()
        data_iter = iter(train_loader)
        data, target = next(data_iter)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        init_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

        # init x and h as the same random tensor and q_h_i/q_g_i as zeros
        x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))

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
        # shortcuts
        model, x, h, v, m, g,  = self.model, self.x, self.h, self.v, self.m, self.g
        neighbor_states, q_h_i, q_g_i = self.neighbor_states, self.q_h_i, self.q_g_i

        if self.c != 0:
            q_h_j, q_g_j = communicate_with_neighbors(self.rank, self.world_size, q_h_i, q_g_i, self.adjacency_matrix)
            for n in neighbor_states:
                neighbor_states[n]["h"] += q_h_j[n]
                neighbor_states[n]["g"] += q_g_j[n]

        weighted_diffs = sum(
            [self.weights[self.rank][n] * (neighbor_states[n]["h"] - h) for n in neighbor_states])
        x += self.gamma * weighted_diffs - self.eta * v

        param_shapes = [p.shape for p in model.parameters()]
        param_numels = [p.numel() for p in model.parameters()]

        with torch.no_grad():
            x_split = x.split(param_numels)
            for param, x_i, shape in zip(model.parameters(), x_split, param_shapes):
                param.data = x_i.view(shape)

        q_h_i = self.comp_func((x - h), self.com_ratio)
        h += q_h_i

        criterion = nn.CrossEntropyLoss()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

        m_old = m.clone()
        m = (1 - self.lbd) * m + self.lbd * grad
        weighted_diffs_glob_grad = sum(
            [self.weights[self.rank][n] * (neighbor_states[n]["g"] - g) for n in neighbor_states])
        v += self.gamma * weighted_diffs_glob_grad + m - m_old

        q_g_i = self.comp_func((v - g), self.com_ratio)
        g += q_g_i
        self.c = 1
        return loss






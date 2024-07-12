import torch
def motefSimulation(model, num_workers):
    # init tensors
    num_params = sum(p.numel() for p in model.parameters())
    x = torch.rand(num_workers, num_params)
    w = torch
    W_shifted = W - xp.eye(self.p.n_agent)

    self.H = xp.zeros((self.p.dim, self.p.n_agent))
    self.V = self.grad(self.x)
    self.G = xp.zeros((self.p.dim, self.p.n_agent))
    self.M = xp.zeros((self.p.dim, self.p.n_agent))

    self.M_previous = xp.zeros((self.p.dim, self.p.n_agent))


def main():
    print("start simulation")
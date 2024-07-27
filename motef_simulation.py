import torch

import numpy as np
from nda import log

def create_mixing_matrix(n):
    """
    Create a mixing matrix W of dimensions n x n with the following properties:
    1. Symmetric (W = W^T)
    2. Doubly stochastic (W1 = 1, 1^T W = 1^T)
    3. Eigenvalues: 1 = |λ1(W)| > |λ2(W)| ≥ ... ≥ |λn(W)|

    Args:
    n (int): Dimension of the matrix

    Returns:
    numpy.ndarray: The mixing matrix W
    """
    # Start with a random symmetric matrix
    A = np.random.rand(n, n)
    A = (A + A.T) / 2

    # Ensure positive definiteness
    A = A + n * np.eye(n)

    # Apply Sinkhorn-Knopp algorithm to make the matrix doubly stochastic
    D1 = np.eye(n)
    D2 = np.eye(n)
    for _ in range(1000):  # Usually converges much faster
        D1 = np.diag(1 / np.sum(A, axis=1))
        D2 = np.diag(1 / np.sum(D1 @ A, axis=0))
        A = D1 @ A @ D2

    # Ensure symmetry after Sinkhorn-Knopp
    W = (A + A.T) / 2

    # Adjust eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    new_eigenvalues = np.abs(eigenvalues)
    new_eigenvalues = new_eigenvalues / np.max(new_eigenvalues)  # Ensure largest eigenvalue is 1
    new_eigenvalues[0] = 1  # Set largest eigenvalue to exactly 1
    new_eigenvalues[1:] = 0.99 * new_eigenvalues[1:]  # Ensure strict inequality

    # Reconstruct W with adjusted eigenvalues
    W = eigenvectors @ np.diag(new_eigenvalues) @ eigenvectors.T

    # Final normalization to ensure double stochasticity
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]

    return W


class MoTEFSimulation:
    def __init__(self, params, eta=0.1, gamma=0.1, lmbd=0.9, batch_size=1, compressor_type=None, compressor_param=None):
        self.params = list(params)
        self.eta = eta
        self.gamma = gamma
        self.lmbd = lmbd
        log.info(f'gamma = {gamma:.3f}')
        self.batch_size = batch_size
        self.compressor_param = compressor_param
        self.n_agent = len(self.params)
        self.dim = sum(p.numel() for p in self.params)
        self.x

        # Initialize W as identity matrix for now (you may want to set this differently)
        self.W = torch.eye(self.n_agent)
        self.W_shifted = self.W - torch.eye(self.n_agent)

        self.H = torch.zeros(self.dim, self.n_agent)
        self.V = self.grad()
        self.G = torch.zeros(self.dim, self.n_agent)
        self.M = torch.zeros(self.dim, self.n_agent)
        self.M_previous = torch.zeros(self.dim, self.n_agent)

        self.comm_rounds = 0
        # Compressor
        if compressor_type == 'top':
            self.C = self.top_k
        elif compressor_type == 'random':
            self.C = self.random_k
        elif compressor_type == 'gsgd':
            self.C = self.gsgd
        else:
            self.C = lambda x, _: x  # identity

    def grad(self, x):
        # Compute gradient for all parameters
        if x is None:
            return torch.cat([p.grad.flatten() for p in self.params])
        else:
            # This part needs to be implemented based on your specific use case
            # as it's not clear how 'j' is used to compute partial gradients
            raise NotImplementedError("Partial gradient computation not implemented")

    def step(self):
        self.comm_rounds += 1

        # Flatten all parameters
        x = torch.cat([p.data.flatten() for p in self.params])

        # Update x
        x += self.gamma * self.H.mm(self.W_shifted) - self.eta * self.V

        # Update H
        self.H += self.C(x.unsqueeze(1) - self.H, self.compressor_param)

        self.M_previous = self.M.clone()

        # Compute new gradient
        grad = self.grad()
        self.M = (1 - self.lmbd) * self.M + self.lmbd * grad.unsqueeze(1)

        # Update V
        self.V += self.gamma * self.G.mm(self.W_shifted) + self.M - self.M_previous

        # Update G
        self.G += self.C(self.V.unsqueeze(1) - self.G, self.compressor_param)

        # Update parameters
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data = x[offset:offset + numel].view_as(p.data)
            offset += numel




def main():
    # Example usage
    n = 5
    W = create_mixing_matrix(n)
    print("Mixing matrix W:")
    print(W)

    # Verify properties
    print("\nIs symmetric:", np.allclose(W, W.T))
    print("Is doubly stochastic:", np.allclose(W @ np.ones(n), np.ones(n)) and
          np.allclose(np.ones(n) @ W, np.ones(n)))
    eigenvalues = np.linalg.eigvals(W)
    print("Eigenvalues:", sorted(np.abs(eigenvalues), reverse=True))
    print("Spectral gap:", 1 - np.abs(sorted(eigenvalues, key=lambda x: abs(x), reverse=True)[1]))
    print("start simulation")

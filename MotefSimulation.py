import torch

import numpy as np
from model.resnet8 import ResNet8

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

def calculateGradient(model,train_loader):


def motefSimulation(model, num_workers, w):
    # create models for each worker
    models = []
    for i in range(num_workers):
        models.append(ResNet8(10))

    # init tensors
    x = torch.zeros_like(torch.cat([p.data.view(-1) for p in model.parameters()]))
    num_params = x.size(-1)
    x = torch.rand(num_workers, num_params)

    w_shifted = w - torch.eye(num_workers)

    H = torch.zeros_like(x)
    V = grad(x)
    G = torch.zeros_like(x)
    M = torch.zeros_like(x)

    M_previous = np.zeros((p.dim, p.n_agent))




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

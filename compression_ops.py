import numpy as np

import torch


def uniform_quantize(x, num_bits=8):
    """
    Applies uniform quantization to a PyTorch tensor.

    Args:
    x (torch.Tensor): Input tensor to be quantized
    num_bits (int): Number of bits for quantization (default: 8)

    Returns:
    torch.Tensor: Quantized tensor
    """
    # Determine the range of the input
    x_min, x_max = x.min(), x.max()

    # Calculate the step size
    step = (x_max - x_min) / (2 ** num_bits - 1)

    # Quantize the values
    quantized = torch.round((x - x_min) / step)

    # Clip values to ensure they're within range
    quantized = torch.clamp(quantized, 0, 2 ** num_bits - 1)

    # Scale back to the original range
    x_quantized = quantized * step + x_min

    return x_quantized


def random_k_compression(x, alpha):
    """
    Applies random k compression to the input tensor.

    Args:
    tensor (torch.Tensor): Input tensor to be compressed
    k (int): Number of elements to keep

    Returns:
    torch.Tensor: Compressed tensor
    """
    k = int(alpha * x.numel())
    if k >= x.numel():
        return x

    # Flatten the tensor
    flat_tensor = x.flatten()

    # Get the number of elements in the tensor
    n = flat_tensor.numel()
    # Randomly select k indices
    indices = torch.randperm(n)[:k]

    # Create a mask tensor
    mask = torch.zeros_like(flat_tensor)
    mask[indices] = 1.0

    # Apply the mask and scale the values
    compressed_tensor = flat_tensor * mask

    # Reshape the tensor back to its original shape
    return compressed_tensor.reshape(x.shape)


# Top_K_Compressor proposed by stich et al: https://arxiv.org/pdf/1809.07599
# It acts on the input by preserving K largest entries by magnitude while zeroing out the rest
# variables:
#       x the gradient
#       alpha the percentage of entries to keep in the gradient
#
def top_k(x, alpha):
    # Get the top k values and their indices
    top_k_values, top_k_indices = torch.topk(x.abs(), int(alpha * x.numel()))

    # Create a new tensor filled with zeros
    compressed = torch.zeros_like(x)

    # Place the top k values back into their original positions
    compressed[top_k_indices] = x[top_k_indices]
    return compressed


def gsgd(x, b):
    norm = torch.norm(x)
    if norm < 1e-10:
        return x

    delta = np.sqrt(x.shape[0]) / (2 ** (b - 1))
    tau = 1 + delta if delta > 1 else 1 + delta ** 2
    tmp = (2 ** (b - 1)) / norm * torch.abs(x) + torch.randn(x.shape, device=x.device)
    tmp = torch.max(tmp, torch.zeros(1, device=x.device))
    return torch.sign(x) * torch.floor(tmp) * (norm / (2 ** (b - 1)) / tau)

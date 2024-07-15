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


# Example usage
x = torch.randn(5, 5)
quantized_data = uniform_quantize(x, num_bits=6)

print("Original data:", x)
print("Quantized data:", quantized_data)

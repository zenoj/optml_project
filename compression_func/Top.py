import torch


# Top_K_Compressor proposed by stich et al: https://arxiv.org/pdf/1809.07599
# It acts on the input by preserving K largest entries by magnitude while zeroing out the rest
# variables:
#       K the number of entries to keep in the gradient
#       x the gradient
def top_k_compress(x, alpha):
    # Get the top k values and their indices
    top_k_values, top_k_indices = torch.topk(x.abs(), int(alpha*x.numel()))

    # Create a new tensor filled with zeros
    compressed = torch.zeros_like(x)

    # Place the top k values back into their original positions
    compressed[top_k_indices] = x[top_k_indices]
    return compressed

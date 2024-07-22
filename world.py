import networkx as nx
import torch
import torch.distributed as dist


def create_adjacency_matrix(topology, world_size, prob=0.1):
    match topology:
        case "ring":
            G = nx.cycle_graph(world_size)
        case "fully-connected":
            G = nx.complete_graph(world_size)
        case 'star':
            G = nx.star_graph(world_size)
        case 'grid':
            size = int(world_size ** 0.5)
            G = nx.grid_2d_graph(size, size)
            G = nx.convert_node_labels_to_integers(G)
        case 'erdos-renyi':
            G = nx.erdos_renyi_graph(world_size, prob)
        case _:
            raise ValueError(f"Unsupported topology {topology}")

    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    return adjacency_matrix


def communicate_with_neighbors(rank, world_size, q_h_i, q_g_i, adjacency_matrix):
    neighbors = [i for i in range(world_size) if adjacency_matrix[rank][i] != 0]
    device = q_h_i.device

    recv_q_h_i = {neighbor: torch.zeros_like(q_h_i) for neighbor in neighbors}
    recv_q_g_i = {neighbor: torch.zeros_like(q_g_i) for neighbor in neighbors}

    send_requests = []
    for neighbor in neighbors:
        send_requests.append(dist.isend(q_h_i, dst=neighbor))

    for neighbor in neighbors:
        dist.recv(recv_q_h_i[neighbor], src=neighbor)

    for request in send_requests:
        request.wait()  # Ensure all sends are completed

    send_requests = []
    for neighbor in neighbors:
        send_requests.append(dist.isend(q_g_i, dst=neighbor))

    for neighbor in neighbors:
        dist.recv(recv_q_g_i[neighbor], src=neighbor)

    for request in send_requests:
        request.wait()  # Ensure all sends are completed

    dist.barrier()

    return {neighbor: recv_q_h_i[neighbor] for neighbor in neighbors}, {neighbor: recv_q_g_i[neighbor] for neighbor in
                                                                        neighbors}


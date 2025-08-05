def jaccard_index(graph, node_u, node_v):
    neighbors_u = set(nx.neighbors(graph, node_u))
    neighbors_v = set(nx.neighbors(graph, node_v))

    intersection_size = len(neighbors_u.intersection(neighbors_v))
    union_size = len(neighbors_u.union(neighbors_v))
    if union_size == 0:
        return 0
    
    return intersection_size / union_size

def salton_index(graph, node_u, node_v):
    common_neighbors = list(nx.common_neighbors(graph, node_u, node_v))
    if len(common_neighbors) == 0: # To prevent division by 0 errors
        return 0
    
    degree_u = len(list(nx.neighbors(graph, node_u)))
    degree_v = len(list(nx.neighbors(graph, node_v)))

    return len(common_neighbors) / ((degree_u * degree_v) ** 0.5)

def sorensen_index(graph, node_u, node_v):
    common_neighbors = list(nx.common_neighbors(graph, node_u, node_v))
    if len(common_neighbors) == 0: # To prevent division by 0 errors
        return 0
    
    degree_u = len(list(nx.neighbors(graph, node_u)))
    degree_v = len(list(nx.neighbors(graph, node_v)))

    return 2 * len(common_neighbors) / (degree_u + degree_v)


def jaccard_3_paths(graph, node_u, node_v, num):
    neighbors_u = set(nx.neighbors(graph, node_u))
    neighbors_v = set(nx.neighbors(graph, node_v))

    union_size = len(neighbors_u.union(neighbors_v))
    if union_size == 0:
        return 0
    
    return num / union_size

def salton_3_paths(graph, node_u, node_v, num):    
    degree_u = len(list(nx.neighbors(graph, node_u)))
    degree_v = len(list(nx.neighbors(graph, node_v)))
    if degree_u * degree_v == 0:
        return 0

    return num / ((degree_u * degree_v) ** 0.5)

def sorensen_3_paths(graph, node_u, node_v, num):
    degree_u = len(list(nx.neighbors(graph, node_u)))
    degree_v = len(list(nx.neighbors(graph, node_v)))
    if degree_u + degree_v == 0:
        return 0

    return 2 * num / (degree_u + degree_v)

    
def create_array(x, y, max_):
    x, y = int(x), int(y)
    arr = np.zeros(max_)
    if x == y:
        arr[x] = 1
    elif x != y:
        arr[x] = arr[y] = 1
    return arr

def hub_depressed_index(G, u, v):
    if u not in G or v not in G:
        return 0.0
    neighbors_u = set(nx.neighbors(G, u))
    neighbors_v = set(nx.neighbors(G, v))
    cn = len(neighbors_u.intersection(neighbors_v))
    max_deg = max(G.degree[u], G.degree[v])
    return cn / max_deg if max_deg > 0 else 0

def hub_promoted_index(G, u, v):
    if u not in G or v not in G:
        return 0.0
    neighbors_u = set(nx.neighbors(G, u))
    neighbors_v = set(nx.neighbors(G, v))
    cn = len(neighbors_u.intersection(neighbors_v))
    min_deg = min(G.degree[u], G.degree[v])
    return cn / min_deg if min_deg > 0 else 0


def safe_cosine_similarity(f_u, f_v):
    norm_u = np.linalg.norm(f_u)
    norm_v = np.linalg.norm(f_v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0  # or np.nan if you prefer to flag it
    else:
        return 1 - cosine(f_u, f_v)

import torch
from collections import defaultdict

def build_adj_list(edge_index, num_nodes):
    """
    Build adjacency list from edge_index tensor.
    """
    adj_list = defaultdict(set)
    edge_array = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    src_nodes, dst_nodes = edge_array
    for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
        adj_list[src].add(dst)
        adj_list[dst].add(src)  # undirected
    return adj_list

def count_simple_paths_for_edge_sparse(adj_list, u, v):
    """
    Count 2-hop and 3-hop paths using adjacency lists.
    - 2-path: u → x → v
    - 3-path: u → x → y → v
    """
    neighbors_u = adj_list[u] - {v, u}
    neighbors_v = adj_list[v] - {u, v}

    # 2-paths: u - x - v
    count_2paths = sum(1 for x in neighbors_u if v in adj_list[x])

    # 3-paths: u - x - y - v
    count_3paths = 0
    for x in neighbors_u:
        count_3paths += sum(1 for y in adj_list[x] if y in neighbors_v and y != u and y != v)

    return count_2paths, count_3paths


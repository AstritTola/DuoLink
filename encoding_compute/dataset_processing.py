from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor
from torch_geometric.datasets import HeterophilousGraphDataset
import random

def dataset_import(name):

    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='/tmp/'+name, name=name)
        g_type = 'undirected'
    elif name in ['Photo', 'Computers']:
        dataset = Amazon(root='/tmp/'+name, name=name)
        g_type = 'undirected'
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        try:
            dataset = WikipediaNetwork(root='/tmp/'+name, name=name)
        except:
            dataset = WikipediaNetwork(root='/tmp/'+name, name=name)
        g_type = 'directed'
    elif name in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = WebKB(root='/tmp/'+name, name=name)
        g_type = 'directed'
    elif name in ['Actor']:
        dataset = Actor(root='/tmp/'+name)
        g_type = 'directed'
    elif name in ['Roman-empire']:
        dataset = HeterophilousGraphDataset(root='data/', name=name)
        g_type = 'undirected'



    edge_index = dataset.data.edge_index.t().tolist()
    node_index = np.unique(np.array(edge_index)[:,0].tolist()+np.array(edge_index)[:,1].tolist())
    y = dataset[0].y
    Y = y.numpy()
    x = dataset[0].x
    X = x.numpy()

    return edge_index, node_index, Y, g_type, X

# partition the positive set into train/valdation/test sets
def pos_partition(edge_list, g_type):
    print('Positive set partition')
    # Generate a list of indices
    indices = list(range(len(edge_list)))
    random.shuffle(indices)

    # Split the indices
    pos_train_ind = indices[:int(0.85 * len(indices))]
    remaining_indices = indices[int(0.85 * len(indices)):]
    pos_val_ind = remaining_indices[:int(1/3 * len(remaining_indices))]
    pos_test_ind = remaining_indices[int(1/3 * len(remaining_indices)):]

    pos_train = [edge_list[x] for x in pos_train_ind]
    pos_val = [edge_list[x] for x in pos_val_ind]
    pos_test = [edge_list[x] for x in pos_test_ind]

    return pos_train, pos_val, pos_test

# randomly create negative sets
def neg_set(edge_list, node_list, M, g_type):
    print('Negitive set selection')
    edge_set = set(map(tuple, edge_list))  # Convert edge_list to a set for O(1) lookups
    neg_list = []

    while len(neg_list) < M:
        i, j = random.choice(node_list), random.choice(node_list)
        if g_type == 'undirected':
            if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                neg_list.append((i, j))

        elif g_type == 'directed':
            if i != j and (i, j) not in edge_set:
                edge_set.add((i, j))
                neg_list.append((i, j))

    return list(edge_set), neg_list  # Convert back to list if needed

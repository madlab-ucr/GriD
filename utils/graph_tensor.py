import scipy.io
import networkx as nx
import numpy as np
from tqdm import tqdm

import torch

import pickle

from tensorly.contrib.sparse import decomposition
import sparse


def load_graph_network(path):
    
    try:
        data = scipy.io.loadmat(path)
    except:
        print('Invalid data path')

    G = nx.from_scipy_sparse_array(data["Network"])
    # nx.set_node_attributes(G, bc_data["Attributes"], 'Attributes')
    print(str(G))

    # convert list of lists to list
    labels = [j for i in data["Label"] for j in i]

    # Add labels to each node
    for i in range(len(G.nodes)):
        G.nodes[i]['Anomaly'] = labels[i]

    is_undirected = not nx.is_directed(G)

    # G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
    # G = nx.convert_node_labels_to_integers(G)

    ego_gs, roots = [], []

    # if 0-degree node(s), remove label(s) from consideration
    if len(labels) != G.number_of_nodes():
        labels = list(nx.get_node_attributes(G, 'Anomaly').values())

    for i in tqdm(range(G.number_of_nodes())):
        roots.append(G.nodes[i]['Anomaly'])
        G_ego = nx.ego_graph(G, i, radius=1, undirected=is_undirected)
        if G_ego.number_of_nodes() >= 2:
            ego_gs.append(G_ego)

    return G, ego_gs, roots, labels


def load_network_menu():
    load = input('Load previous loaded network? (y/n): ')
    if load.lower()[0] == 'n':

        dataset = int(input('Enter dataset/network path: \n\t (1) BlogCatalog \n\t (2) Flickr \n\t (3) ACM\n'))

        if dataset == 1: 
            data_path = 'datasets/blogcatalog.mat'
            dataset = 'bc'
        elif dataset == 2: 
            data_path = 'datasets/Flickr.mat'
            dataset = 'flr'
        elif dataset == 3: 
            data_path = 'datasets/ACM.mat'
            dataset = 'acm'
        
        G, ego_gs, roots, labels = load_graph_network(data_path)

        path = f'{dataset}_data.sav'

        saved_model = open(path, 'wb')
        pickle.dump((G, ego_gs, roots, labels), saved_model)
        saved_model.close()

    else:
        data_path = input('Enter file path for network: ')
        saved_model = open(data_path , 'rb')
        G, ego_gs, roots, labels = pickle.load(saved_model)
        saved_model.close()
        roots = [int(r) for r in roots]
        dataset = data_path.split('_')[0]

    return G, ego_gs, roots, labels, dataset


def build_graph_tensor():

    G, ego_gs, _, labels, dataset = load_network_menu()

    N = G.number_of_nodes()

    # %%
    indices = []
    padded_gs = []

    undirected = not nx.is_directed(G)

    for i, g in enumerate(tqdm(ego_gs)):
        ego_adj_list = dict(g.adjacency())
        
        result = np.zeros((N, N))
        for node in ego_adj_list.keys():    
            for neighbor in ego_adj_list[node].keys():

                result[node][neighbor] = 1
                if undirected:
                    result[neighbor][node] = 1
                indices.append([i, node, neighbor])
                indices.append([i, neighbor, node])
                
        padded_gs.append(result)

# %%
    print('\nConstructing Tensor...')
    i = torch.tensor(list(zip(*indices)))
    values = torch.ones(len(indices))

    cube = sparse.COO(i, data=values)

    return cube, padded_gs, labels, dataset
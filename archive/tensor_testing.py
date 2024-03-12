# %%
import numpy as np

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from tqdm import tqdm

import networkx as nx

# %%
with open(input('Enter saved dataset file path: '), 'rb') as f:
    G, ego_gs, roots, labels = pickle.load(f)

N = G.number_of_nodes()

# %%
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
            
    padded_gs.append(result)

# %%
factor_path = input('\nEnter file path for factors: ')

with open(factor_path, 'rb') as f:
    factors = pickle.load(f)

# %%
# tensor_path = input('Enter file path for tensor: ')

# with open(tensor_path, 'rb') as f:
#     cube = pickle.load(f)

# # %%
# # generate random egonets
# random_gs = []

# for i in tqdm(range(N)):
#     gs_r = np.random.randint(0, 2, size=(N, N))
#     random_gs.append(gs_r)

# %%
A, B, C = factors

decomp = factor_path.split('_')[1]

# %%
# errors = []
# print("\nCalculating Reconstruction Errors...")
# for gs in tqdm(padded_gs):
#     if decomp == 'tkd':
#         # projection
#         gs_p = ((A.T @ gs) @ B)
#         # reconstruction
#         gs_r = (A @ gs_p @ B.T)
#     elif decomp == 'cpd':
#         # projection
#         gs_p = ((np.linalg.pinv(A) @ gs) @ B)
#         # reconstruction
#         gs_r = (A @ gs_p @ np.linalg.pinv(B))
#     d = np.linalg.norm(gs - gs_r, ord='fro')

#     # # absolute error
#     # errors.append(d / np.linalg.norm())

#     # relative error
#     errors.append(d / np.linalg.norm(gs, ord='fro'))

# errors = np.array(errors).reshape(-1, 1)

# %%
random_errors = []
print("\nCalculating Reconstruction Errors For Random Graphs...")
for _ in tqdm(range(N)):
    gs = np.random.randint(0, 2, size=(N, N))
    if decomp == 'tkd':
        # projection
        gs_p = ((A.T @ gs) @ B)
        # reconstruction
        gs_r = (A @ gs_p @ B.T)
    elif decomp == 'cpd':
        # projection
        gs_p = ((np.linalg.pinv(A) @ gs) @ B)
        # reconstruction
        gs_r = (A @ gs_p @ np.linalg.pinv(B))
    d = np.linalg.norm(gs - gs_r, ord='fro')

    # # absolute error
    # errors.append(d / np.linalg.norm())

    # relative error
    random_errors.append(d / np.linalg.norm(gs, ord='fro'))

random_errors = np.array(random_errors).reshape(-1, 1)

# %%
dataset = factor_path.split('_')[0]
rank = factor_path.split('_')[2].split('.')[0]
path = f'{dataset}_{decomp}_{rank}_random.sav'

saved_errors = open(path, 'wb')
pickle.dump(random_errors, saved_errors)
saved_errors.close()

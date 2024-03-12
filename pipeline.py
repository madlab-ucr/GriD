import torch
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os

from tensorly.decomposition import tucker, constrained_parafac

from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorly.contrib.sparse import decomposition
import sparse

from pyod.models.lof import LOF

from utils.graph_tensor import build_graph_tensor
from utils.text_tensor import build_text_tensor

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    ## Load Tensor for Decomposition
    tensor_option = int(
        input("Select an option: \n\t (1) Graph Tensor \n\t (2) Text Tensor\n")
    )

    if tensor_option == 1:
        tensor, padded_slices, labels, dataset = build_graph_tensor()
        ten_type = "graph-ten"
    elif tensor_option == 2:
        tensor, padded_slices, labels, dataset = build_text_tensor()
        ten_type = "text-ten"

    ranks = [int(r) for r in input("\nEnter ranks, space separated: ").split()]

    scores = []
    for rank in ranks:
        print(f"\nUSING RANK {rank}\n")
        load = input("\nLoad Reconstruction Errors? (y/n): ")
        # not loading previously calculated reconstruction errors
        if load.lower()[0] == "n":

            # checking for valid input
            load = input("\nLoad Previous Decomposition? (y/n): ")
            while load.lower()[0] != "n" and load.lower()[0] != "y":
                print("Invalid Input!")
                load = input("Load Previous Decomposition? (y/n): ")
            decomp = input("Select Tucker (1) or CP (2) Decomposition: ")
            while decomp != "1" and decomp != "2":
                print("Invalid Input!")
                decomp = input("Select Tucker (1) or CP (2) Decomposition: ")

            decomp_alg = "tkd" if decomp == "1" else "cpd"

            if load.lower()[0] == "n":
                if not os.path.exists("decomposition_results/factors"):
                    os.makedirs("decomposition_results/factors")
                path = (
                    f"decomposition_results/factors/{dataset}_{decomp_alg}_r{rank}.sav"
                )
                # path = input('Enter file name to save factors as: ')
                if decomp == "1":
                    print("Tucker Decomposition...")
                    _, factors = decomposition.tucker(tensor, rank=rank, init="random")
                elif decomp == "2":
                    print("Parafac Decomposition...")
                    _, factors = decomposition.parafac(tensor, rank=rank, init="random")
                print(f"Factors Saved to {path}\n")
                saved_model = open(path, "wb")
                pickle.dump(factors, saved_model)
                saved_model.close()
            else:
                with open(input("Enter file path: "), "rb") as f:
                    factors = pickle.load(f)
                    f.close()
                    print()

            A, B, C = factors
            if decomp == "1":
                (
                    A,
                    B,
                    C,
                ) = (
                    np.array(A),
                    np.array(B),
                    np.array(C),
                )
            elif decomp == "2":
                A, B, C = A.todense(), B.todense(), C.todense()

            # path = input('Enter file name to save reconstruction errors: ')
            if not os.path.exists("decomposition_results/errors"):
                os.makedirs("decomposition_results/errors")
            path = f"decomposition_results/errors/{dataset}_{ten_type}_{decomp_alg}_r{rank}_errors.sav"

            errors = []
            print("Calculating Reconstruction Errors...")
            # Assuming a shape of M, N, N or N, N, N where dimensions 2 and 3 represent each slice
            for i in tqdm(range(tensor.shape[0])):
                gs = tensor[i, :, :].todense()
                if decomp == "1":
                    # projection
                    gs_p = (A.T @ gs) @ B
                    # reconstruction
                    gs_r = A @ gs_p @ B.T
                elif decomp == "2":
                    # projection
                    gs_p = (np.linalg.pinv(A) @ gs) @ B
                    # reconstruction
                    gs_r = A @ gs_p @ np.linalg.pinv(B)
                d = np.linalg.norm(gs - gs_r, ord="fro")

                # # absolute error
                # errors.append(d / np.linalg.norm())

                # relative error
                errors.append(d / np.linalg.norm(gs, ord="fro"))

            # Replace NaN values with 0
            np.nan_to_num(errors, copy=False, nan=0)
            errors = np.array(errors).reshape(-1, 1)

            saved_model = open(path, "wb")
            pickle.dump(errors, saved_model)
            saved_model.close()
            print()

        # loading previously calculated reconstruction errors
        else:
            with open(input("Enter file path: "), "rb") as f:
                errors = pickle.load(f)
                f.close()
                print()

        scale = MinMaxScaler()
        embeddings = scale.fit_transform(np.array(errors))

        scores.append(("No Model", rank, roc_auc_score(labels, embeddings)))

        ### pyOD Models

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.33, random_state=42
        )

        clf = LOF()
        clf.fit(X_train, y_train)

        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

        scores.append(("\nLOF", rank, roc_auc_score(y_test, y_test_scores)))

    for name, rank, auc in scores:
        print(f"Model: {name}, Rank: {rank}, AUC score: {auc}")

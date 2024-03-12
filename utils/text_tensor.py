import json
import random

import json
import zstandard as zstd

from collections import Counter

import torch
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sparse

import pandas as pd
import re

from scipy.io import savemat
import os

import time

def load_text_dataset_from_csv(path, subset_size=None, train_size=None):
    data = pd.read_csv(path)
    
    dataset = data.to_numpy()

    data, labels = dataset[:, 0], dataset[:, 1]
    data, labels = shuffle(data, labels, random_state=10)

    total_data, total_labels = data, labels
    train_data = test_data = train_labels = test_labels = None

    if subset_size:
        print("Grabbing dataset of size:", subset_size)
        data, labels = data[:len(data)*subset_size], labels[:len(labels)*subset_size]

    if train_size:
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_size)
        
        # Remove GPT data from training data
        data = [x for x, label in zip(train_data, train_labels) if label != 1]
        labels = [0] * len(data)

        # Add unused GPT data from training set to test set
        gpt_data = [x for x, label in zip(train_data, train_labels) if label == 1]
        test_data = np.append(test_data, gpt_data)
        test_labels = np.append(test_labels, [1] * len(gpt_data))

    # Initialize an empty dictionary to store term indices
    term_indices = {}

    for idx, document in enumerate(data):
        try:
            # for term in document.split():
            for term in re.findall(r"(\w+|[^\w\s])", document):
                if term not in term_indices:
                    # Assign the next available index to the term
                    term_indices[term] = len(term_indices)
        except:
            print(f'Failed to process document #{idx}')
            continue

    # Get the number of unique terms
    num_unique_terms = len(term_indices)

    return (total_data, total_labels), (data, labels), (test_data, test_labels), num_unique_terms, term_indices


def load_text_dataset_from_json(path):

    try:
        # Load the compressed JSON file
        with open(path, 'rb') as f:
            compressed_data = f.read()
    except:
        print('Invalid data path')

    # Decompress the data
    decompressed_data = zstd.decompress(compressed_data)

    # Decode the JSON content
    json_data = decompressed_data.decode('utf-8')

    # Parse the JSON data
    posts = json.loads(json_data)

    # Extract comments and GPT responses from each post
    data = []
    labels = []

    for post in posts:
        comments = [comment['body'] for comment in post['comments']]
        gpt_response = post['gpt']
        
        # Append comments and GPT response to the data list
        data.extend(comments)
        data.append(gpt_response)
        
        # Assign labels to indicate whether the content is chat GPT generated or not
        labels.extend([0] * len(comments))
        labels.append(1)

    # Shuffle the data and labels together
    data, labels = shuffle(data, labels, random_state=10)

    # # Print the shuffled data and labels
    # for d, label in zip(data, labels):
    #     print(f'Label: {label}, Content: {d}')

    # Count the occurrences of all terms in the dataset
    term_counts = Counter()

    # Initialize an empty dictionary to store term indices
    term_indices = {}

    for document in data:
        for term in re.findall(r"(\w+|[^\w\s])", document):
            if term not in term_indices:
                # Assign the next available index to the term
                term_indices[term] = len(term_indices)

    # Get the number of unique terms
    num_unique_terms = len(term_indices)

    return data, labels, num_unique_terms, term_indices


def build_text_tensor(window_size=5):

    option = input('Load from: \n\t (1) CSV\n\t (2) JSON\n')

    if option == '1':
        data, labels, total_num_terms, term_indices = load_text_dataset_from_csv(input('Enter CSV data path: '))
    elif option == '2':
        # Get data and labels from load_dataset()
        data, labels, total_num_terms, term_indices = load_text_dataset_from_json(input('Enter JSON data path: '))

    indices = []
    padded_slices = []

    # Loop over each document in the data
    for doc_idx, document in enumerate(data):

        terms = re.findall(r"(\w+|[^\w\s])", document)

        slice = np.zeros((total_num_terms, total_num_terms))
        # Loop over each of the terms
        for term_idx, term1 in enumerate(terms):

            # Loop over terms within window from i
            for term2 in terms[term_idx + 1: term_idx + window_size]:

                # Append indices of co-occurrence terms
                indices.append([doc_idx, term_indices[term1], term_indices[term2]])
                indices.append([doc_idx, term_indices[term2], term_indices[term1]])

                slice[term_indices[term1], term_indices[term2]] += 1
                slice[term_indices[term2], term_indices[term1]] += 1
            
        padded_slices.append(slice)

    i = torch.tensor(list(zip(*indices)))
    values = torch.ones(len(indices))

    tensor = sparse.COO(i, data=values)

    return tensor, padded_slices, labels, 'reddit'


def get_text_tensor_indices(window_size=5, use_gpt=True):

    option = input('Load from: \n\t (1) CSV\n\t (2) JSON\n')

    if option == '1':
        (total_data, total_labels), (data, labels), (test_data, test_labels), total_num_terms, term_indices = load_text_dataset_from_csv(input('Enter CSV data path: '), train_size=0.9)
    elif option == '2':
        # Get data and labels from load_dataset()
        data, labels, total_num_terms, term_indices = load_text_dataset_from_json(input('Enter JSON data path: '))

    # If false, do not use gpt data to build training tensor
    if use_gpt == False:
        # Gather slice data for test responses
        test_tensor_indices = []
        test_tensor_size = (len(test_data), total_num_terms, total_num_terms)

        print('Gathering co-occurance indices for test data...')
        for doc_idx, document in enumerate(test_data):

            try:
                terms = re.findall(r"(\w+|[^\w\s])", document)
            except:
                print(f'Failed to process document #{doc_idx}')
                continue

            # Loop over each of the terms
            for term_idx, term1 in enumerate(terms):

                # Loop over terms within window from i
                for term2 in terms[term_idx + 1: term_idx + window_size]:

                    # Append indices of co-occurrence terms
                    if all(term in term_indices for term in (term1, term2)):
                        test_tensor_indices.append([doc_idx + 1, term_indices[term1] + 1, term_indices[term2] + 1])
                        test_tensor_indices.append([doc_idx + 1, term_indices[term2] + 1, term_indices[term1] + 1])
        
        print('Test Tensor Size:', test_tensor_size)
        # Create the directory if it doesn't exist
        if not os.path.exists('tensor_data/'):
            os.makedirs('tensor_data/')
        savemat(f'tensor_data/reddit_test_tensor_data.mat', {'indices': np.array(test_tensor_indices), 'size': test_tensor_size})


    indices = []
    tensor_size = (len(data), total_num_terms, total_num_terms)

    # Loop over each document in the data
    for doc_idx, document in enumerate(data):

        try:
            terms = re.findall(r"(\w+|[^\w\s])", document)
        except:
            print(f'Failed to process document #{doc_idx}')
            continue

        # Loop over each of the terms
        for term_idx, term1 in enumerate(terms):

            # Loop over terms within window from i
            for term2 in terms[term_idx + 1: term_idx + window_size]:

                # Append indices of co-occurrence terms
                indices.append([doc_idx + 1, term_indices[term1] + 1, term_indices[term2] + 1])
                indices.append([doc_idx + 1, term_indices[term2] + 1, term_indices[term1] + 1])

    return (data, labels), indices, tensor_size, 'reddit', term_indices

if __name__ == '__main__':

    # tensor, slices, labels = build_text_tensor()

    # print('Tensor Shape:', tensor.shape)
    # print('Slices Shape:', f'({len(slices)}, {slices[0].shape[0]}, {slices[0].shape[1]})')
    # print('Labels Shape:', len(labels))

    dataset, indices, tensor_size, dataset_name, _ = get_text_tensor_indices(use_gpt=False)

    i = np.array(indices)
    # values = np.ones(len(indices))

    print("Tensor Size:", tensor_size)

    # Create the directory if it doesn't exist
    if not os.path.exists('tensor_data/'):
        os.makedirs('tensor_data/')

    # Save indices and values to text files
    savemat(f'tensor_data/{dataset_name}_tensor_data_nogpt.mat', {'indices': i, 'size':tensor_size})


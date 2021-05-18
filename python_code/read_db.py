#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np


class MnistiidDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open("data/MNIST/" + file_path, "rb") as pickle_file:
            self.dataset = pickle.load(pickle_file)[k]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # 3D input 1x28x28
        sample_x = torch.Tensor([np.array(self.dataset[idx][0])])

        sample_y = torch.ones((1,), dtype=torch.long)
        sample_y = sample_y.new_tensor(self.dataset[idx][1])

        return sample_x, sample_y


def clients_set_MNIST_iid(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""

    print("data/MNIST/" + file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistiidDataset(file_name, k)
        dataset_dl = DataLoader(dataset_object, batch_size=batch_size, shuffle=shuffle)
        list_dl.append(dataset_dl)

    return list_dl


class MnistShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open("data/MNIST/" + file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]])

        y = torch.ones((1,), dtype=torch.long)
        y = y.new_tensor(self.labels[idx])

        return x, y


def clients_set_MNIST_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""
    print("data/MNIST/" + file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(dataset_object, batch_size=batch_size, shuffle=shuffle)
        list_dl.append(dataset_dl)

    return list_dl


class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        dataset = pickle.load(open("data/CIFAR-10/" + file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1)
        y = self.y[idx]

        return x, y


def clients_set_CIFAR(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""
    print("data/CIFAR-10/" + file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)
        dataset_dl = DataLoader(dataset_object, batch_size=batch_size, shuffle=shuffle)
        list_dl.append(dataset_dl)

    return list_dl


class ShakespeareDataset(Dataset):
    """Convert the Shakespeare pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open("data/shakespeare/" + file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = dataset[0][k]
            self.labels = dataset[1][k]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        x = torch.Tensor(self.features[idx]).long()

        y = torch.ones((1,), dtype=torch.long)
        y = y.new_tensor(self.labels[idx][0])

        return x, y


def clients_set_shakespeare(
    file_name: str, n_clients: int, batch_size=100, shuffle=True
):
    """Download for all the clients their respective dataset"""

    print("data/shakespeare/" + file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = ShakespeareDataset(file_name, k)
        dataset_dl = DataLoader(dataset_object, batch_size=batch_size)
        list_dl.append(dataset_dl)

    return list_dl


def download_dataset(
    dataset,
    n_clients,
    samples_clients_training=0,
    samples_clients_test=0,
    batch_size=100,
    shuffle=True,
):
    """Download for all the clients their respective dataset and convert
    those PyTorch Dataset object ad PyTorch DataLoader objects that can be read
    by models."""

    if dataset == "MNIST-iid":
        train_path = f"MNIST_iid_train_{n_clients}_{samples_clients_training}.pkl"
        training_dls = clients_set_MNIST_iid(train_path, n_clients)

        test_path = f"MNIST_iid_test_{n_clients}_{samples_clients_test}.pkl"
        testing_dls = clients_set_MNIST_iid(test_path, n_clients)

        fr_samples = samples_clients_training

    elif dataset == "MNIST-shard":
        samples_per_shard_train = int(samples_clients_training // 4)
        train_path = f"MNIST_shard_train_{n_clients}_{samples_per_shard_train}.pkl"
        training_dls = clients_set_MNIST_shard(train_path, n_clients)

        samples_per_shard_test = int(samples_clients_test // 4)
        test_path = f"MNIST_shard_test_{n_clients}_{samples_per_shard_test}.pkl"
        testing_dls = clients_set_MNIST_shard(test_path, n_clients)

        fr_samples = samples_clients_training

    elif dataset == "CIFAR-10":
        train_path = f"CIFAR_iid_train_{n_clients}_{samples_clients_training}.pkl"
        training_dls = clients_set_CIFAR(train_path, n_clients)

        test_path = f"CIFAR_iid_test_{n_clients}_{samples_clients_test}.pkl"
        testing_dls = clients_set_CIFAR(test_path, n_clients)

        fr_samples = samples_clients_training

    elif dataset == "shakespeare":
        train_path = "Shakespeare_train.pkl"
        test_path = "Shakespeare_test.pkl"
        training_dls = clients_set_shakespeare(train_path, n_clients)
        testing_dls = clients_set_shakespeare(test_path, n_clients)

        fr_samples = int(np.mean([len(dl.dataset) for dl in training_dls]))

    return training_dls, testing_dls, fr_samples

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os

"""
-------------
MNIST non-iid
-------------
"""


def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.train_labels[row] == digit:
            shard.append(ds.train_data[row].numpy())
        row += 1

    return row, shard


def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test):

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                MNIST_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = "./data/"
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


def create_MNIST_small_niid(
    n_clients: int,
    samples_train: list,
    samples_test: list,
    clients_digits: list,
):

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    X_train, X_test = [], []
    y_train, y_test = [], []

    for digits, n_train, n_test in zip(
        clients_digits, samples_train, samples_test
    ):

        client_samples_train, client_samples_test = [], []
        client_labels_train, client_labels_test = [], []

        n_train_per_shard = int(n_train / len(digits))
        n_test_per_shard = int(n_test / len(digits))

        for digit in digits:

            row_train, row_test = 0, 0
            _, shard_train = get_1shard(
                MNIST_train, row_train, digit, n_train_per_shard
            )
            _, shard_test = get_1shard(
                MNIST_test, row_test, digit, n_test_per_shard
            )

            client_samples_train += shard_train
            client_samples_test += shard_test

            client_labels_train += [digit] * n_train_per_shard
            client_labels_test += [digit] * n_test_per_shard

        X_train.append(client_samples_train)
        X_test.append(client_samples_test)

        y_train.append(client_labels_train)
        y_test.append(client_labels_test)

    folder = "./data/"
    train_path = f"MNIST_small_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((np.array(X_train), np.array(y_train)), output)

    test_path = f"MNIST_small_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((np.array(X_test), np.array(y_test)), output)


class MnistShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
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
        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y


def clients_set_MNIST_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl


"""
-------
CIFAR 10
Dirichilet distribution
----
"""


def partition_CIFAR_dataset(
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    if balanced:
        n_samples = [500] * n_clients
    elif not balanced and train:
        n_samples = (
            [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        )
    elif not balanced and not train:
        n_samples = [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10

    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


def create_CIFAR10_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    matrix = dirichlet([alpha] * n_classes, size=n_clients)

    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


def clients_set_CIFAR(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


"""
---------
Upload any dataset
Puts all the function above together
---------
"""


def get_dataloaders(dataset, batch_size: int, shuffle=True):

    folder = "./data/"

    if dataset == "MNIST_iid":

        n_clients = 100
        samples_train, samples_test = 600, 100

        mnist_trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_train_split = torch.utils.data.random_split(
            mnist_trainset, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_train_split
        ]

        mnist_testset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test_split = torch.utils.data.random_split(
            mnist_testset, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_test_split
        ]

    elif dataset == "MNIST_shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            create_MNIST_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    elif dataset == "CIFAR10_iid":
        n_clients = 100
        samples_train, samples_test = 500, 100

        CIFAR10_train = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_train_split = torch.utils.data.random_split(
            CIFAR10_train, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_train_split
        ]

        CIFAR10_test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_test_split = torch.utils.data.random_split(
            CIFAR10_test, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_test_split
        ]

    elif dataset[:5] == "CIFAR":

        n_classes = 10
        n_clients = 100
        balanced = dataset[8:12] == "bbal"
        alpha = float(dataset[13:])

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_CIFAR10_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )

    # Save in a file the number of samples owned per client
    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    return list_dls_train, list_dls_test

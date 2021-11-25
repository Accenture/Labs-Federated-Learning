#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import json
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

"""
-----------
CIFAR10 Dataset
-----------
"""


class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.features = torch.Tensor(dataset[0][k])
        self.features = self.features.permute(0, 3, 1, 2) / 255
        self.features = (self.features - 0.5) / 0.5

        self.labels = torch.tensor(dataset[1][k]).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        # x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        # x = (x - 0.5) / 0.5
        # y = self.y[idx]

        X = self.features[idx]
        y = self.labels[idx]
        return X, y


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
        n_samples = [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
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


def clients_set_CIFAR(file_name: str, n_clients: int, batch_size: int, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(dataset_object, batch_size=batch_size, shuffle=shuffle)

        list_dl.append(dataset_dl)

    return list_dl


"""
-----------
Shakespeare Dataset
-----------
"""


def conversion_string_to_vector(sentence):

    all_characters = string.printable

    vector = [all_characters.index(sentence[c]) for c in range(len(sentence))]

    return vector


class Shakespearedataset(Dataset):
    """convert the shakespeare pkl file into a pytorch dataset"""

    def __init__(self, data, user):

        strings = data["user_data"][user]["x"]
        vectors = [conversion_string_to_vector(s) for s in strings]
        self.features = torch.tensor(vectors).long()

        label_strings = data["user_data"][user]["y"]
        ints = [conversion_string_to_vector(s) for s in label_strings]
        self.labels = torch.tensor(ints).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        x = self.features[idx]
        y = self.labels[idx]

        return x, y


def clients_set_Shakespeare(path, batch_size: int, shuffle: bool):
    """Download for all the clients their respective dataset"""

    list_json = os.listdir(path)
    list_json = sorted(list_json)

    list_dls = []

    for js in list_json:

        with open(path + js) as json_file:
            data = json.load(json_file)

        users = data["users"]

        for user in users:
            dataset = Shakespearedataset(data, user)

            dataloader = DataLoader(dataset, batch_size, shuffle)

            list_dls.append(dataloader)

    return list_dls


"""
---------
Upload any dataset
Puts all the function above together
---------
"""


def get_dataloaders(dataset, batch_size: int, shuffle=True):

    #    folder = "./data/"

    torch.manual_seed(0)

    if dataset[:11] == "Shakespeare":

        rep_path = os.getcwd()
        os.chdir("data/leaf/data/shakespeare")
        try:
            os.system("bash ./preprocess.sh -s niid --sf 1 -k 0 -t sample --tf 0.8")
            print("Shakespeare dataset created")
        except BaseException:
            print("Shakespeare dataset already exists")
        os.chdir(rep_path)

        path_train = "data/leaf/data/shakespeare/data/train/"
        list_dls_train = clients_set_Shakespeare(path_train, batch_size, True)
        print("Number of clients in Shakespeare: ", len(list_dls_train))

        path_test = "data/leaf/data/shakespeare/data/test/"

        list_dls_test = clients_set_Shakespeare(path_test, batch_size, True)

        if len(dataset) == 11:
            n_clients = 80
        else:
            mode = int(dataset[11:])

            if mode == 2:
                n_clients = 40
            elif mode == 3:
                n_clients = 20
            elif mode == 4:
                n_clients = 10

        list_dls_train = list_dls_train[:n_clients]
        list_dls_test = list_dls_test[:n_clients]

    elif dataset[:7] == "CIFAR10":

        folder = "data/"

        n_classes = 10
        n_clients = 100
        balanced = False
        alpha = dataset[8:]

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_CIFAR10_dirichlet(dataset, balanced, alpha, n_clients, n_classes)

        list_dls_train = clients_set_CIFAR(path_train, n_clients, batch_size, True)

        list_dls_test = clients_set_CIFAR(path_test, n_clients, batch_size, True)

    # Save in a file the number of samples owned per client
    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    print("participating clients:", len(list_len))

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    return list_dls_train, list_dls_test

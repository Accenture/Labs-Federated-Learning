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
from numpy.random import dirichlet
from PIL import Image


"""
-------------
DIRICHLET
-------------
"""


def backdoor(img: torch.Tensor, k):
    np.random.seed(42 + k)

    for _ in range(10):
        pixel = tuple([np.random.randint(dim_i) for dim_i in img.shape])
        img[pixel] = np.max(img)

    return img

def create_pkl_files_dirichlet(
    dataset_name: str,
    matrix: np.array,
    n_clients: int,
    backdoored: bool,
    train: bool,
):

    name = dataset_name.split("_")[0]
    if name == "MNIST":
        db = datasets.MNIST(root='./data', train=train, download=True)
    elif name == "FashionMNIST":
        db = datasets.FashionMNIST(root='./data', train=train, download=True)
    elif name.split("-")[0] == "CIFAR10":
        db = datasets.CIFAR10(root='./data', train=train, download=True)
    elif name.split("-")[0] == "CIFAR100":
        db = datasets.CIFAR100(root='./data', train=train, download=True)

    save_name = f"{dataset_name}_M{n_clients}"
    if train:
        save_name += "_train"
    else:
        save_name += "_test"

    n_samples = len(db) // n_clients
    n_classes = max(db.targets) + 1

    # INDICES OF EACH CLASS,
    idx_classes = [
        np.where(np.array(db.targets) == k)[0] for k in range(n_classes)
    ]

    list_X, list_y = [], []

    for i in range(n_clients):

        # AMOUNT OF SAMPLES FOR EACH CLASS
        samples_repart_i = (matrix[i] * n_samples).astype(int)
        samples_repart_i[-1] = n_samples - np.sum(samples_repart_i[:-1])

        # GET THE INDICES OF THE `client` DATA POINTS
        data_points_idx_i = np.concatenate(
            [
                np.random.choice(class_k_idx, n_samples)
                for class_k_idx, n_samples in zip(idx_classes, samples_repart_i)
            ]
        ).astype(int)

        # GIVE THE 'client' ITS FEATURES AND LABELS
        if not backdoored:
            list_X += [np.array([np.array(db.data[idx])
                                 for idx in data_points_idx_i])]
        elif backdoored:
            list_X += [np.array([backdoor(np.array(db.data[idx]), i%10)
                                 for idx in data_points_idx_i])]

        list_y += [np.array([db.targets[idx] for idx in data_points_idx_i])]

    # Save clients dataset
    with open(f"data/{save_name}.pkl", "wb") as output:
        pickle.dump((list_X, list_y), output)


def create_ds_with_dirichlet(
        dataset_name: str, alpha: float, n_clients: int, backdoored: bool):
    """Create partitions of CIFAR according to Dir(alpha)"""

    if dataset_name.split("_")[0][:8] == "CIFAR100":
        n_classes = 100
    else:
        n_classes = 10

    if alpha == 0:
        matrix = np.concatenate(
            [np.identity(n_classes) for _ in range(n_clients // n_classes + 1)]
        )[:n_clients]
    elif alpha > 0.:
        np.random.seed(42)
        matrix = dirichlet([alpha] * n_classes, size=n_clients)



    create_pkl_files_dirichlet(
        dataset_name, matrix, n_clients, backdoored, train=True
    )
    create_pkl_files_dirichlet(
        dataset_name, matrix, n_clients, backdoored, train=False
    )


class DatasetDirichlet(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, dataset_name: str, file_path: str, k: int,
                 transform: transforms):

        dataset = pickle.load(open(file_path, "rb"))

        if dataset_name[:7] == "CIFAR10":
            self.features = [Image.fromarray(arr) for arr in dataset[0][k]]
        else:
            self.features = dataset[0][k]

        self.labels = torch.Tensor(dataset[1][k]).long()

        self.transformed_features = [transform(X) for X in self.features]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.transformed_features[idx], self.labels[idx]


def dataloaders_dirichlet(
    path: str, dataset_name: str, n_clients: int,
    batch_size: int, shuffle: bool, transf: transforms
):
    """Download for all the clients their respective dataset"""

    dls = []
    for k in range(n_clients):
        dataset = DatasetDirichlet(dataset_name, path, k, transf)
        dataloader = DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle
        )
        dls.append(dataloader)
    return dls

def train_test_dirichlet(
        dataset_name: str, n_clients: int,
        batch_size: int, shuffle: bool, transf: transforms
):
    """Download for all the clients their respective dataset"""

    path_train = f"./data/{dataset_name}_M{n_clients}_train.pkl"
    print(path_train)
    dls_train = dataloaders_dirichlet(path_train, dataset_name,
                                      n_clients, batch_size, shuffle, transf)

    path_test = f"./data/{dataset_name}_M{n_clients}_test.pkl"
    dls_test = dataloaders_dirichlet(path_test, dataset_name,
                                      n_clients, batch_size, shuffle, transf)

    return dls_train, dls_test


"""
-----------
Shakespeare Dataset
-----------
"""


def conversion_string_to_vector(sentence):
    all_characters = string.printable
    return [all_characters.index(sentence[c]) for c in range(len(sentence))]


class Shakespearedataset(Dataset):
    """convert the shakespeare pkl file into a pytorch dataset"""

    def __init__(self, data, user):

        strings = data["user_data"][user]["x"]
        vectors = [conversion_string_to_vector(s) for s in strings]
        self.features = torch.tensor(vectors).long()

        label_strings = data["user_data"][user]["y"]
        ints = [conversion_string_to_vector(s) for s in label_strings]
        self.labels = torch.tensor(ints).long().view(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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
UPLOAD DATASET
"""
def get_dataloaders(dataset_name: str, batch_size: int, n_clients: int,
                    verbose = True) -> (list, list):

    print("Participating clients:", n_clients)

    # Reproductibility in the creation of the datasets
    np.random.seed(0)
    torch.manual_seed(0)

    if dataset_name.split("_")[0] in \
            ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR10-flat", "CIFAR10-flat-full",
             "CIFAR10-ResNet", "CIFAR100", "CIFAR100-flat"]:

        dataset, alpha = dataset_name.split("_")
        print(f"dataset: {dataset} with Dirichlet parameter: {alpha}")

        path_train = f"data/{dataset_name}_M{n_clients}_train.pkl"
        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_ds_with_dirichlet(dataset_name, float(alpha), n_clients, False)

        sequential = [transforms.ToTensor()]
        if dataset == "MNIST":
            sequential.append(transforms.Normalize(0.1307, 0.3081))

        elif dataset == "CIFAR100":
            sequential.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            )

        elif dataset == "CIFAR100-flat":
            sequential.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            )
            sequential.append(transforms.Resize(size=(16, 16)))

        elif dataset in ["CIFAR10", "CIFAR10-flat-full"]:
            sequential.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            )

        elif dataset == "CIFAR10-flat":
            sequential.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            )
            sequential.append(transforms.Resize(size=(16, 16)))

        elif dataset == "FashionMNIST":
            sequential.append(transforms.Normalize((0.5,), (0.5,)))

        transf = transforms.Compose(sequential)

        dls_train, dls_test = train_test_dirichlet(
            dataset_name, n_clients, batch_size, True, transf
        )

    if dataset_name == "Shakespeare":

        path_train = "data/leaf/data/shakespeare/data/train/"
        path_test = "data/leaf/data/shakespeare/data/test/"

        if not os.path.isdir(path_train):
            os.chdir("data/leaf/data/shakespeare")
            os.system(
                "bash ./preprocess.sh -s niid --sf 0.2 -k 0 -t sample --tf 0.8 --smplseed 0 --spltseed 0"
            )
            print("Shakespeare dataset created")
            os.chdir("../../../../")
        else:
            print("Shakespeare dataset already created")

        dls_train_all = clients_set_Shakespeare(path_train, batch_size, True)
        dls_test_all = clients_set_Shakespeare(path_test, batch_size, True)

        dls_train =dls_train_all[:n_clients]
        dls_test =dls_test_all[:n_clients]
        print("Number of clients in Shakespeare: ", len(dls_train))

    list_len = [len(dl.dataset) for dl in dls_train]
    print("clients amount of data samples", list_len)

    return dls_train, dls_test

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
import torchvision
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

    if dataset_name[:5] == "MNIST":
        db = datasets.MNIST(root='./data', train=train, download=True)
    elif dataset_name[:8] == "CIFAR100":
        db = datasets.CIFAR100(root='./data', train=train, download=True)
    elif dataset_name[:7] == "CIFAR10":
        db = datasets.CIFAR10(root='./data', train=train, download=True)
    elif dataset_name[:12] == "FashionMNIST":
        db = datasets.FashionMNIST(root='./data', train=train, download=True)

    save_name = f"{dataset_name}_M{n_clients}"
    if train:
        save_name += "_train"
    else:
        save_name += "_test"

    n_samples = 100
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

    if dataset_name[:8] == "CIFAR100":
        n_classes = 100
    else:
        n_classes = 10

    if alpha == 0.0:
        matrix = np.concatenate(
            [np.identity(n_classes) for _ in range(n_clients // n_classes + 1)]
        )[:n_clients]
    else:
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

        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.transform(self.features[idx]), self.labels[idx]


def dataloaders_dirichlet(
        dataset_name: str, file_name: str, n_clients: int,
        batch_size: int, shuffle: bool, transf: transforms
):
    """Download for all the clients their respective dataset"""
    print(file_name)

    dls = []

    for k in range(n_clients):
        dataset = DatasetDirichlet(dataset_name, file_name, k, transf)
        dataset_dl = DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle
        )
        dls.append(dataset_dl)

    return dls




"""
-----------
FEMNIST Dataset
-----------
"""


# class FEMNISTdataset(Dataset):
#
#     def __init__(self, data, user):
#
#         self.features = torch.tensor(data["user_data"][user]["x"])
#         # print(self.features.shape)
#         self.labels = torch.tensor(data["user_data"][user]["y"]).long()
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]
#
#
# def clients_set_FEMNIST(path: str, batch_size: int, n_clients: int, shuffle=True):
#     """Download for all the clients their respective dataset"""
#
#     list_json = os.listdir(path)
#     list_json = sorted(list_json)
#
#     list_dls = []
#     n_users = 0
#     for js in list_json:
#
#         with open(path + js) as json_file:
#             data = json.load(json_file)
#
#         for user in data["users"]:
#             dataset = FEMNISTdataset(data, user)
#             dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#             list_dls.append(dataloader)
#
#             n_users += 1
#             if n_users == n_clients:
#                 return list_dls


"""
---------
Simple dataset used for testing
with only two clients
---------
"""


class SimpleDataset(Dataset):
    def __init__(self, features, y):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_test_dataset():

    ds_train = [
        SimpleDataset([[0], [1], [2]], [[0], [1], [2]]),
        SimpleDataset([[0], [1], [2]], [[0], [2], [4]]),
    ]
    dls_train = [DataLoader(ds, batch_size=10, shuffle=True) for ds in ds_train]

    return dls_train, dls_train


"""
---------
Regression
---------
"""


class RegressionDataset(Dataset):
    def __init__(self, k, n_clients):

        np.random.seed(k)

        dim = 1
        n_samples = 10 ** 3

        features = np.random.uniform(0, 1, (n_samples, dim))
        if k <= 2:
            mu = 0.5
        else:
            mu = 1.0
        mu = np.array([k / (n_clients - 1)] * dim)
        y = features.dot(mu).reshape(-1, 1)  # + np.random.normal(0, 0.1, size=(n_samples, 1))

        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

"""
---------
Celeba
---------
"""

def create_pkl_celeba(path_train: str, n_clients: int, backdoored: bool):

    ds = torchvision.datasets.CelebA(root="./data", download=True,
                                     transform=transforms.Resize((64, 64)))

    list_X, list_y = [], []
    X, y = [], []
    
    for i in range(len(ds)):
        idx_client = torch.where(ds.identity == i+1)[0]

        if len(X) >= 1:
            # GIVE THE 'client' ITS FEATURES AND LABELS
            if not backdoored:
                list_X.append(np.array(X[:100]))
            elif backdoored:
                list_X.append(np.array(
                    [backdoor(np.array(img), i%10) for img in X[:100]]
                ))

            list_y.append(np.array(y[:100]))
            X, y = [], []
        else:
            X += [np.array(ds[idx][0]) for idx in idx_client]
            y += [np.array(ds[idx][1]) for idx in idx_client]

        if len(list_X) == n_clients:
            break

    # Save clients dataset
    with open(path_train, "wb") as output:
        pickle.dump((list_X, list_y), output)


def create_pkl_celeba_leaf(path_train: str, n_clients: int, backdoored: bool):

    ds = torchvision.datasets.CelebA(root="./data", download=True,
                                     transform=transforms.Resize((64, 64)))

    list_X, list_y = [], []
    # X, y = [], []

    for i in range(len(ds)):
        idx_client = torch.where(ds.identity == i+1)[0]

        X = [np.array(ds[idx][0]) for idx in idx_client]
        y = [np.array(ds[idx][1]) for idx in idx_client]

        if len(X) >= 25:
        #     GIVE THE 'client' ITS FEATURES AND LABELS
            if not backdoored:
                list_X.append(np.array(X[:100]))
            elif backdoored:
                list_X.append(np.array(
                    [backdoor(np.array(img), i%10) for img in X[:100]]
                ))

            list_y.append(np.array(y[:100]))
            # X, y = [], []

        if len(list_X) == n_clients:
            break

    # Save clients dataset
    with open(path_train, "wb") as output:
        pickle.dump((list_X, list_y), output)


def dl_celeba(dataset_name: str, backdoored: bool, batch_size: int, n_clients: int,
              trans: transforms) -> DataLoader:

    path_train = f"data/{dataset_name}_M{n_clients}.pkl"
    # if backdoored:
    #     path_train = f"data/celeba_backdoored_M{n_clients}.pkl"

    try:
        dss = [ClientDatasetCeleba(path_train, i, trans) for i in range(n_clients)]
    except:
        print("create pkl files")
        if dataset_name == "celeba":
            create_pkl_celeba(path_train, n_clients, backdoored)
        elif dataset_name == "celeba-leaf":
            create_pkl_celeba_leaf(path_train, n_clients, backdoored)
        dss = [ClientDatasetCeleba(path_train, i, trans) for i in range(n_clients)]

    dls = [DataLoader(
        ds, batch_size=min(batch_size, len(ds)), shuffle=True
    )
        for ds in dss]
    return dls

# def dl_celeba(batch_size: int, n_clients: int, trans: transforms) -> DataLoader:
#
#
#     ds = torchvision.datasets.CelebA(root="./data", download=True, transform=trans)
#
#     path_train = f"data/celeba_M{n_clients}.pkl"
#     dss = [ClientDatasetCeleba(path_train, i, trans) for i in range(n_clients)]
#
#     dls = [DataLoader(
#         ds, batch_size=min(batch_size, len(ds)), shuffle=True
#     )
#         for ds in dss]
#     return dls






class ClientDatasetCeleba(Dataset):
    def __init__(self, path: str, i: int, trans: transforms):

        dataset = pickle.load(open(path, "rb"))

        # idx_client = dataset.identity[dataset.identity == i]
        #
        # self.features, self.labels = [], []
        #
        # for idx in idx_client:
        #     self.features.append(dataset[idx][0])
        #     self.labels.append(dataset[idx][1])



        self.features = [Image.fromarray(arr) for arr in dataset[0][i]]
        self.labels = torch.Tensor(dataset[1][i][:, 31]).long() #smiling

        self.transform = trans

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.transform(self.features[idx]), self.labels[idx]





class addBackDoor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img: torch.Tensor):

        x, y, z = img.shape

        np.random.seed(42)
        for _ in range(4):
            pixel = (np.random.randint(x),
                     np.random.randint(y),
                     np.random.randint(z))
            img[pixel] = 1.
        return img


"""
---------
Combine them all
---------
"""



def get_dataloaders(
        dataset_name: str, batch_size: int, n_clients: int, verbose=True
):

    # Reproductibility in the creation of the datasets
    np.random.seed(0)
    torch.manual_seed(0)


    if dataset_name.split("_")[0] in \
            ["MNIST-shard", "CIFAR10", "CIFAR100", "FashionMNIST"]:

        name_split = dataset_name.split("_")
        dataset = name_split[0]
        alpha = float(name_split[1])

        if len(name_split) == 2:
            backdoored = False
        elif name_split[2] == "backdoored":
            backdoored = True

        path_train = f"data/{dataset_name}_M{n_clients}_train.pkl"
        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_ds_with_dirichlet(dataset_name, alpha, n_clients, backdoored)

        sequential = [transforms.ToTensor()]
        if dataset == "MNIST-shard":
            sequential.append(transforms.Normalize(0.1307, 0.3081))
        elif dataset == "CIFAR100":
            sequential.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            )
        elif dataset_name[:7] == "CIFAR10":
            sequential.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            )
        elif dataset_name[:12] == "FashionMNIST":
            sequential.append(transforms.Normalize((0.5,), (0.5,)))
        transf = transforms.Compose(sequential)

        dls_train = dataloaders_dirichlet(
            dataset_name, path_train, n_clients, batch_size, True, transf
        )
        dls_test = dls_train


    elif dataset_name.split("_")[0] in ["celeba", "celeba-leaf"]:
        backdoored = False
        if len(dataset_name.split("_")) == 2:
            backdoored = dataset_name.split("_")[1] == "backdoored"
            print("backdoored dataset")

        trans_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dls_train = dl_celeba(dataset_name, backdoored,
                              batch_size, n_clients, trans_train)
        dls_test = dls_train



    # Save in a file the number of samples owned per client
    list_len = [len(dl.dataset) for dl in dls_train]
    with open(f"./saved_exp_info/len_dbs/{dataset_name}_M{n_clients}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    if verbose:
        print("Participating clients:", len(dls_train))

    return dls_train, dls_test

    # if dataset == "test":
    #     dls_train, dls_test = create_test_dataset()
    #     return dls_train, dls_test

    # elif dataset == "Regression":
    #
    #     ds_train = [RegressionDataset(k, n_clients) for k in range(n_clients)]
    #     dls_train = [
    #         DataLoader(ds, batch_size=10, shuffle=True) for ds in ds_train
    #     ]
    #     dls_test = dls_train

    # elif dataset == "MNIST":
    #
    #     dls_train, _ = create_MNIST_iid(n_clients, batch_size)
    #     dls_test = dls_train


    # elif dataset == "FEMNIST":
    #
    #     if not os.path.isdir("data/leaf/data/femnist/data/train/"):
    #         os.chdir("data/leaf/data/femnist")
    #         os.system(
    #             "bash ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample "
    #             "--tf 0.8 --smplseed 0 --spltseed 0"
    #         )
    #         print("FEMNIST dataset created")
    #         os.chdir("../../../../")
    #     else:
    #         print("FEMNIST dataset already created")
    #
    #     path_train = "data/leaf/data/femnist/data/train/"
    #     dls_train = clients_set_FEMNIST(path_train, batch_size, n_clients)
    #
    #     path_test = "data/leaf/data/femnist/data/test/"
    #
    #     dls_test = clients_set_FEMNIST(path_test, batch_size, n_clients)
    #
    #


if __name__ == "__main__":
    create_pkl_celeba('data/celeba_backdoored_M100.pkl', 100, backdoored=True)
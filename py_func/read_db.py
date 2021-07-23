#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import json


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
            os.system("bash ./preprocess.sh -s niid --sf 0.1 -k 0 -t sample --tf 0.8")
            print("Shakespeare dataset created")
        except BaseException:
            print("Shakespeare dataset already exists")
        os.chdir(rep_path)

        path_train = "data/leaf/data/shakespeare/data/train/"
        list_dls_train = clients_set_Shakespeare(path_train, batch_size, True)

        path_test = "data/leaf/data/shakespeare/data/test/"

        list_dls_test = clients_set_Shakespeare(path_test, batch_size, True)

        if dataset[11:] == "":
            list_dls_train = list_dls_train[:80]
            list_dls_test = list_dls_test[:80]

        elif dataset[11:] == "2":
            list_dls_train = list_dls_train[:40]
            list_dls_test = list_dls_test[:40]

        elif dataset[11:] == "3":
            list_dls_train = list_dls_train[:20]
            list_dls_test = list_dls_test[:20]

        elif dataset[11:] == "4":
            list_dls_train = list_dls_train[:10]
            list_dls_test = list_dls_test[:10]

    # Save in a file the number of samples owned per client
    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    print("participating clients:", len(list_len))

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    return list_dls_train, list_dls_test

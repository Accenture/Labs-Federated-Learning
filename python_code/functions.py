#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
import torch


def get_n_iter(dataset: str, E: int, n_freeriders: int):

    if E == 5:
        if dataset == "MNIST-iid":
            n_iter = 200
        elif dataset == "MNIST-shard":
            n_iter = 300
        elif dataset == "CIFAR-10":
            n_iter = 150
        elif dataset == "shakespeare":
            n_iter = 100

    elif E == 20:
        if dataset == "MNIST-iid":
            n_iter = 100
        elif dataset == "MNIST-shard":
            n_iter = 150
        elif dataset == "CIFAR-10":
            n_iter = 75
        elif dataset == "shakespeare":
            n_iter = 50

    if n_freeriders == 5:
        n_iter *= 2
    elif n_freeriders == 45:
        n_iter *= 3

    return n_iter


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""

    with open(f"{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def exist(file_name):
    """check if a file exists"""
    return os.path.exists(file_name)


def load_initial_model(dataset):
    """Function that helps keep the same initial model for all the simulations
    If an initial has never been used before creates one that will be saved in
    variables"""

    from python_code.models import MultinomialLogisticRegression
    from python_code.models import CNN_CIFAR
    from python_code.models import LSTM_Shakespeare

    if dataset == "MNIST-iid" or dataset == "MNIST-shard":
        m_initial = MultinomialLogisticRegression()

        if os.path.exists("variables/model_MNIST_0.pth"):
            print(f"initial model for {dataset} already exists")
            m_initial_dic = torch.load("variables/model_MNIST_0.pth")
            m_initial.load_state_dict(m_initial_dic)

        else:
            print(f"initial model for {dataset} does not exist")
            torch.save(m_initial.state_dict(), "variables/model_MNIST_0.pth")

    elif dataset == "CIFAR-10":

        m_initial = CNN_CIFAR()

        if os.path.exists("variables/model_CIFAR-10_0.pth"):
            print(f"initial model for {dataset} already exists")
            m_initial_dic = torch.load("variables/model_CIFAR-10_0.pth")
            m_initial.load_state_dict(m_initial_dic)

        else:
            print(f"initial model for {dataset} does not exist")
            torch.save(m_initial.state_dict(), "variables/model_CIFAR-10_0.pth")

    elif dataset == "shakespeare":
        m_initial = LSTM_Shakespeare()

        if os.path.exists("variables/model_shakespeare_0.pth"):
            print(f"initial model for {dataset} already exists")
            m_initial_dic = torch.load("variables/model_shakespeare_0.pth")
            m_initial.load_state_dict(m_initial_dic)

        else:
            print(f"initial model for {dataset} does not exist")
            torch.save(m_initial.state_dict(), "variables/model_shakespeare_0.pth")

    return m_initial

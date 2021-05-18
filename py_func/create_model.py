#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, layer_1)
        self.fc3 = nn.Linear(layer_1, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = self.fc3(x)
        return x


# class CNN_CIFAR(torch.nn.Module):
#   """Model Used by the paper introducing FedAvg"""
#   def __init__(self):
#        super(CNN_CIFAR, self).__init__()
#        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=(3,3))
#        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
#        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
#
#        self.fc1 = nn.Linear(4*4*64, 64)
#        self.fc2 = nn.Linear(64, 10)
#
#   def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.max_pool2d(x, 2, 2)
#
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#
#        x=self.conv3(x)
#        x = x.view(-1, 4*4*64)
#
#        x = F.relu(self.fc1(x))
#
#        x = self.fc2(x)
#        return x


class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


def load_model(dataset, seed):

    torch.manual_seed(seed)

    if dataset == "MNIST_shard" or dataset == "MNIST_iid":
        model = NN(50, 10)

    elif dataset[:7] == "CIFAR10":
        #        model = CNN_CIFAR()
        model = CNN_CIFAR_dropout()

    return model

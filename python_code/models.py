#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn


# MULTINOMIAL LOGISTIC REGRESSION FOR MNIST
class MultinomialLogisticRegression(torch.nn.Module):
    def __init__(self):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784, 10)

    def forward(self, x):
        outputs = self.linear(x.view(-1, 784))
        return outputs


class CNN_CIFAR(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class LSTM_Shakespeare(torch.nn.Module):
    def __init__(self):
        super(LSTM_Shakespeare, self).__init__()
        self.n_characters = 100
        self.hidden_dim = 100
        self.n_layers = 2
        self.len_seq = 80
        self.batch_size = 100
        self.embed_dim = 8

        self.embed = torch.nn.Embedding(self.n_characters, self.embed_dim)

        self.lstm = torch.nn.LSTM(
            self.embed_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.n_characters)

    def forward(self, x):

        embed_x = self.embed(x)
        output, _ = self.lstm(embed_x)
        output = self.fc(output[:, -1])
        return output

    def init_hidden(self, batch_size):

        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
        )

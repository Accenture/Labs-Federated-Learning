#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        embeded_x = self.embed(x)
        output, _ = self.lstm(embeded_x)
        output = self.fc(output[:, -1])
        return output


class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

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


def load_model(dataset: str, seed: int, file_previous: str):

    torch.manual_seed(seed)

    if dataset[:11] == "Shakespeare":
        model = LSTM_Shakespeare()

    if dataset[:7] == "CIFAR10":
        model = CNN_CIFAR_dropout()

    if file_previous != "":

        model.load_state_dict(
            torch.load(f"saved_exp_info/final_model/{file_previous}.pth")
        )

    return model

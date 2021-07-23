#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


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


def load_model(dataset: str, seed: int):

    torch.manual_seed(seed)

    if dataset[:11] == "Shakespeare":
        model = LSTM_Shakespeare()

    return model

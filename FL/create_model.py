#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LogisticRegression(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(LogisticRegression, self).__init__()
        self.last = nn.Linear(layer_1, layer_2)
        self.layer_1 = layer_1

    def forward(self, x):
        x = self.last(x.view(-1, self.layer_1))
        return x


class TwoLayers(nn.Module):
    def __init__(self, layer_1, layer_2, layer_3):
        super(TwoLayers, self).__init__()
        self.fc1 = nn.Linear(layer_1, layer_2)
        self.last = nn.Linear(layer_2, layer_3)
        self.layer_1 = layer_1

    def forward(self, x):
        x = self.fc1(x.view(-1, self.layer_1))
        x = F.leaky_relu(x)
        x = self.last(x)
        return x

class CNN_Celeba(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN_Celeba, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        #         self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.pool(self.cnn1(x))
        out = self.pool(self.cnn2(out))
        out = self.pool(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        #         print(out.shape)
        out = self.fc1(out)
        return out


class CNN_CIFAR(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.reshape(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class CNN_CIFAR100(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 100)

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.reshape(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


FLATTEN_SIZE = 64*7*7
class CNN_FashionMNIST(nn.Module):
    def __init__(self):
        """
        Initializes the CNN Model Class and the required layers
        """
        super(CNN_FashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(FLATTEN_SIZE, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Form the Feed Forward Network by combininig all the layers
        :param x: the input image for the network
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        pred = F.log_softmax(x, dim=1)
        return pred


class simpleLinear(torch.nn.Module):
    def __init__(self, inputSize: int, outputSize: int, seed=0):
        super(simpleLinear, self).__init__()
        torch.manual_seed(seed)
        self.last = torch.nn.Linear(inputSize, outputSize, bias=True)

    def params(self):
        return [layer.data.numpy() for layer in self.parameters()][0][0, 0]

    def forward(self, x):
        return self.last(x)


class CNN_celeba(nn.Module):
    def __init__(self, in_channels: int =3, num_classes: int =2):
        super(CNN_celeba, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        #         self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.pool(self.cnn1(x))
        out = self.pool(self.cnn2(out))
        out = self.pool(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out


def load_model(dataset_name: str, model_type: str="default", seed: int =42):

    torch.manual_seed(seed)

    dataset = dataset_name.split("_")[0]

    if dataset in ["MNIST", "MNIST-shard"]:
        model = LogisticRegression(784, 10)
        loss_f = torch.nn.CrossEntropyLoss()

    elif dataset == "FashionMNIST":
        model = CNN_FashionMNIST()

        loss_f = torch.nn.CrossEntropyLoss()

    elif dataset == "CIFAR10":
        if model_type == "default":
            assert False, "only the CNN is programmed with CIFAR10"
        elif model_type == "CNN":
            model = CNN_CIFAR()

        loss_f = torch.nn.CrossEntropyLoss()

    elif dataset == "CIFAR100":
        if model_type == "default":
            assert False, "only the CNN is programmed with CIFAR100"
        elif model_type == "CNN":
            model = CNN_CIFAR100()

            loss_f = torch.nn.CrossEntropyLoss()

    elif dataset in ["celeba", "celeba-leaf"]:
        if model_type == "default":
            assert False, "only the CNN is programmed with celeba"
        elif model_type == "CNN":
            model = CNN_Celeba()
            loss_f = torch.nn.CrossEntropyLoss()

    print(model)

    return model, loss_f

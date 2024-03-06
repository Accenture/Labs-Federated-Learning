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

    # def before_last(self, x):
    #     x = self.fc1(x.view(-1, self.layer_1))
    #     x = F.leaky_relu(x)
    #     return x


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
        # Return the first weight
        # return np.array([layer.data.numpy() for layer in self.parameters()])[0, 0]
        return [layer.data.numpy() for layer in self.parameters()][0][0, 0]

    def forward(self, x):
        return self.last(x)

#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet1(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet1, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = torch.flatten(out, 1)
#         out = self.linear(out)
#         return F.log_softmax(out, dim=1)
#
#
# def ResNet18():
#     return ResNet1(BasicBlock, [2, 2, 2, 2])
#
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=False)
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 16
#         self.conv = conv3x3(3, 16)
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 16, layers[0])
#         self.layer2 = self.make_layer(block, 32, layers[1], 2)
#         self.layer3 = self.make_layer(block, 64, layers[2], 2)
#         self.avg_pool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64, num_classes)
#
#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels))
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


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
        #         print(out.shape)
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
        # model.load_state_dict(
        #     torch.load("data/CIFAR100.pth", map_location="cpu")
        # )

        loss_f = torch.nn.CrossEntropyLoss()

    elif dataset == "CIFAR10":

        # model = CNN_CIFAR_dropout()
        # model = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)

        if model_type == "default":
            assert False, "only the CNN is programmed with CIFAR10"
        elif model_type == "CNN":
            model = CNN_CIFAR()
            # model.load_state_dict(
            #     torch.load("data/CIFAR100.pth", map_location="cpu")
            # )

        loss_f = torch.nn.CrossEntropyLoss()

    elif dataset == "CIFAR100":
        if model_type == "default":
            assert False, "only the CNN is programmed with CIFAR100"
        elif model_type == "CNN":
            model = CNN_CIFAR100()
            # model.load_state_dict(
            #     torch.load("data/CIFAR10.pth", map_location="cpu")
            # )
            loss_f = torch.nn.CrossEntropyLoss()

    elif dataset in ["celeba", "celeba-leaf"]:
        if model_type == "default":
            assert False, "only the CNN is programmed with celeba"
        elif model_type == "CNN":
            model = CNN_Celeba()
            loss_f = torch.nn.CrossEntropyLoss()

    print(model)

    return model, loss_f

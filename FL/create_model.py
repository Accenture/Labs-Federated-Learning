#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LogisticRegression(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(layer_1, layer_2)
        self.layer_1 = layer_1

    def forward(self, x):
        x = self.fc1(x.view(-1, self.layer_1))
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

        embeded_x = self.embed(x)
        output, _ = self.lstm(embeded_x)
        output = self.fc(output[:, -1])
        return output


class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self, p_dropout: int):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=p_dropout)

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


class simpleLinear(torch.nn.Module):
    def __init__(self, inputSize: int, outputSize: int, seed=0):
        super(simpleLinear, self).__init__()

        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

    def params(self):
        # Return the first weight
        # return np.array([layer.data.numpy() for layer in self.parameters()])[0, 0]
        return [layer.data.numpy() for layer in self.parameters()][0][0, 0]

    def forward(self, x):
        return self.linear(x)



class CNN_CIFAR100(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self, p_dropout):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 100)

        self.dropout = nn.Dropout(p=p_dropout)

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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes:int):
        super(ResNet1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


def ResNet18_CIFAR10():
    return ResNet1(BasicBlock, [2, 2, 2, 2], 10)

def load_model(dataset_name: str, seed: int):

    torch.manual_seed(seed)
    name = dataset_name.split("_")[0]

    if name =="MNIST":
        model = LogisticRegression(784, 10)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "FEMNIST":
        model = LogisticRegression(784, 63)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "CIFAR10":
        model = CNN_CIFAR_dropout(p_dropout=0.)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "CIFAR10-flat":
        model = LogisticRegression(3 * 16 * 16, 10)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "CIFAR10-ResNet":
        model = torchvision.models.resnet18(pretrained=False)
        path_file = "saved_exp_info/final_model/CIFAR10-Resnet.pth"
        try:
            model.load_state_dict(
                torch.load(path_file, map_location="cpu")
            )
            print("parameters loaded")
        except:
            print("save parameters")
            torch.save(model.state_dict(), path_file)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "CIFAR100":
        model = CNN_CIFAR100(p_dropout=0.)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "CIFAR100-flat":
        model = LogisticRegression(3 * 16 * 16, 100)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "CIFAR10-flat-full":
        model = LogisticRegression(3 * 32 * 32, 10)
        loss_f = torch.nn.CrossEntropyLoss()

    elif name == "Shakespeare":
        model = LSTM_Shakespeare()
        loss_f = torch.nn.CrossEntropyLoss()

    print(model)
    return model, loss_f

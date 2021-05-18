#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from torchvision import datasets
import pickle


# In[2]:
trainset = datasets.CIFAR10(root="./data", train=True, download=True)
testset = datasets.CIFAR10(root="./data", train=False, download=True)

# Mapping from the numbers to the classes names
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# In[2]:
# PLOT SOME SAMPLES FROM THE DATASET
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainset.data[i], cmap=plt.cm.binary)
    plt.xlabel(classes[trainset.targets[i]])
plt.savefig("../../plots/CIFAR_samples.png")


# In[2]:
# PARAMETERS
n_clients = 5
samples_clients_training = 10000
samples_clients_test = 2000


# In[2]:
# CREATE THE DISTRIBUTED DATASET, i.e. CREATE THE DATASET OF EACH CLIENT

# Create the list of the index that we will use to create the dataset of each client
index_train = [i for i in range(50000)]
index_test = [i for i in range(10000)]

random.shuffle(index_train)
random.shuffle(index_test)


def get_client_dataset(dataset, list_samples):
    def get_sample(dataset, idx):
        return dataset.data[idx], dataset.targets[idx]

    X_client = list()
    y_client = list()

    for sample in list_samples:

        X_sample, y_sample = get_sample(dataset, sample)

        X_client.append(X_sample)
        y_client.append(y_sample)

    return np.array(X_client), y_client


X_train, y_train = [], []
X_test, y_test = [], []

for i in range(n_clients):

    samples_train = index_train[
        i * samples_clients_training : (i + 1) * samples_clients_training
    ]

    X_train_client, y_train_client = get_client_dataset(trainset, samples_train)

    X_train.append(X_train_client)
    y_train.append(y_train_client)

    samples_test = index_test[i * samples_clients_test : (i + 1) * samples_clients_test]

    X_test_client, y_test_client = get_client_dataset(testset, samples_test)
    X_test.append(X_test_client)
    y_test.append(y_test_client)


# In[2]:
# SAVE THE CREATED DATASETS
train_path = f"CIFAR_iid_train_{n_clients}_{samples_clients_training}.pkl"
with open(train_path, "wb") as output:
    pickle.dump((X_train, y_train), output)


test_path = f"CIFAR_iid_test_{n_clients}_{samples_clients_test}.pkl"
with open(test_path, "wb") as output:
    pickle.dump((X_test, y_test), output)

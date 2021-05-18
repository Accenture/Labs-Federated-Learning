#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CREATE TWO SHARDS OF EACH DIGIT
MINIMUM NUMBER OF CLIENTS IS 5
IN THAT CASE EACH CLIENT RECEIVE 2 SHARDS OF A DIFFERENT DIGIT 
AND 2 RANDOM SHARDS
CURRENT CODE ONLY WORKS FOR 5 CLIENTS"""

from torchvision import datasets
import random
import pickle

# In[2]:

# PARAMETERS
n_clients = 5
samples_per_shard_train = 150
samples_per_shard_test = 75

# DOWNLOAD/LOAD MNIST DATASET
MNIST_train = datasets.MNIST(root="", train=True, download=True)
MNIST_test = datasets.MNIST(root="", train=False, download=True)


# CHECKING IG THE GIVEN PARAMETERS ARE FEASIBLE
if n_clients * samples_per_shard_train > len(MNIST_train):
    print(
        "TOTAL NUMBER OF REQURIED TRAINING SAMPLES FOR THE CHOSEN PARAMETERS TOO HIGH"
    )

if n_clients * samples_per_shard_test > len(MNIST_test):
    print("TOTAL NUMBER OF REQURIED TESTING SAMPLES FOR THE CHOSEN PARAMETERS TOO HIGH")


# In[2]:
# Digits for the different shards
digits_for_shard = [i for i in range(10)] * (n_clients // 2)


# In[2]:
# CREATE THE SHARDS
# List of the different shards
# the shard order in the list corresponds to the digit associated to the shard

# Shards that will be attribuated to the clients knowingly.
shards_known_train = list()
shards_known_test = list()

# Shards that will be randomly attribuated to the clients.
shards_random_train = list()
shards_random_test = list()


for digit in range(10):

    row = 0

    shard_known_train = list()
    while len(shard_known_train) < samples_per_shard_train:
        if MNIST_train.train_labels[row] == digit:
            shard_known_train.append(MNIST_train.train_data[row])
        row += 1

    shard_known_test = list()
    while len(shard_known_test) < samples_per_shard_test:
        if MNIST_train.train_labels[row] == digit:
            shard_known_test.append(MNIST_train.test_data[row])
        row += 1

    shards_known_train.append(shard_known_train)
    shards_known_test.append(shard_known_test)

    shard_random_train = list()
    while len(shard_random_train) < samples_per_shard_train:
        if MNIST_train.train_labels[row] == digit:
            shard_random_train.append(MNIST_train.train_data[row])
        row += 1

    shard_random_test = list()
    while len(shard_random_test) < samples_per_shard_test:
        if MNIST_train.train_labels[row] == digit:
            shard_random_test.append(MNIST_train.test_data[row])
        row += 1

    shards_random_train.append(shard_random_train)
    shards_random_test.append(shard_random_test)


# In[2]:
index = [i for i in range(10)]

clients_X_train = [shards_known_train[2 * i : 2 * (i + 1)] for i in range(n_clients)]
clients_X_test = [shards_known_test[2 * i : 2 * (i + 1)] for i in range(n_clients)]

clients_labels_train = [[2 * i, 2 * i + 1] for i in range(5)]
clients_labels_test = [[2 * i, 2 * i + 1] for i in range(5)]


random_labels = [i for i in range(10)]

for idx in range(n_clients):
    for i in range(2):

        digit = index.pop(random.randrange(len(index)))

        clients_X_train[idx].append(shards_random_train[digit])
        clients_X_test[idx].append(shards_random_test[digit])

        clients_labels_train[idx].append(digit)
        clients_labels_test[idx].append(digit)


# In[2]:
# convert the lists into arrays for faster saving with pickle
import numpy as np
from itertools import product

for i, j, k in product(
    range(len(clients_X_train)),
    range(len(clients_X_train[0])),
    range(len(clients_X_train[0][0])),
):

    clients_X_train[i][j][k] = np.array(clients_X_train[i][j][k])

for i, j, k in product(
    range(len(clients_X_test)),
    range(len(clients_X_test[0])),
    range(len(clients_X_test[0][0])),
):

    clients_X_test[i][j][k] = np.array(clients_X_test[i][j][k])


# In[2]:
train_path = f"MNIST_shard_train_{n_clients}_{samples_per_shard_train}.pkl"
with open(train_path, "wb") as output:
    pickle.dump((clients_X_train, clients_labels_train), output)

test_path = f"MNIST_shard_test_{n_clients}_{samples_per_shard_test}.pkl"
with open(test_path, "wb") as output:
    pickle.dump((clients_X_test, clients_labels_test), output)

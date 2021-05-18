#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
import string
import numpy as np

n_clients = 5

# In[2]:
# OPEN THE FILES CREATED BY LEAF
file_train = "all_data_niid_1_keep_3000_train_8.json"
file_test = "all_data_niid_1_keep_3000_test_8.json"

with open(f"data/train/{file_train}") as json_file:
    data_train = json.load(json_file)
with open(f"data/test/{file_test}") as json_file:
    data_test = json.load(json_file)


# In[2]:
def conversion_string_to_vector(sentence):

    all_characters = string.printable

    vector = [all_characters.index(sentence[c]) for c in range(len(sentence))]

    return vector


# In[2]:
def create_clients(data, n_clients):
    clients_X = []
    clients_y = []

    for client in range(n_clients):

        client_X = []
        client_y = []

        dic_client = data["user_data"][data["users"][client]]
        print(data["num_samples"][client])
        X, y = dic_client["x"], dic_client["y"]

        for X_i, y_i in zip(X, y):

            client_X.append(conversion_string_to_vector(X_i))
            client_y.append(conversion_string_to_vector(y_i))

        clients_X.append(client_X)
        clients_y.append(client_y)

    return clients_X, clients_y


train_path = f"Shakespeare_train.pkl"
with open(train_path, "wb") as output:
    pickle.dump(create_clients(data_train, 5), output)

test_path = f"Shakespeare_test.pkl"
with open(test_path, "wb") as output:
    pickle.dump(create_clients(data_test, 5), output)


# In[2]:

clients_X_train, clients_y_train = create_clients(data_train, 5)
clients_X_test, clients_y_test = create_clients(data_test, 5)

n_samples = [len(clients_X_train[k]) + len(clients_X_test[k]) for k in range(n_clients)]
print(sum(n_samples), np.mean(n_samples), np.std(n_samples))

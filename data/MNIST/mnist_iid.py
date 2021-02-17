#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torchvision import datasets
import torch

import pickle

# In[2]:
#PARAMETERS
n_clients=5
samples_clients_training=600
samples_clients_test=300

#DOWNLOAD/LOAD MNIST DATASET
MNIST_train = datasets.MNIST(root='', train=True, download=True)
MNIST_test = datasets.MNIST(root='', train=False, download=True)


#CHECKING IG THE GIVEN PARAMETERS ARE FEASIBLE
if n_clients*samples_clients_training>len(MNIST_train):
    print("TOTAL NUMBER OF REQURIED TRAINING SAMPLES FOR THE CHOSEN PARAMETERS TOO HIGH")
    
if n_clients*samples_clients_test>len(MNIST_test):
    print("TOTAL NUMBER OF REQURIED TESTING SAMPLES FOR THE CHOSEN PARAMETERS TOO HIGH")
# In[2]:
    
list_client_samples_training=[samples_clients_training]*n_clients+[len(MNIST_train)-n_clients*samples_clients_training]
list_client_samples_test=[samples_clients_test]*n_clients+[len(MNIST_test)-n_clients*samples_clients_test]

train_data=torch.utils.data.random_split(MNIST_train,list_client_samples_training)[:-1]
test_data=torch.utils.data.random_split(MNIST_test,list_client_samples_test)[:-1]
# In[2]: 
train_path=f"MNIST_iid_train_{n_clients}_{samples_clients_training}.pkl"
with open(train_path, 'wb') as output:
    pickle.dump(train_data, output)
    
test_path=f"MNIST_iid_test_{n_clients}_{samples_clients_test}.pkl"
with open(test_path, 'wb') as output:
    pickle.dump(test_data, output)
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import pickle
import torch
import sys
sys.path.append('./python_code')

print("n_gaussians - cycle - outliers")


# In[2]:
n_gaussians=int(sys.argv[1])
cycle=int(sys.argv[2])
outliers=sys.argv[3]=="True"


#PARAMETERS
dataset="MNIST-shard"
n_clients=5
samples_per_client=600
samples_clients_test=300

epochs=20
lr=5*10**-4
n_iter=100
power=0.75

mu=0

device=("cuda" if torch.cuda.is_available() else "cpu")

std_0=10**-3



experiment_specific=f"{n_clients}_{epochs}_{samples_per_client}_{n_iter}_{n_gaussians}_{std_0}_{lr}_{n_gaussians}_{power}_{cycle}"

if outliers: experiment_specific+="_outlier"

noise_shape="linear"

# In[2]: 
#LOAD THE CLIENTS' DATASETS
from python_code.read_db import download_dataset
training_dls,testing_dls,fl_samples=download_dataset(dataset,n_clients,samples_per_client,
    samples_clients_test)

# In[2]:         
from python_code.functions import load_initial_model
m_initial=load_initial_model(dataset)

# In[2]:
from python_code.models import loss_MNIST
from python_code.Freeloader_functions import FL_freeloader_mixture
server_h_add_m,client_h_add_m,loss,acc=FL_freeloader_mixture(deepcopy(m_initial),m_initial,
    training_dls,fl_samples, n_iter,testing_dls,loss_MNIST,device,mu,noise_shape,outliers=outliers,
    noise="add",epochs=epochs,lr=lr,std_0=std_0,power=power,n_components=n_gaussians,cycle=cycle)
  
 
with open(f'fig4/{dataset}_FedAvg_server_{experiment_specific}.pkl', 'wb') as output:
    pickle.dump(server_h_add_m, output)
with open(f'fig4/{dataset}_FedAvg_clients_{experiment_specific}.pkl', 'wb') as output:
    pickle.dump(client_h_add_m, output)
    
with open(f'fig4/{dataset}_FedAvg_loss_{experiment_specific}.pkl', 'wb') as output:
    pickle.dump(loss, output)
with open(f'fig4/{dataset}_FedAvg_acc_{experiment_specific}.pkl', 'wb') as output:
    pickle.dump(acc, output)
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./python_code')

import pickle
import torch
import matplotlib.pyplot as plt

# In[2]: 
device=("cuda" if torch.cuda.is_available() else "cpu")

algo="FedAvg"
dataset="MNIST-shard"

n_clients=5
samples_per_client=600
samples_clients_test=300

epochs=20
lr=5*10**-4
std_0=10**-3
power=0.75


n_iter=100
mu=0


n_gaussians=1
cycle=100
experiment_specific=f"{n_clients}_{epochs}_{samples_per_client}_{n_iter}_{n_gaussians}_{std_0}_{lr}_{n_gaussians}_{power}_{cycle}"

server_g1=pickle.load(open(f'fig4/{dataset}_{algo}_server_{experiment_specific}.pkl', 'rb'))
client_g1=pickle.load(open(f'fig4/{dataset}_{algo}_clients_{experiment_specific}.pkl', 'rb'))
acc_g1=pickle.load(open(f'fig4/{dataset}_{algo}_acc_{experiment_specific}.pkl', 'rb'))


n_gaussians=3
cycle=100
experiment_specific=f"{n_clients}_{epochs}_{samples_per_client}_{n_iter}_{n_gaussians}_{std_0}_{lr}_{n_gaussians}_{power}_{cycle}"

server_g2=pickle.load(open(f'fig4/{dataset}_{algo}_server_{experiment_specific}.pkl', 'rb'))
client_g2=pickle.load(open(f'fig4/{dataset}_{algo}_clients_{experiment_specific}.pkl', 'rb'))
acc_g2=pickle.load(open(f'fig4/{dataset}_{algo}_acc_{experiment_specific}.pkl', 'rb'))


n_gaussians=3
cycle=100
experiment_specific=f"{n_clients}_{epochs}_{samples_per_client}_{n_iter}_{n_gaussians}_{std_0}_{lr}_{n_gaussians}_{power}_{cycle}"
experiment_specific+="_outlier"

server_g3=pickle.load(open(f'fig4/{dataset}_{algo}_server_{experiment_specific}.pkl', 'rb'))
client_g3=pickle.load(open(f'fig4/{dataset}_{algo}_clients_{experiment_specific}.pkl', 'rb'))
acc_g3=pickle.load(open(f'fig4/{dataset}_{algo}_acc_{experiment_specific}.pkl', 'rb'))


n_gaussians=3
cycle=50
experiment_specific=f"{n_clients}_{epochs}_{samples_per_client}_{n_iter}_{n_gaussians}_{std_0}_{lr}_{n_gaussians}_{power}_{cycle}"
experiment_specific+="_outlier"

server_g4=pickle.load(open(f'fig4/{dataset}_{algo}_server_{experiment_specific}.pkl', 'rb'))
client_g4=pickle.load(open(f'fig4/{dataset}_{algo}_clients_{experiment_specific}.pkl', 'rb'))
acc_g4=pickle.load(open(f'fig4/{dataset}_{algo}_acc_{experiment_specific}.pkl', 'rb'))



# In[2]: 
"""PARAMETERS FOR THE DIFFERENT DATASETS"""
params_shard={
    "n_clients":5,
    "samples":600,

    "n_iter":200,
    "epochs":20,    
    "lr":5*10**-4,

    "n_fload_samples":600,
    "list_pow":[2],
    "std_0":10**-3 
        }

params_shard["exp_specific"]=(f"{params_shard['n_clients']}_{params_shard['samples']}"
      +f"_{params_shard['epochs']}_{params_shard['n_iter']}"
      +f"_{params_shard['lr']}")



# In[2]: 
from python_code.functions_plot_fig_3_4 import plot_iteration_2_alone
plot_iteration_2_alone(client_g2,server_g2,1)
plt.savefig("plots/free-rider_server_fit.png")


# In[2]: 
from python_code.functions_plot_fig_3_4 import load_dataset_algo
FA_shard=load_dataset_algo("MNIST-shard","FedAvg",params_shard["exp_specific"],params_shard)
FP_shard=load_dataset_algo("MNIST-shard","FedProx",params_shard["exp_specific"],params_shard)


# In[2]:
from python_code.functions_plot_fig_3_4 import plot_metric_evolution_with_server

save_name="2"
plot_metric_evolution_with_server(FA_shard,client_g1,server_g1,acc_g1,client_g2,
    server_g2,acc_g2,client_g3,server_g3,acc_g3,client_g4,server_g4,acc_g4,"",save_name)






    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import torch
device=("cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('./python_code')

from python_code.functions import exist


"""Descriptions of the needed parameters to run this code in the ReadMe.md"""
print("algo - dataset - epochs - noise type - coef - power - n_freeriders - redo")
print(sys.argv[1:])


# In[2]:
"""COMMUNICATION WITH THE USER ABOUT HIS INPUTS"""
"""ALGO"""
algo=sys.argv[1]
print(f"FL algo chosen: {algo}")
if algo!='FedAvg' and algo!='FedProx': print("Only `FedAvg` or `FedProx'")
  

"""DATASET"""
dataset=sys.argv[2]
print(f"dataset chosen: {dataset}")
if (dataset!="MNIST-iid" and dataset!="MNIST-shard" and dataset!="CIFAR-10" 
        and dataset!="shakespeare"):
    print("Only `MNIST-iid`, `MNIST-shard', 'CIFAR-10' and 'shakespeare'")


"""NUMBER OF EPOCHS"""
epochs=int(sys.argv[3])
print("Number of epochs:", epochs)


"""TYPE OF FL FREE-RIDING RUN"""
FL=sys.argv[4]=="FL"
if FL: print(f"{algo} with no free-riders run and fixed initial model")

plain=sys.argv[4]=="plain"
if plain: print(f"{algo} with one plain free-rider")

disguised=sys.argv[4]=="disguised"
if disguised: print(f"{algo} with one disguised free-rider")

many=sys.argv[4][:4]=="many"
if many: print(f"{algo} with no free-riders run and random initial model")


"""COEF MULTIPLYING THE STD FOR THE HEURISTIC"""
coef=int(sys.argv[5])
if disguised: print("Multiplicative coeff for the heuristic std:", coef)


"""DECAYING POWER FOR THE FREE-RIDER NORMAL"""
power=float(sys.argv[6])
if disguised: print("Gamma for the noise:",power)


"""NUMBER OF FREE-RIDERS PARTICIPATING TO THE LEARNING PROCESS"""
n_freeriders=int(sys.argv[7])
print("Number of free-riders",n_freeriders)

"""RERUN THE EXPERIMENT EVEN IF IT HAS ALREADY BEEN RUN"""
force=sys.argv[8]=="True"
if force: print("the simulation will be rerun even if it has already been run")


   


"""FIXED PARAMETERS NOT DECIDED BY THE USER"""
n_clients=5
std_original=10**-3

from python_code.functions import get_n_iter
n_iter=get_n_iter(dataset,epochs,n_freeriders)
    
if dataset=="MNIST-iid":

    samples_per_client=600
    samples_clients_test=300
    
    lr=10**-3
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    
    mu=1. #regularization parameter for FedProx
    
    
elif dataset=="MNIST-shard":
    
    samples_per_client=600
    samples_clients_test=300
    
    lr=10**-3
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    
    mu=1.
    
    
elif dataset=="CIFAR-10":
    
    samples_per_client=10000    
    samples_clients_test=2000
    
    lr=10**-3
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    
    mu=1.
    
    
elif dataset=="shakespeare":
    
    samples_per_client=0
    samples_clients_test=0
    
    lr=0.5
    
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"

    mu=0.001
    
    

if algo=="FedAvg":
    mu=0.

print("FedProx regularization term mu:", mu)
print("learning rate", lr)


# In[2]: 
"""LOAD THE CLIENTS' DATASETS"""
from python_code.read_db import download_dataset
training_dls,testing_dls,fr_samples=download_dataset(dataset,n_clients,
    samples_per_client, samples_clients_test)


# In[2]:
"""LOAD THE INITIAL MODEL OR CREATES IT IF THERE IS NONE"""        
from python_code.functions import load_initial_model
m_initial=load_initial_model(dataset)


# In[2]: 
"""TRADITIONAL FEDERATED LEARNING WITH NO ATTACKERS"""
from python_code.FL_functions import FedProx
from python_code.FL_functions import loss_classifier

file_root_name=f"{dataset}_{algo}_FL_{experiment_specific}"

if FL and (not exist(f"hist/acc/{file_root_name}.pkl") or force):
    print(f"{algo} {dataset} FedProx mu={mu} not run yet")
    
    FedProx(deepcopy(m_initial),
        training_dls,n_iter,loss_classifier,testing_dls,device,mu,file_root_name,
        epochs=epochs,lr=lr)
    
elif FL:
    print(f"{algo} {dataset} FedProx mu={mu} already run")


# In[2]: 
"""FREE RIDING ATTACKS WITH PLAIN FREE-RIDERS"""
from python_code.Freeloader_functions import FL_freeloader

file_root_name=f"{dataset}_{algo}_plain_{n_freeriders}_{experiment_specific}"

if plain and (not exist(f"hist/acc/{file_root_name}.pkl") or force):
    print(f"{algo} {dataset} FedProx mu={mu} and Plain Free-riding not run yet")
    
    FL_freeloader(n_freeriders,deepcopy(m_initial),
        training_dls,fr_samples, n_iter,testing_dls,loss_classifier,device,mu,
        file_root_name,coef,noise_type="plain",epochs=epochs,lr=lr)
    
elif plain:
    print(f"{algo} {dataset} FedProx mu={mu} and Plain Free-riding already run")


# In[2]:
"""FREE RIDING ATTACKS WITH DISGUISED FREE-RIDERS"""

file_root_name=(f"{dataset}_{algo}_disguised_{power}_{std_original}_"
    f"{n_freeriders}_{experiment_specific}_{coef}" )
if disguised and (not exist(f"hist/acc/{file_root_name}.pkl") or force):  
    print(f"{algo} {dataset} FedProx mu={mu} and Disguised Free-riding not run yet")

    FL_freeloader(n_freeriders,deepcopy(m_initial),training_dls,
        fr_samples, n_iter,testing_dls,loss_classifier,device,mu,file_root_name,
        coef,noise_type="disguised",std_0=std_original,
        power=power,epochs=epochs,lr=lr)
        
elif disguised:
    print(f"{algo} {dataset} FedProx mu={mu} and Disguised Free-riding already run")


# In[2]:
"""FEDERATED LEARNING WITH RANDOM INITIALIZATIONS"""
if many:
    i_0=int(sys.argv[4][4:])
    experiment_specific+=f"_{i_0}"
    file_root_name=f"{dataset}_{algo}_FL_{experiment_specific}"
    
if many and(not exist(f"hist/acc/{file_root_name}.pth") or force):
    print(f"{algo} {dataset} FedProx mu={mu} random {i_0} not run yet") 
    
    from models import MultinomialLogisticRegression, LSTM_Shakespeare,CNN_CIFAR
    if dataset=="MNIST-iid" or dataset=="MNIST-shard":
        model=MultinomialLogisticRegression()
    elif dataset=="CIFAR-10":
        model=CNN_CIFAR()
    elif dataset=="shakespeare":
        model=LSTM_Shakespeare()

    FedProx(model,training_dls,n_iter,loss_classifier,testing_dls,
        device,mu,file_root_name,epochs=epochs,lr=lr)
    
    
elif many:
    print(f"{algo} {dataset} FedProx mu={mu} random {i_0} not run yet") 
            
            
            
        
        

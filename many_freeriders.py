#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./python_code')

from copy import deepcopy
import torch

from python_code.functions import save_variable,exist


print("algo - dataset - epochs - noise type - noise shape - coef -power- free-riders - redo")
print(sys.argv[1:])

# In[2]:

device=("cuda" if torch.cuda.is_available() else "cpu")

#Ask the algorithm to run. Can only be `FedAvg` or `FedProx`
algo=sys.argv[1]
print(f"FL algo chosen: {algo}")
if algo!='FedAvg' and algo!='FedProx':
    print("FL algo not supported. Enter `FedAvg` or `FedProx'")
    
dataset=sys.argv[2]
print(f"dataset chosen: {dataset}")
if dataset!="MNIST-iid" and dataset!="MNIST-shard":
     print("Dataset not supported. Enter `MNIST-iid` or `MNIST-shard'")

epochs=int(sys.argv[3])


FL=sys.argv[4]=="FL"
no_fr=sys.argv[4]=="none"
additive=sys.argv[4]=="add"
many=sys.argv[4][:4]=="many"
if additive: print("noise applied")
elif many:print("many simulation run")
else: print("no noise applied")


noise_shape=sys.argv[5]
print("noise shape ",noise_shape)


coef=int(sys.argv[6])
print("coef",coef)


power=int(sys.argv[7])
print("power",power)


n_freeriders=int(sys.argv[8])
print("n_freeriders",n_freeriders)


force=sys.argv[9]=="True"
print(force)


if dataset=="MNIST-iid":

    n_clients=5
    samples_per_client=600
    samples_clients_test=300
    
    lr=5*10**-4
    std_original=10**-3
    
    if epochs==5 and n_freeriders==5:n_iter=300
    if epochs==5 and n_freeriders==45:n_iter=800
    if epochs==20 and n_freeriders==5:n_iter=150
    if epochs==20 and n_freeriders==45:n_iter=450

    if noise_shape=="linear":list_power=[power]
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    mu=1
    
    
if dataset=="MNIST-shard":
    
    n_clients=5
    samples_per_client=600
    samples_clients_test=300
    
    lr=5*10**-4
    std_original=10**-3
    if noise_shape=="linear":list_power=[power]
    
    if epochs==5 and n_freeriders==5:n_iter=400
    if epochs==5 and n_freeriders==45:n_iter=1000
    if epochs==20 and n_freeriders==5:n_iter=200
    if epochs==20 and n_freeriders==45:n_iter=600
    
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    mu=1
    
if dataset=="shakespeare":

    n_clients=5
    
    lr=0.5
    std_original=10**-3
    
    if noise_shape=="linear":list_power=[2]
    
    if epochs==5 and n_freeriders==5:n_iter=100
    if epochs==5 and n_freeriders==45:n_iter=200
    if epochs==20 and n_freeriders==5:n_iter=75
    if epochs==20 and n_freeriders==45:n_iter=150
    
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"

    samples_per_client=0
    samples_clients_test=0
    mu=0.001
    
if algo=="FedAvg":
    mu=0








# In[2]: 
#LOAD THE CLIENTS' DATASETS
from python_code.read_db import download_dataset
training_dls,testing_dls,fl_samples=download_dataset(dataset,n_clients,samples_per_client,
    samples_clients_test)


# In[2]:         
from python_code.functions import load_initial_model
m_initial=load_initial_model(dataset)


# In[2]: 
from python_code.FL_functions import FedAvg
from python_code.models import loss_MNIST

file_root_name=f"{dataset}_{algo}_fair_{experiment_specific}"


if not exist(f"models_s/{file_root_name}.pth") and FL:
    print("no initial train for FedAvg")
    m_without,loss_without,acc_without=FedAvg(deepcopy(m_initial),
        training_dls,n_iter,loss_MNIST,testing_dls,device,mu,file_root_name,epochs=epochs,lr=lr)
    
    save_variable(loss_without,f"{dataset}_{algo}_loss_fair_{experiment_specific}")
    save_variable(acc_without,f"{dataset}_{algo}_acc_fair_{experiment_specific}")
    
    torch.save(m_without.state_dict(), f"models_s/{file_root_name}.pth")

if exist(f"models_s/{file_root_name}.pth"):
    print("FedAvg already exists")




# In[2]: 
from python_code.Freeloader_functions import FL_freeloader_many

file_root_name=f"{dataset}_{algo}_without_{n_freeriders}_{experiment_specific}"
if not exist(f"models_s/{file_root_name}.pth") and no_fr:
    print("FedAvg+Naive freeloader does not exist")
    m_with,loss_with,acc_with,stds=FL_freeloader_many(n_freeriders,deepcopy(m_initial),
        training_dls,fl_samples, n_iter,testing_dls,loss_MNIST,device,mu,file_root_name,noise_shape,coef,noise="naive",epochs=epochs,lr=lr)

    save_variable(loss_with,f"{dataset}_{algo}_loss_none_{n_freeriders}_{experiment_specific}")
    save_variable(acc_with,f"{dataset}_{algo}_acc_none_{n_freeriders}_{experiment_specific}")

    torch.save(m_with.state_dict(), f"models_s/{file_root_name}.pth")
    
if exist(f"models_s/{file_root_name}.pth"):
    print("FedAvg+Naive freeloader already exists")


# In[2]:
if noise_shape=="exp":
    experiment_specific+=f"_{noise_shape}"

if coef==3:
    print("coef ",coef)
    experiment_specific+=f"_{coef}"
    


if additive:
  
    for power in list_power:
        print(f"additive {power}")
        
        file_root_name=f"{dataset}_{algo}_add_{power}_{std_original}_{n_freeriders}_{experiment_specific}"
        if not exist(f"variables/{dataset}_{algo}_loss_add_{power}_{std_original}_{n_freeriders}_{experiment_specific}.pkl") or force:
            
            m,loss,acc,list_std=FL_freeloader_many(n_freeriders,deepcopy(m_initial),
                training_dls,fl_samples, n_iter,testing_dls,loss_MNIST,
                device,mu,file_root_name,noise_shape,coef,noise="add",std_0=std_original,
                power=power,epochs=epochs,lr=lr)
            
            save_variable(loss,f"{dataset}_{algo}_loss_add_{power}_{std_original}_{n_freeriders}_{experiment_specific}")
            save_variable(acc,f"{dataset}_{algo}_acc_add_{power}_{std_original}_{n_freeriders}_{experiment_specific}")
            
            save_variable(list_std,f"{dataset}_{algo}_params_add_{power}_{std_original}_{n_freeriders}_{experiment_specific}")
            
        else:
            print(f"Simulation for additive {power} has already been run")


# In[2]:
if many:
    
    i_0=int(sys.argv[4][4:])
    file_root_many=f"{dataset}_{algo}_fair_{experiment_specific}_{i_0}"
    
    if not exist(f"variables/random_simu/{dataset}_{algo}_loss_fair_{experiment_specific}_{i_0}.pkl") or force:

        print(f"{dataset}_{algo}_{i_0}")
        from models import MultinomialLogisticRegression, LSTM_Shakespeare
        
        if dataset=="MNIST-iid" or dataset=="MNIST-shard":
            model=MultinomialLogisticRegression()
        elif dataset=="shakespeare":
            model=LSTM_Shakespeare()
        
        model,loss,acc=FedAvg(model,training_dls,
        n_iter,loss_MNIST,testing_dls,device,mu,file_root_many,epochs=epochs,lr=lr)
        
        save_variable(loss,f"random_simu/{dataset}_{algo}_loss_fair_{experiment_specific}_{i_0}")
        save_variable(acc,f"random_simu/{dataset}_{algo}_acc_fair_{experiment_specific}_{i_0}")
        
        

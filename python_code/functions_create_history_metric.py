#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import sys 
import torch

import pickle

import numpy as np
from scipy.stats import ks_2samp



def load_model_params(dataset,path):
    
    from models import MultinomialLogisticRegression,LSTM_Shakespeare
    
    #Get the final model
    if dataset=="MNIST-iid" or dataset=="MNIST-shard":
        model=MultinomialLogisticRegression()      
        
    elif dataset=="shakespeare":
        model=LSTM_Shakespeare()
    
    model_dic=torch.load(path)
    model.load_state_dict(model_dic)
        
    params_model=[tens_param.detach() for tens_param in list(model.parameters())]
    
    return params_model



def get_metric(metric,dataset,list_A,list_B):
    
    n_tensors=len(list_A)
    
    if metric=="KS":
        distri_A=np.concatenate([list_A[k].reshape(-1)  for k in range(n_tensors)]) 
        distri_B=np.concatenate([list_B[k].reshape(-1)  for k in range(n_tensors)])
        return ks_2samp(distri_A,distri_B)[0]
        
    
    elif metric=="L2":
        
        return sum([torch.sum((list_A[k]-list_B[k])**2) for k in range(n_tensors)])
    
    
    
def create_history(noise_type,metric,dataset,algo,n_iter,exp_specific_ref,exp_specific_simu):
    
    metric_hist=[]
    
    path=f"models_s/{dataset}_{algo}_fair_{exp_specific_ref}.pth"
    params_final_fair=load_model_params(dataset,path)

    #Get the history for the associated experiment at each iteration and compute the metric for this file.
    for i in range(n_iter):
        
        path=f"saved_models/{dataset}_{algo}_{noise_type}_{exp_specific_simu}_{i}_server.pth"
        iteration_parameters=load_model_params(dataset,path)
        
        metric_hist.append(get_metric(metric,dataset,iteration_parameters,params_final_fair))
        
    #Save the history
    with open(f'created_histories/{dataset}_{algo}_{noise_type}_{metric}_{exp_specific_simu}.pkl', 'wb') as output:
        pickle.dump(metric_hist, output)

    return metric_hist


def create_history_per_fr_type(noise_type,dataset,n_iter,experiment_specific,exp_specific_simu):

    try:
        create_history(noise_type,"L2",dataset,"FedAvg",n_iter,experiment_specific,exp_specific_simu)
        create_history(noise_type,"KS",dataset,"FedAvg",n_iter,experiment_specific,exp_specific_simu)
        print("Working","FedAvg",dataset,noise_type)
    except:
        print("Not Working","FedAvg",dataset,noise_type,exp_specific_simu)
        
    try:
        create_history(noise_type,"L2",dataset,"FedProx",n_iter,experiment_specific,exp_specific_simu)
        create_history(noise_type,"KS",dataset,"FedProx",n_iter,experiment_specific,exp_specific_simu)
        print("Working","FedProx",dataset,noise_type)
    except:
        print("Not Working","FedProx",dataset,noise_type,exp_specific_simu)



def get_history_experiment(dataset,n_iter,n_freeriders,experiment_specific):
    
    if n_freeriders>1:experiment_specific_fr=f"{n_freeriders}_{experiment_specific}"
    else:experiment_specific_fr=experiment_specific
    print(experiment_specific_fr)
    
    #Plain Free-riders
    create_history_per_fr_type("without",dataset,n_iter,experiment_specific,
        experiment_specific_fr)

    #Disguised Free-riders sigma x1 power=1
    exp_specific_1="1_0.001_"+experiment_specific_fr
    create_history_per_fr_type("add",dataset,n_iter,experiment_specific,
        exp_specific_1)
        
        
    #Disguised Free-riders sigma x3 power=1
    exp_specific_2="1_0.001_"+experiment_specific_fr+"_3"
    create_history_per_fr_type("add",dataset,n_iter,experiment_specific,
        exp_specific_2)
    
    
    #Disguised Free-riders sigma x1 power=0.5
    exp_specific_3="0.5_0.001_"+experiment_specific_fr
    create_history_per_fr_type("add",dataset,n_iter,experiment_specific,
        exp_specific_3)
        
        
    #Disguised Free-riders sigma x3 power=0.5
    exp_specific_4="0.5_0.001_"+experiment_specific_fr+"_3"
    create_history_per_fr_type("add",dataset,n_iter,experiment_specific,
        exp_specific_4)
    
    #Disguised Free-riders sigma x1 power=2
    exp_specific_5="2_0.001_"+experiment_specific_fr
    create_history_per_fr_type("add",dataset,n_iter,experiment_specific,
        exp_specific_5)
        
        
    #Disguised Free-riders sigma x3 power=2
    exp_specific_6="2_0.001_"+experiment_specific_fr+"_3"
    create_history_per_fr_type("add",dataset,n_iter,experiment_specific,
        exp_specific_6)



def KS_L2_history_fig1(epochs):

    #GENERAL PARAMETERS
    n_clients=5
    samples_per_client=600
    
    
    #MNIST-iid
    dataset="MNIST-iid"
    if epochs==5: n_iter=300
    elif epochs==20: n_iter=150
    lr=5*10**-4
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    
    get_history_experiment(dataset,n_iter,1,experiment_specific)
     
    #MNIST-shard
    dataset="MNIST-shard"
    if epochs==5: n_iter=400
    elif epochs==20: n_iter=200
    lr=5*10**-4
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,1,experiment_specific) 
        
    #shakespeare
    dataset="shakespeare"
    if epochs==5: n_iter=100
    elif epochs==20: n_iter=50
    lr=0.5    
    
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,1,experiment_specific) 



def KS_L2_history_fig2(epochs):
     
    #GENERAL PARAMETERS
    n_clients=5
    
    samples_per_client=600
    
    #MNIST-iid
    dataset="MNIST-iid"
    if epochs==5: n_iter=300
    elif epochs==20:n_iter=150
    lr=5*10**-4
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,5,experiment_specific) 
    
    if epochs==5: n_iter=800
    elif epochs==20:n_iter=450
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,45,experiment_specific) 
    
    
    #MNIST-shard
    dataset="MNIST-shard"
    if epochs==5: n_iter=400
    elif epochs==20:n_iter=200
    lr=5*10**-4
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,5,experiment_specific) 
    
    if epochs==5: n_iter=1000
    elif epochs==20:n_iter=600
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,45,experiment_specific) 
        
    #shakespeare
    dataset="shakespeare"
    n_iter=50    
    lr=0.5    
    
    if epochs==5: n_iter=100
    elif epochs==20:n_iter=75
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,5,experiment_specific) 
    
    if epochs==5: n_iter=200
    elif epochs==20:n_iter=150
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"
    get_history_experiment(dataset,n_iter,45,experiment_specific) 

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import sys 
import torch

import pickle

import numpy as np
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
import seaborn as sn


def open_history(noise_type,metric,dataset,algo,exp_specific_simu):

    with open(f'created_histories/{dataset}_{algo}_{noise_type}_{metric}_{exp_specific_simu}.pkl', 'rb') as output:
        metric_hist=pickle.load(output)
        
    if metric=="L2":
        if dataset=="shakespeare":metric_hist=np.array(metric_hist)/135700
        else:metric_hist=np.array(metric_hist)/7850
    
    return metric_hist



def get_dic(dataset,algo,metric,n_freerider,experiment_specific):
    
    if n_freerider>1:
    
    
        exp_specific_without=f"{n_freerider}_"+experiment_specific
        exp_specific_1_1="1.0_0.001_"+f"{n_freerider}_"+experiment_specific
        exp_specific_3_1="1.0_0.001_"+f"{n_freerider}_"+experiment_specific+"_3"
        exp_specific_1_05="0.5_0.001_"+f"{n_freerider}_"+experiment_specific
        exp_specific_3_05="0.5_0.001_"+f"{n_freerider}_"+experiment_specific+"_3"
        exp_specific_1_2="2.0_0.001_"+f"{n_freerider}_"+experiment_specific
        exp_specific_3_2="2.0_0.001_"+f"{n_freerider}_"+experiment_specific+"_3"
    else:

        exp_specific_without=experiment_specific
        exp_specific_1_1="1.0_0.001_"+experiment_specific
        exp_specific_3_1="1.0_0.001_"+experiment_specific+"_3"
        exp_specific_1_05="0.5_0.001_"+experiment_specific
        exp_specific_3_05="0.5_0.001_"+experiment_specific+"_3"
        exp_specific_1_2="2.0_0.001_"+experiment_specific
        exp_specific_3_2="2.0_0.001_"+experiment_specific+"_3"
    
    
    dic={
        "without":open_history("without",metric,dataset,algo,exp_specific_without),
        "add_1_1":open_history("add",metric,dataset,algo,exp_specific_1_1), 
        "add_3_1":open_history("add",metric,dataset,algo,exp_specific_3_1), 
        "add_1_0.5":open_history("add",metric,dataset,algo,exp_specific_1_05), 
        "add_3_0.5":open_history("add",metric,dataset,algo,exp_specific_3_05),
        "add_1_2":open_history("add",metric,dataset,algo,exp_specific_1_2), 
        "add_3_2":open_history("add",metric,dataset,algo,exp_specific_3_2),
        }

    return dic



def KS_L2_many_freerider(metric,epochs,algo):

    n_clients=5
    samples_per_client=600
    
    #MNIST-iid 1 free-rider
    dataset="MNIST-iid"
    lr=5*10**-4
    
    if epochs==5:n_iter=300
    elif epochs==20:n_iter=150
    n_fr=5
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    FA_iid_5=get_dic(dataset,algo,metric,n_fr,experiment_specific)
    
    if epochs==5:n_iter=800
    elif epochs==20:n_iter=450
    n_fr=45
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    FA_iid_45=get_dic(dataset,algo,metric,n_fr,experiment_specific)
    
    
    #MNIST-shard 1 free-rider
    dataset="MNIST-shard"
    n_iter=200
    lr=5*10**-4
    
    if epochs==5:n_iter=400
    elif epochs==20:n_iter=200
    n_fr=5
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    FA_shard_5=get_dic(dataset,algo,metric,n_fr,experiment_specific)
    
    if epochs==5:n_iter=1000
    elif epochs==20:n_iter=600
    n_fr=45
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    FA_shard_45=get_dic(dataset,algo,metric,n_fr,experiment_specific)
    
    
        
    #shakespeare free-rider
    dataset="shakespeare"
    if epochs==5:n_iter=100
    elif epochs==20:n_iter=75
    lr=0.5 
    n_fr=5   
    
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"
    FA_shak_5=get_dic(dataset,algo,metric,n_fr,experiment_specific)
    
    
    if epochs==5:n_iter=200
    elif epochs==20:n_iter=150
    n_fr=45
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"
    FA_shak_45=get_dic(dataset,algo,metric,n_fr,experiment_specific)
    
    return FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,FA_shak_45



def KS_L2_single_freerider(metric,epochs):

    #GENERAL PARAMETERS
    n_clients=5
    
    samples_per_client=600
    
    
    #MNIST-iid 1 free-rider
    dataset="MNIST-iid"
    if epochs==5: n_iter=300
    elif epochs==20: n_iter=150
    lr=5*10**-4
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    
    FA_iid=get_dic(dataset,"FedAvg",metric,1,experiment_specific)
    FP_iid=get_dic(dataset,"FedProx",metric,1,experiment_specific)
    
    
    #MNIST-shard 1 free-rider
    dataset="MNIST-shard"
    if epochs==5: n_iter=400
    elif epochs==20: n_iter=200
    lr=5*10**-4
    
    experiment_specific=f"{n_clients}_{samples_per_client}_{epochs}_{n_iter}_{lr}"
    FA_shard=get_dic(dataset,"FedAvg",metric,1,experiment_specific)
    FP_shard=get_dic(dataset,"FedProx",metric,1,experiment_specific)
        
    #shakespeare free-rider
    dataset="shakespeare"
    if epochs==5: n_iter=100
    elif epochs==20: n_iter=50
    lr=0.5    
    
    experiment_specific=f"{n_clients}_{epochs}_{n_iter}_{lr}"
    FA_shak=get_dic(dataset,"FedAvg",metric,1,experiment_specific)
    FP_shak=get_dic(dataset,"FedProx",metric,1,experiment_specific)
    
    return FA_iid,FP_iid,FA_shard,FP_shard,FA_shak,FP_shak



def plot_KS_L2(FA_iid,FA_shard,FA_shak,FP_iid,FP_shard,FP_shak,
        y_label,title=None,save_name=None,fig=0,legend_idx=0):
    
    
    cols = ["MNIST-iid","MNIST-shard","Shakespeare"]
    if fig==1:rows=["FedAvg","FedProx"]           
    else:rows=["5 Free-riders","45 Free-riders"]           
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17,8))

    sn.set_style("ticks")

    if title: plt.suptitle(title)
    
    #Create the titles for the columns
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    
    #Create the titles for the rows
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center',rotation=90) 
    
        
    def plot_dic(dic,plot_idx):

        
        ax=axes[plot_idx//3][plot_idx%3]

        ax.plot(dic["without"][1:],label="Plain")
        ax.plot(dic["add_1_1"][1:],label=r"Disguised $\sigma$, $\gamma=1$")
        ax.plot(dic["add_3_1"][1:],label=r"Disguised $3\sigma$, $\gamma=1$")
        ax.plot(dic["add_1_0.5"][1:],label=r"Disguised $\sigma$, $\gamma=0.5$",linestyle='--')
        ax.plot(dic["add_3_0.5"][1:],label=r"Disguised $3\sigma$, $\gamma=0.5$",linestyle='--')
        ax.plot(dic["add_1_2"][1:],label=r"Disguised $\sigma$, $\gamma=2$",linestyle='--')
        ax.plot(dic["add_3_2"][1:],label=r"Disguised $3\sigma$, $\gamma=2$",linestyle='--')

        
        if plot_idx==legend_idx:ax.legend(ncol=2)
        
        ax.autoscale(enable=True, axis='x', tight=True)
        if plot_idx//3==1: ax.set_xlabel("# rounds")
        if plot_idx%3==0:ax.set_ylabel(y_label)
        
        ax.set_yscale("log")

    
    plot_dic(FA_iid,0)
    plot_dic(FA_shard,1)
    plot_dic(FA_shak,2)
    
    plot_dic(FP_iid,3)
    plot_dic(FP_shard,4)
    plot_dic(FP_shak,5)    
    
    fig.tight_layout()
    
    plt.savefig(f"plots/{save_name}.png")




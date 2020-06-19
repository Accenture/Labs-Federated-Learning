#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sn
from copy import deepcopy


def load_metric(algo,metric,dataset,experiment_specific,params):
    
    fair=pickle.load(open(f'variables/{dataset}_{algo}_{metric}_fair_{experiment_specific}.pkl', 'rb'))
    none=pickle.load(open(f'variables/{dataset}_{algo}_{metric}_none_{params["n_freeriders"]}_{experiment_specific}.pkl', 'rb'))
    additive_1_1=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_1.0_{params['std_0']}_{params['n_freeriders']}_{experiment_specific}.pkl", 'rb'))
    additive_3_1=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_1.0_{params['std_0']}_{params['n_freeriders']}_{experiment_specific}_3.pkl", 'rb'))
    additive_1_05=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_0.5_{params['std_0']}_{params['n_freeriders']}_{experiment_specific}.pkl", 'rb'))
    additive_3_05=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_0.5_{params['std_0']}_{params['n_freeriders']}_{experiment_specific}_3.pkl", 'rb'))
    additive_1_2=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_2.0_{params['std_0']}_{params['n_freeriders']}_{experiment_specific}.pkl", 'rb'))
    additive_3_2=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_2.0_{params['std_0']}_{params['n_freeriders']}_{experiment_specific}_3.pkl", 'rb'))
    
    dic={
        "fair":[np.mean(hist_i) for hist_i in fair],
        "none":[np.mean(hist_i) for hist_i in none],
        "add_1_1":[np.mean(hist_i) for hist_i in additive_1_1],
        "add_3_1":[np.mean(hist_i) for hist_i in additive_3_1],
        "add_1_0.5":[np.mean(hist_i) for hist_i in additive_1_05],
        "add_3_0.5":[np.mean(hist_i) for hist_i in additive_3_05],
        "add_1_2":[np.mean(hist_i) for hist_i in additive_1_2],
        "add_3_2":[np.mean(hist_i) for hist_i in additive_3_2],
    } 
      
    return dic



def params_accuracy_plot_many_freerider(algo,n_epochs):
    params_iid={
        "n_clients":5,
        "samples":600,
    
        "epochs":n_epochs,    
        "lr":5*10**-4,
    
        "n_fload_samples":600,
        "list_pow":[2],
        "std_0":10**-3 
            }
    
    params_shard={
        "n_clients":5,
        "samples":600,

        "epochs":n_epochs,    
        "lr":5*10**-4,
    
        "n_fload_samples":600,
        "list_pow":[2],
        "std_0":10**-3 
            }
    
    params_shak={
        "n_clients":5,
        "samples":0,
    
        "epochs":n_epochs,    
        "lr":0.5,
    
        "n_fload_samples":600,
        "list_pow":[2],
        "std_0":10**-3 ,
        "mu":0.001
            }
    
    params_iid_5=deepcopy(params_iid)
    params_iid_45=deepcopy(params_iid)
    params_shard_5=deepcopy(params_shard)
    params_shard_45=deepcopy(params_shard)
    params_shak_5=deepcopy(params_shak)
    params_shak_45=deepcopy(params_shak)
    
    params_iid_5["n_freeriders"]=5
    params_shard_5["n_freeriders"]=5
    params_shak_5["n_freeriders"]=5
    params_iid_45["n_freeriders"]=45
    params_shard_45["n_freeriders"]=45
    params_shak_45["n_freeriders"]=45

    
    if n_epochs==5:
        params_iid_5["n_iter"]=300
        params_shard_5["n_iter"]=400
        params_shak_5["n_iter"]=100
        params_iid_45["n_iter"]=800
        params_shard_45["n_iter"]=1000
        params_shak_45["n_iter"]=200
    elif n_epochs==20:
        params_iid_5["n_iter"]=150
        params_shard_5["n_iter"]=200
        params_shak_5["n_iter"]=75
        params_iid_45["n_iter"]=450
        params_shard_45["n_iter"]=600
        params_shak_45["n_iter"]=150
        
    
    params_iid_5["exp_specific"]=(f"{params_iid_5['n_clients']}_{params_iid_5['samples']}"
          +f"_{params_iid_5['epochs']}_{params_iid_5['n_iter']}"
          +f"_{params_iid_5['lr']}")
    
    params_shard_5["exp_specific"]=(f"{params_shard_5['n_clients']}_{params_shard_5['samples']}"
          +f"_{params_shard_5['epochs']}_{params_shard_5['n_iter']}"
          +f"_{params_shard_5['lr']}")
    
    params_shak_5["exp_specific"]=(f"{params_shak_5['n_clients']}"
          +f"_{params_shak_5['epochs']}_{params_shak_5['n_iter']}"
          +f"_{params_shak_5['lr']}")
    
    
    params_iid_45["exp_specific"]=(f"{params_iid_45['n_clients']}_{params_iid_45['samples']}"
          +f"_{params_iid_45['epochs']}_{params_iid_45['n_iter']}"
          +f"_{params_iid_45['lr']}")
    
    params_shard_45["exp_specific"]=(f"{params_shard_45['n_clients']}_{params_shard_45['samples']}"
          +f"_{params_shard_45['epochs']}_{params_shard_45['n_iter']}"
          +f"_{params_shard_45['lr']}")
    
    params_shak_45["exp_specific"]=(f"{params_shak_45['n_clients']}"
          +f"_{params_shak_45['epochs']}_{params_shak_45['n_iter']}"
          +f"_{params_shak_45['lr']}")
    
    FA_iid_5=load_dataset_algo("MNIST-iid",algo,params_iid_5["exp_specific"],params_iid_5)
    FA_iid_45=load_dataset_algo("MNIST-iid",algo,params_iid_45["exp_specific"],params_iid_45)
    
    FA_shard_5=load_dataset_algo("MNIST-shard",algo,params_shard_5["exp_specific"],params_shard_5)
    FA_shard_45=load_dataset_algo("MNIST-shard",algo,params_shard_45["exp_specific"],params_shard_45)
    
    FA_shak_5=load_dataset_algo("shakespeare",algo,params_shak_5["exp_specific"],params_shak_5)
    FA_shak_45=load_dataset_algo("shakespeare",algo,params_shak_45["exp_specific"],params_shak_45)
    
    
    return FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,FA_shak_45

    
  
    
    
def standard_deviation(algo,dataset,metric,params):
    """Use the history of the 30 simulations to get the min and max curve for 
    the confidence interval"""
    
    path=os.getcwd()+"/variables/random_simu"

    file_rd_simu=os.listdir(path)
    
    file_begin=(f"{dataset}_{algo}_{metric}_fair_"
        +params["exp_specific"  ])
    
    if dataset=="shakespeare":
        file_begin=(f"{dataset}_{algo}_{metric}_fair_"
            +f"{params['n_clients']}_{params['epochs']}_150"
            +f"_{params['lr']}")
    
    
    file_rd_simu=[file for file in file_rd_simu if file_begin in file ]
    
    hists_simu=[pickle.load(open(path+"/"+file,"rb")) for file in file_rd_simu]
    print(file_begin,len(hists_simu))

    #Get the server loss from the local one of each client
    hists_simu=[np.mean(hist,axis=1) for hist in hists_simu]

    return np.array(hists_simu)



def load_dataset_algo(dataset,algo,experiment_specific,params):
    
    dic={  
        "loss":load_metric(algo,"loss",dataset,experiment_specific,params),
        "acc":load_metric(algo,"acc",dataset,experiment_specific,params),
        "acc_hist":standard_deviation(algo,dataset,"acc",params),
        "loss_hist":standard_deviation(algo,dataset,"loss",params)
    }


    return dic



def plot_metric_for_all(FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,
        FA_shak_45,y_label,title=None,save_name=None,hist=True):
    

    cols = ["MNIST-iid","MNIST-shard","Shakespeare"]
    rows=["50% free-riders","90% free-riders"]
            
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,7))

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
                    size='large', ha='right', va='center',rotation =90) 
    
        
    def plot_dic(dic,plot_idx,metric):

        
        ax=axes[plot_idx//3][plot_idx%3]
        
        if metric=="acc":
            if hist:
                x=[i for i in range(len(dic[f"{metric}_hist"][0]))]
                min_hist=np.min(dic[f"{metric}_hist"],axis=0)
                max_hist=np.max(dic[f"{metric}_hist"],axis=0)                
                if plot_idx==2:
                    x=x[:len(dic[metric]["fair"])]
                    min_hist=min_hist[:len(dic[metric]["fair"])]
                    max_hist=max_hist[:len(dic[metric]["fair"])]
                

                ax.fill_between(x, min_hist,max_hist,alpha=0.3)
            

            ax.plot(dic[metric]["fair"],label="Only Fair")
            ax.plot(dic[metric]["none"],label="Plain")
            ax.plot(dic[metric]["add_1_1"],label=r"Disguised $\sigma$, $\gamma=1$")
            ax.plot(dic[metric]["add_3_1"],label=r"Disguised $3\sigma$, $\gamma=1$")
            ax.plot(dic[metric]["add_1_0.5"],label=r"Disguised $\sigma$, $\gamma=0.5$",linestyle='--')
            ax.plot(dic[metric]["add_3_0.5"],label=r"Disguised $3\sigma$, $\gamma=0.5$",linestyle='--')
            ax.plot(dic[metric]["add_1_2"],label=r"Disguised $\sigma$, $\gamma=2$",linestyle='--')
            ax.plot(dic[metric]["add_3_2"],label=r"Disguised $3\sigma$, $\gamma=2$",linestyle='--')
            
        
        if plot_idx%3==0:ax.set_ylim(85,90)
        elif plot_idx%3==1:ax.set_ylim(75,90)
        elif plot_idx%3==2:ax.set_ylim(30)
        
        if plot_idx==0:ax.legend(loc="lower right",ncol=2)
        
        ax.autoscale(enable=True, axis='x', tight=True)
        if plot_idx//3==1: ax.set_xlabel("# rounds")
        if plot_idx%3==0:ax.set_ylabel(y_label)

        
    metric="acc"
    
    plot_dic(FA_iid_5,0,metric)
    plot_dic(FA_shard_5,1,metric)
    plot_dic(FA_shak_5,2,metric)
    
    plot_dic(FA_iid_45,3,metric)
    plot_dic(FA_shard_45,4,metric)
    plot_dic(FA_shak_45,5,metric)
    
    fig.tight_layout()
    
    plt.savefig(f"plots/many_fr_{save_name}.png")

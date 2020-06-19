#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sn






def load_metric(algo,metric,dataset,experiment_specific,params):
    
    fair=pickle.load(open(f'variables/{dataset}_{algo}_{metric}_fair_{experiment_specific}.pkl', 'rb'))
    none=pickle.load(open(f'variables/{dataset}_{algo}_{metric}_none_{experiment_specific}.pkl', 'rb'))
    additive_1_1=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_1.0_{params['std_0']}_{experiment_specific}.pkl", 'rb'))    
    additive_3_1=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_1.0_{params['std_0']}_{experiment_specific}_3.pkl", 'rb'))
    additive_1_05=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_0.5_{params['std_0']}_{experiment_specific}.pkl", 'rb'))    
    additive_3_05=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_0.5_{params['std_0']}_{experiment_specific}_3.pkl", 'rb'))
    additive_1_2=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_2.0_{params['std_0']}_{experiment_specific}.pkl", 'rb'))    
    additive_3_2=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_2.0_{params['std_0']}_{experiment_specific}_3.pkl", 'rb'))

    
    
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


def standard_deviation(algo,dataset,metric,params):
    
    path=os.getcwd()+"/variables/random_simu"

    file_rd_simu=os.listdir(path)
    
    file_begin=(f"{dataset}_{algo}_{metric}_fair_"
        +params["exp_specific"  ])
    
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



def accuracy_params(n_epochs):
    
    fair=pickle.load(open(f'variables/MNIST-iid_FedAvg_acc_fair_5_600_5_300_0.0005.pkl', 'rb'))
    none=pickle.load(open(f'variables/MNIST-iid_FedAvg_acc_without_5_600_5_300_0.0005.pkl', 'rb'))
    additive_1_2=pickle.load(open(f"variables/MNIST-iid_FedAvg_acc_add_1_0.001_5_600_5_300_0.0005.pkl'", 'rb'))   
    
    return fair,none,additive_1_2




def plot_metric_for_all(FA_iid,FA_shard,FA_shak,FP_iid,FP_shard,FP_shak,
        y_label,title=None,save_name=None):
    
        
    cols = ["MNIST-iid","MNIST-shard","Shakespeare"]
    rows=["FedAvg","FedProx"]
            
    
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
                    size='large', ha='right', va='center',rotation=90) 
    
        
    def plot_dic(dic,plot_idx,metric):

        
        ax=axes[plot_idx//3][plot_idx%3]
        
        if metric=="acc":
            x=[i for i in range(len(dic[f"{metric}_hist"][0]))]
            min_hist=np.min(dic[f"{metric}_hist"],axis=0)
            max_hist=np.max(dic[f"{metric}_hist"],axis=0)
            
            
            ax.fill_between(x, min_hist,max_hist,alpha=0.3)

            ax.plot(x,dic[metric]["fair"],label="Only Fair")
            ax.plot(x,dic[metric]["none"],label="Plain")
            ax.plot(x,dic[metric]["add_1_1"],label=r"Disguised $\sigma$, $\gamma=1$")
            ax.plot(x,dic[metric]["add_3_1"],label=r"Disguised $3\sigma$, $\gamma=1$")
            ax.plot(x,dic[metric]["add_1_0.5"],label=r"Disguised $\sigma$, $\gamma=0.5$",linestyle='--')
            ax.plot(x,dic[metric]["add_3_0.5"],label=r"Disguised $3\sigma$, $\gamma=0.5$",linestyle='--')
            ax.plot(x,dic[metric]["add_1_2"],label=r"Disguised $\sigma$, $\gamma=2$",linestyle='--')
            ax.plot(x,dic[metric]["add_3_2"],label=r"Disguised $3\sigma$, $\gamma=2$",linestyle='--')
            
        
        if plot_idx%3==0:ax.set_ylim(85,90)
        elif plot_idx%3==1:ax.set_ylim(75)
        elif plot_idx%3==2:ax.set_ylim(30)
        
        if plot_idx==0:ax.legend(loc="lower right",ncol=2)
        
        ax.autoscale(enable=True, axis='x', tight=True)
        if plot_idx//3==1: ax.set_xlabel("# rounds")
        if plot_idx%3==0:ax.set_ylabel(y_label)

        
    metric="acc"
    
    plot_dic(FA_iid,0,metric)
    plot_dic(FA_shard,1,metric)
    plot_dic(FA_shak,2,metric)
    
    plot_dic(FP_iid,3,metric)
    plot_dic(FP_shard,4,metric)
    plot_dic(FP_shak,5,metric)    
    
    fig.tight_layout()
    
    plt.savefig(f"plots/{save_name}.png")



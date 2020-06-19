#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle
import os



def plot_iteration_2_alone(client_h,server_h,t): 
    """
    At iteration t, plot:
        - the parameters distribution of the server theta(t+1)-theta(t)
        - the parameters distribution sent by the free-rider theta_i(t)-theta(t)
    """
    plt.figure(figsize=(3.4,2.9))    

    n_tensors=len(server_h[0])       
    
    array_1=(server_h[t][0]-server_h[t-1][0]).reshape(-1)
    sn.kdeplot(array_1,gridsize=100000,label="Server")
    
    array_2=np.concatenate([client_h[t][0][k].reshape(-1)-server_h[t][k].reshape(-1)   for k in range(n_tensors)])
    sn.kdeplot(array_2,gridsize=100000,label="Free-rider")
    
    bord_min,bord_max=min(np.min(array_1),np.min(array_2)),max(np.min(array_1),np.max(array_2))
    border=min(abs(bord_min),bord_max) 
    
    plt.xlim(-border,border) 

    plt.xlabel("parameter difference") 
    plt.ylabel("probability density")
    plt.legend()
    plt.tight_layout()
    


def load_metric(algo,metric,dataset,experiment_specific,params):
    
    
    fair=pickle.load(open(f'variables/{dataset}_{algo}_{metric}_fair_{experiment_specific}.pkl', 'rb'))
    none=pickle.load(open(f'variables/{dataset}_{algo}_{metric}_none_{experiment_specific}.pkl', 'rb'))
    additive=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_1.0_{params['std_0']}_{experiment_specific}.pkl", 'rb'))    
    additive_extreme=pickle.load(open(f"variables/{dataset}_{algo}_{metric}_add_1.0_{params['std_0']}_{experiment_specific}_3.pkl", 'rb'))    
    
    dic={
        "fair":[np.mean(hist_i) for hist_i in fair],
        "none":[np.mean(hist_i) for hist_i in none],
        "add":[np.mean(hist_i) for hist_i in additive],
        "add_ext":[np.mean(hist_i) for hist_i in additive_extreme],
    } 
      
    return dic    



def confidence_interval(algo,dataset,metric,params):
    
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
        "acc_hist":confidence_interval(algo,dataset,"acc",params),
        "loss_hist":confidence_interval(algo,dataset,"loss",params)
    }

    return dic



from scipy.stats import ks_2samp
def get_vector_with_server(metric:str,client_h,server_h,t:int):
    
    n_clients=len(client_h[t])
    n_tensors=len(client_h[t][0])
    
    vector_h=list()
    for i in range(n_clients):
        
        vector_t=list()
        
        if metric=="KS":
            distri_1=np.concatenate([client_h[t][i][k].reshape(-1)  for k in range(n_tensors)]) #-server_h[t][k].reshape(-1) 
            distri_2=np.concatenate([server_h[t][k].reshape(-1)  for k in range(n_tensors)])
        
            vector_t.append(ks_2samp(distri_1,distri_2)[0])
            
        elif metric=="L2":
            vector_t.append(sum([(np.sum((client_h[t][i][k]-server_h[t][k]).reshape(-1)**2)) for k in range(n_tensors)]))
            
        vector_h.append(vector_t)
        
    return np.array(vector_h)   
    


def plot_metric_with_server(axes,metric,client_h,server_h,plot_idx):
    
    ax=axes[plot_idx//4,plot_idx%4]   
    
    n_iter=len(client_h)
    n_clients=len(client_h[0])
    
    metric_h=np.array([get_vector_with_server(metric,client_h,server_h,t) for t in range(n_iter)])
    
    xaxis=range(2,n_iter+1)
    
    for i in range(1,n_clients):
        if i==1:ax.plot(xaxis,metric_h[1:,i],color=sn.color_palette()[0],label="Fair clients")
        else:ax.plot(xaxis,metric_h[1:,i],color=sn.color_palette()[0])
    ax.plot(xaxis,metric_h[1:,0],color=sn.color_palette()[1],label="Free-rider")
        
        
    if plot_idx%4==0:ax.legend()
    else:ax.legend().remove()
    ax.set_yscale("log")
    
    ax.autoscale(enable=True, axis='x', tight=True)



def plot_acc(axes,dic,acc_h,plot_idx):
    
    ax=axes[plot_idx//4,plot_idx%4] 
    metric="acc"

    x=[i+1 for i in range(len(dic[f"{metric}_hist"][0])-1)]
    min_hist=np.min(dic[f"{metric}_hist"],axis=0)[1:101]
    max_hist=np.max(dic[f"{metric}_hist"],axis=0)[1:101]
                        
    ax.fill_between(x[:100], min_hist,max_hist,alpha=0.3)
    ax.plot(x[:100],dic[metric]["fair"][1:101],label="FedAvg")   
    ax.plot(x[:100],[np.mean(t) for t in acc_h][1:],label="Free-riding")

    if plot_idx==8:ax.legend()
    ax.set_ylim(75)
    
    ax.autoscale(enable=True, axis='x', tight=True)



def plot_metric_evolution_with_server(FA_shard,c1,s1,a1,c2,s2,a2,c3,s3,a3,c4,s4,
        a4,title=None,save_name=None):
    """ci is the history of the clients of experiment i
    si is the hisotry of the global model of experiment i"""
    
    print(save_name)
    cols = ["(A)Gaussian","(B)GMM","(C)GMM +outliers","(D)GMM + outliers \n+ recalibration"]
    rows=["KS - log scale","L2 norm - log scale","Accuracy"]
    
    sn.set_style("ticks")
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14,8))

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
    
    plot_metric_with_server(axes,"KS",c1,s1,0)
    plot_metric_with_server(axes,"KS",c2,s2,1)
    plot_metric_with_server(axes,"KS",c3,s3,2)
    plot_metric_with_server(axes,"KS",c4,s4,3)    
    
    plot_metric_with_server(axes,"L2",c1,s1,4)
    plot_metric_with_server(axes,"L2",c2,s2,5)
    plot_metric_with_server(axes,"L2",c3,s3,6)
    plot_metric_with_server(axes,"L2",c4,s4,7)
    
    plot_acc(axes,FA_shard,a1,8)
    plot_acc(axes,FA_shard,a2,9)
    plot_acc(axes,FA_shard,a3,10)
    plot_acc(axes,FA_shard,a4,11)
    
    plt.tight_layout()
    
    if save_name: plt.savefig(f"plots/metric_evolution_simple_{save_name}.png")































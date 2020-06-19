#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from copy import deepcopy
import torch.optim as optim
import numpy as np



def get_n_params(model):
    """ return the number of parameters in the model"""
    
    n_params=sum([np.prod(tensor.size()) for tensor in list(model.parameters())])
    return n_params


def linear_noising(model,list_std,list_power,iteration,noise_shape,std_multiplicator):
    """Return the noised model of the free-rider"""
    
    if noise_shape=="linear":
        for idx,layer_tensor in enumerate(model.parameters()):
            
            mean_0=torch.zeros(layer_tensor.size())
            std_tensor=torch.zeros(layer_tensor.size())+std_multiplicator*list_std[1]*iteration**(-list_power[1])
            noise_additive=torch.normal(mean=mean_0,std=std_tensor)

            layer_tensor.data+=noise_additive
    
        return model


    elif noise_shape=="exp":
        for idx,layer_tensor in enumerate(model.parameters()):
        
            mean_0=torch.zeros(layer_tensor.size())
            std_tensor=torch.zeros(layer_tensor.size())+std_multiplicator*list_std[1]*np.exp(-(iteration-1)*list_power[1])
            noise_additive=torch.normal(mean=mean_0,std=std_tensor)
            
            layer_tensor.data+=noise_additive
    
            
        return model



def linear_noising_mixture(m_current,m_previous,list_std,list_power,iteration,
        associated_gaussian,noise):
    """Return a noised model
    parameter=prameter*(1+N(0,std^2/t^alpha))+N(0,std^2/t^alpha)
    For the lists, first multipliative then additive"""
                
    if noise=="add": 
        
        std_add=[std[1]/iteration**(list_power[1])for std in list_std]
        print("stds 'add' at this iteration: ",std_add)
        
        for idx,layer_tensor in enumerate(m_current.parameters()):
            
            for i in range(len(list_std)):
                
                mean_0=torch.zeros(layer_tensor.size())
                std_tensor_i=torch.zeros(layer_tensor.size())+std_add[i]
                noise_additive_i=torch.normal(mean=mean_0,std=std_tensor_i)
                
                layer_tensor.data+=((noise_additive_i)*(associated_gaussian[idx]==i))
        
    return m_current



def linear_noising_4(m_current,m_previous,list_std,list_power,iteration,
        associated_gaussian,noise,noise_shape):
    """Return a noised model
    parameter=prameter*(1+N(0,std^2/t^alpha))+N(0,std^2/t^alpha)
    For the lists, first multipliative then additive"""
    
    if noise=="add": attribution_add=associated_gaussian
                
    if noise=="add" or noise=="comb": 

        if noise_shape=="exp":std_add=[std[1]*np.exp(-power[1]/2*(iteration-1))for std,power in zip(list_std,list_power)]
        elif noise_shape=="linear":std_add=[std[1]/iteration**(power[1]/2)for std,power in zip(list_std,list_power)]
        
        print("stds 'add' at this iteration: ",std_add)
        
        for idx,layer_tensor in enumerate(m_current.parameters()):
            
            for i in range(len(list_std)):
                
                mean_0=torch.zeros(layer_tensor.size())
                std_tensor_i=torch.zeros(layer_tensor.size())+std_add[i]
                noise_additive_i=torch.normal(mean=mean_0,std=std_tensor_i)
                
                layer_tensor.data+=((noise_additive_i)*(attribution_add[idx]==i))
        
    return m_current


def get_std(model_A,model_B,noise):
    
    list_tens_A=[tens_param.detach() for tens_param in list(model_A.parameters())]
    list_tens_B=[tens_param.detach() for tens_param in list(model_B.parameters())]
    
    if noise=="naive":
        return [0,0]
    
    
    sum_abs_diff=0
    sum_abs=0
    
    for i in range(len(list_tens_A)):
            sum_abs_diff+=torch.sum(torch.abs(list_tens_A[i]-list_tens_B[i]))
            sum_abs+=torch.sum(torch.abs(list_tens_B[i])) 
    
    if noise=="add":
        
        
        std=sum_abs_diff/get_n_params(model_A)
        
        return [0,std]
    elif noise=="multi":
        
        std=sum_abs_diff/sum_abs
        return [std,0]
    
    elif noise=="comb":
        
        std_1=sum_abs_diff/sum_abs
        std_2=sum_abs_diff/get_n_params(model_A)
        
        return [std_1,std_2]
    


from sklearn.mixture import GaussianMixture   
def get_std_mixture(model_A,model_B,noise,outliers,n_components=2):
    """model_A is the current global model and model_B is the intial model"""
    
    if noise=="naive":
        return [0,0]
    

    list_tens_A=[tens_param.detach() for tens_param in list(model_A.parameters())]
    list_tens_B=[tens_param.detach() for tens_param in list(model_B.parameters())]

    #List of all the parameters absolute value differences. 
    list_diff=list()
    for i in range(len(list_tens_A)):
        list_diff+=list((list_tens_A[i]-list_tens_B[i]).numpy().reshape(-1))
        
    list_diff_relative=list()
    for i in range(len(list_tens_A)):
        list_diff_relative+=list(((list_tens_A[i]-list_tens_B[i])/list_tens_A[i]).numpy().reshape(-1))
        
    mean_A=np.mean(np.concatenate(list([torch.abs(list_tens_A[i]).numpy().reshape(-1) for i in range(len(list_tens_A))])))
    mean_A=float(mean_A)
    print(mean_A)
    
    print("number of gaussians in the mixture: ",n_components)
    
    gmm=GaussianMixture(n_components=n_components,max_iter = 1000,tol=1e-20,covariance_type="spherical",reg_covar=10**-100)
            
    if noise=="add":
        
        X_add=np.array(list_diff).reshape(-1,1)  
        gmm.fit(X_add)
        covs_add=gmm.covariances_
        stds_add=np.sqrt(covs_add)

        associated_gaussian_add=[]        
        for tens_A,tens_B in zip(list_tens_A,list_tens_B):
            
            diff=(tens_A-tens_B).numpy().reshape(-1,1)
            
            attribution_i=torch.Tensor(gmm.predict(diff).reshape(tens_A.shape))
            associated_gaussian_add.append(attribution_i)
            
        if outliers:
            
            
            #put 10% of the valeus in the peak in the tail.
            max_std=np.argmax(gmm.covariances_)
            index_min=(associated_gaussian_add[0]==max_std).nonzero()
            
            idx_kept = torch.randperm(len(index_min))[:10]
            index_min=index_min[idx_kept]
            print(torch.sum(associated_gaussian_add[0]==max_std))
            for idx in index_min:
                associated_gaussian_add[0][idx[0],idx[1]]=n_components
            print(torch.sum(associated_gaussian_add[0]==max_std))
            
            stds_add=np.array(list(stds_add)+[np.max(stds_add)*25])
        
        return [[0,std] for std in stds_add],associated_gaussian_add



def FL_freeloader(model_training,training_dls, n_fr_samples,
        n_iter,testing_set, loss_f,device,mu,file_begin,noise_shape,multiplicator,
        noise_f=linear_noising,noise="naive",epochs=5,lr=10**-4,std_0=0,power=1,
        decay=1):
    """
    Parameters:
        - `model` common structure used by the clients and the server
        - `training_dls`: list of the training sets. At each index is the 
            trainign set of client "index"
        - `n_fr_samples`: number of samples the free-rider pretends having
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `loss_f`: loss function applied to the output of the model
        - `device`: whether the simulation is run on GPU or CPU
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `file_begin`: name that will start all the files saving the global 
            model at every iteration
        - `noise_shape`: noise decreasing function: linear or exp
        - `multiplicator`: number of times the std of the heuristic is multiplied
        - `noise_f`: function use to noise the global model
        - 'noise`: if noise then 'add' if not `naive`
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `std_0`: original std used at iteration 0 before the use of the
            heuristic
        - `power`: gamma in the paper for the noise
        - `decay`: to change the learning rate at each iteration
    
    returns :
        - `model`: the final global model 
        - `loss_hist`: the loss at each iteration of all the clients
        - `acc_hist`: the accuracy at each iteration of all the clients
        - `list_std`: the std computed by the heuristic
    """
    
    from FL_functions import loss_dataset,accuracy_dataset,local_learning,FedAvg_agregation_process
        
    #Variables initialization
    K=len(training_dls) #number of clients
    n_samples=n_fr_samples+sum([len(db.dataset) for db in training_dls])
    weights=[n_fr_samples/n_samples]+[len(db.dataset)/n_samples for db in training_dls]
    print("clients' weights:",weights)
    
    
    loss_hist=[]
    acc_hist=[]
    m_previous=deepcopy(model_training)
    
    
    if noise=="naive":list_std=[0,0]
    elif noise=="add":list_std=[0,std_0]
    list_power=[0,power]
    
    
    loss_hist=[[float(loss_dataset(model_training,testing_set[k],loss_f).detach()) for k in range(K)]]
    acc_hist=[[accuracy_dataset(model_training,testing_set[k]) for k in range(K)]]
    
    
    for i in range(n_iter):
        
        clients_params=[]
        
        
        #WORK DONE BY THE FREE-RIDER
        if i==0:
            local_model=noise_f(deepcopy(model_training),list_std,list_power,1,noise_shape,multiplicator).to(device)
        
        elif i==1:
            list_std=get_std(model_training,m_previous,noise)
            print("noise std",list_std)
        
            local_model=noise_f(deepcopy(model_training),list_std,list_power,1,noise_shape,multiplicator).to(device)
            
        elif i>=1:
            
            local_model=noise_f(deepcopy(model_training),list_std,list_power,i,noise_shape,multiplicator).to(device)


        list_params=list(local_model.parameters())
        list_params=[tens_param.detach() for tens_param in list_params]
        clients_params.append(list_params)
        
#        torch.save(local_model.state_dict(),f"saved_models/{file_begin}_{i}_fr.pth" )
        
        
        
        #WORK DONE BY THE FAIR CLIENTS
        for k in range(K):
            
            local_model=deepcopy(model_training).to(device)
            local_optimizer=optim.SGD(local_model.parameters(),lr=lr)
            
            local_learning(local_model,mu,local_optimizer,training_dls[k],epochs,loss_f,device)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            
#            torch.save(local_model.state_dict(),f"saved_models/{file_begin}_{i}_{k}.pth" )
            
            
        #CREATE THE NEW GLOBAL MODEL
        new_model=FedAvg_agregation_process(deepcopy(model_training),clients_params,device,
            weights=weights).cpu()
        
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(new_model,training_dls[k],loss_f).detach()) for k in range(K)]]
        acc_hist+=[[accuracy_dataset(new_model,testing_set[k]) for k in range(K)]]


        server_loss=np.mean(loss_hist[-1][1:])
        acc_loss=np.mean(acc_hist[-1][1:])
        print(f'====> i: {i} Loss: {server_loss} Accuracy: {acc_loss}')
        
        
        m_previous=deepcopy(model_training)
        model_training=new_model
        torch.save(model_training.state_dict(),f"saved_models/{file_begin}_{i}_server.pth" )
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay
    
        
            
    return model_training,np.array(loss_hist),np.array(acc_hist),list_std






def FL_freeloader_many(n_freeriders,model_training,training_dls, n_fr_samples,
        n_iter,testing_set, loss_f,device,mu,file_begin,noise_shape,multiplicator,noise_f=linear_noising,noise="naive",
        epochs=5,lr=10**-4,std_0=0,power=1,
        decay=1):
    """
    Compare the model obtained at each iteration to the `model_comparison`
    Also, only the client 2 partiipates. The client 1 returns the same model 
    as the one received.
    """
    
    from FL_functions import loss_dataset,accuracy_dataset,local_learning,FedAvg_agregation_process
        
    #Variables initialization
    K=len(training_dls) #number of clients
    n_samples=n_freeriders*n_fr_samples+sum([len(db.dataset) for db in training_dls])
    weights=[n_fr_samples/n_samples]*n_freeriders+[len(db.dataset)/n_samples for db in training_dls]
    print(weights)
    
    loss_hist=[]
    acc_hist=[]
    m_previous=deepcopy(model_training)
    
    if noise=="naive":list_std=[0,0]
    elif noise=="add":list_std=[0,std_0]
    
    list_power=[0,power]
    
    loss_hist=[[float(loss_dataset(model_training,testing_set[k],loss_f).detach()) for k in range(K)]]
    acc_hist=[[accuracy_dataset(model_training,testing_set[k]) for k in range(K)]]

    for i in range(n_iter):
        
        if i==1:
            list_std=get_std(model_training,m_previous,noise)
            print(list_std)
        

        #WORK DONE BY THE FREE-RIDERS
        clients_params=[]
        
        for j in range(n_freeriders):
        
            local_model=noise_f(deepcopy(model_training),list_std,list_power,max(i,1),noise_shape,multiplicator).to(device)
    
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            
#            torch.save(local_model.state_dict(),f"saved_models/{file_begin}_{i}_fr{j}.pth" )
        
        
        
        #WORK DONE BY THE FAIR CLIENTS
        for k in range(K):
            local_model=deepcopy(model_training).to(device)
            local_optimizer=optim.SGD(local_model.parameters(),lr=lr)
            
            local_learning(local_model,mu,local_optimizer,training_dls[k],epochs,loss_f,device)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            
#            torch.save(local_model.state_dict(),f"saved_models/{file_begin}_{i}_{k}.pth" )
            
            
        #CREATE THE NEW GLOBAL MODEL
        new_model=FedAvg_agregation_process(deepcopy(model_training),clients_params,device,
            weights=weights).cpu()
        
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT FAIR CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(new_model,training_dls[k],loss_f).detach()) for k in range(K)]]
        acc_hist+=[[accuracy_dataset(new_model,testing_set[k]) for k in range(K)]]
        
        
        print(loss_hist[-1])
        
        server_loss=np.mean(loss_hist[-1][1:])
        acc_loss=np.mean(acc_hist[-1][1:])
        print(f'====> i: {i} Loss: {server_loss} Accuracy: {acc_loss}')
        
        m_previous=deepcopy(model_training)
        model_training=new_model
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay
        
        torch.save(model_training.state_dict(),f"saved_models/{file_begin}_{i}_server.pth" )
    
        
            
    return model_training,np.array(loss_hist),np.array(acc_hist),list_std








def FL_freeloader_mixture(model_training,model_comparison,training_dls, n_fr_samples,
        n_iter,testing_set, loss_f,device,mu,noise_shape,noise_f=linear_noising,noise="naive",
        epochs=5,lr=10**-4,std_0=0,power=1,cycle=1000,outliers=False,
        decay=1,n_components=2):
    """
    Compare the model obtained at each iteration to the `model_comparison`
    Also, only the client 2 partiipates. The client 1 returns the same model 
    as the one received.
    """
    
    from FL_functions import loss_dataset,accuracy_dataset,local_learning,FedAvg_agregation_process
        
    #Variables initialization
    K=len(training_dls) #number of clients
    n_samples=n_fr_samples+sum([len(db.dataset) for db in training_dls])
    weights=[n_fr_samples/n_samples]+[len(db.dataset)/n_samples for db in training_dls]
    print(weights)
    
    loss_hist=[]
    acc_hist=[]
    m_previous=deepcopy(model_training)
    m_current=deepcopy(model_training)
    client_hist=[]
    
    server_hist=[[tens_param.detach().numpy() for tens_param in list(m_current.parameters())]]
    
    if noise=="naive":list_std=[0,0]
    elif noise=="add":list_std=[0,std_0]
    
    list_power=[power,power]
    
    loss_hist=[[float(loss_dataset(model_training,testing_set[k],loss_f).detach()) for k in range(K)]]
    acc_hist=[[accuracy_dataset(model_training,testing_set[k]) for k in range(K)]]
    
    
    for i in range(n_iter):
        
        clients_params=[]
        
                
        if i%cycle==1:
            list_std,associated_gaussian=get_std_mixture(m_current,deepcopy(m_previous),noise,outliers,n_components)
            print("stds for the gaussian mixture:",list_std)       
        
        
        if i==0:
            local_model=noise_f(deepcopy(m_current),list_std,list_power,1,noise_shape,1).to(device)    

        elif i>=1:
            local_model=linear_noising_mixture(deepcopy(m_current),deepcopy(m_previous),
                list_std,list_power,i%cycle+(i%cycle==0)*cycle,associated_gaussian,noise).to(device)


        list_params=list(local_model.parameters())
        list_params=[tens_param.detach() for tens_param in list_params]
        clients_params.append(list_params)
        
        
        
        #WORK DONE BY THE PARTICIPATING CLIENTS
        for k in range(K):
            local_model=deepcopy(m_current).to(device)
            local_optimizer=optim.SGD(local_model.parameters(),lr=lr)
            
            local_learning(local_model,mu,local_optimizer,training_dls[k],epochs,loss_f,device)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            
        #CREATE THE NEW GLOBAL MODEL
        new_model=FedAvg_agregation_process(model_training,clients_params,device,
            weights=weights).cpu()
        
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(new_model,training_dls[k],loss_f).detach()) for k in range(K)]]
        acc_hist+=[[accuracy_dataset(new_model,testing_set[k]) for k in range(K)]]

        
        server_loss=np.mean(loss_hist[-1][1:])
        server_acc=np.mean(acc_hist[-1][1:])
        print(f'====> i: {i} Loss: {server_loss} Acc: {server_acc}')
        
        
        m_previous=deepcopy(m_current)
        m_current=new_model
        
        
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay
        
        for i in range(len(clients_params)):
            for j in range(len(clients_params[0])):
                clients_params[i][j]=clients_params[i][j].cpu().detach().numpy()
                
        client_hist.append(clients_params)
    
        server_hist.append([tens_param.detach().cpu().numpy() for tens_param in list(new_model.parameters())])
    
        
    return server_hist,client_hist,loss_hist,acc_hist

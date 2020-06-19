#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from functools import reduce



def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)
    
    
        
def client_contribution(new_global_model,local_model,local_model_weight):
    """global_model=global_model+model_weight*local_model"""
    
    local_model_parameters=list(local_model.parameters())
    for idx, layer_weights in enumerate(new_global_model.parameters()):
        
        layer_weights.data.add_(local_model_parameters[idx].data*local_model_weight)


        
def FedAvg_agregation_process(model,clients_models_hist,device,weights=None):
    """Creates the new model of a given iteration with the models of the other
    clients"""
    
    new_model=deepcopy(model).to(device)
    
    set_to_zero_model_weights(new_model)

    
    for k,client_hist in enumerate(clients_models_hist):
        
        for idx, layer_weights in enumerate(new_model.parameters()):
            
            if weights!=None:
                contribution=client_hist[idx].data*weights[k]
            else: 
                contribution=client_hist[idx].data/len(clients_models_hist)

            layer_weights.data.add_(contribution)
            
    return new_model
            
     

def accuracy_dataset(model,dataset):
    """Compute the accuracy of `model` on `test_data`"""
    
    correct=0
    
    for features,labels in iter(dataset):
        
        predictions= model(features)
        
        _,predicted=predictions.max(1,keepdim=True)
        
        correct+=torch.sum(predicted.view(-1)==labels).item()
        
    accuracy = 100*correct/len(dataset.dataset)
        
    return accuracy



def loss_dataset(model,train_data,loss_f):
    """Compute the loss of `model` on `test_data`"""
    loss=0
    
    for idx,(features,labels) in enumerate(train_data):
        
        predictions= model(features)
        loss+=loss_f(predictions,labels)
    
    loss/=(idx+1)
    return loss



def n_params(model):
    """ return the number of parameters in the model"""
    
    n_params=0
    for tensor in list(model.parameters()):
        
        n_params_tot=1
        for k in range(len(tensor.size())):
            n_params_tot*=tensor.size()[k]
            
        n_params+=n_params_tot
    
    return n_params



def difference_models_norm_2(model_1,model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) for i in range(len(tensor_1))])
    
    return norm



def train_step(model,model_0,mu,optimizer,train_data,loss_f,device):
    """Train `model` on one epoch of `train_data`
    """
    
    model.train()
    
    for idx, (features,labels) in enumerate(train_data):
        
        optimizer.zero_grad()
        
        features,labels=features.to(device),labels.to(device)
        
        predictions= model(features)
        predictions=F.log_softmax(predictions, dim=1)

        loss=loss_f(predictions,labels)
        
        loss+=mu/2*difference_models_norm_2(model,model_0)
        
        loss.backward()
        
        optimizer.step()



def local_learning(model,mu,optimizer,train_data,epochs,loss_f,device):
    
    model_0=deepcopy(model)
    
    for e in range(epochs):
        train_step(model,model_0,mu,optimizer,train_data,loss_f,device)
   
     
        
def difference_model(model_1,model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=0
    
    for i in range(len(tensor_1)):
        
        norm+=torch.sum(torch.abs(tensor_1[i]-tensor_2[i]))
        
    #Get the number of parameters in the model
    norm/=n_params(model_1)
    
    return norm.detach().numpy()
        


def FedAvg(model,training_sets,n_iter,loss_f,
        testing_set, device,mu,file_begin,epochs=5,lr=10**-4,decay=1):
    """
    Parameters:
        - `model` common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the 
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `device`: whether the simulation is run on GPU or CPU
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `file_begin`: name that will start all the files saving the global 
            model at every iteration
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration
    
    returns :
        - `model`: the final global model 
        - `loss_hist`: the loss at each iteration of all the clients
        - `acc_hist`: the accuracy at each iteration of all the clients
    """
        
    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=[len(db.dataset)/n_samples for db in training_sets]
    print("Clients' weights:",weights)
    
    
    loss_hist=[[float(loss_dataset(model,testing_set[k],loss_f).detach()) for k in range(K)]]
    acc_hist=[[accuracy_dataset(model,testing_set[k]) for k in range(K)]]
    
    
    for i in range(n_iter):
        
        clients_params=[]
        
        for k in range(K):
            
            local_model=deepcopy(model).to(device)
            local_optimizer=optim.SGD(local_model.parameters(),lr=lr)
            
            local_learning(local_model,mu,local_optimizer,training_sets[k],
                epochs,loss_f,device)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)     
#            torch.save(local_model.state_dict(),f"saved_models/{file_begin}_{i}_{k}.pth" )
        
        
        #CREATE THE NEW GLOBAL MODEL
        new_model=FedAvg_agregation_process(deepcopy(model),clients_params,device, weights=weights).cpu()
        
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(new_model,training_sets[k],loss_f).detach()) for k in range(K)]]
        acc_hist+=[[accuracy_dataset(new_model,testing_set[k]) for k in range(K)]]

        
        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
        
        print(f'====> i: {i} Loss: {server_loss} Server Accuracy: {server_acc}')
        
        model=new_model
        torch.save(model.state_dict(),f"saved_models/{file_begin}_{i}_server.pth" )
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay
            
    return model,np.array(loss_hist),np.array(acc_hist)


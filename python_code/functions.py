#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle 
import os
import torch


def save_variable(variable, name):
    """Function to reduce the number of lines when saving a pickle file
    from the variables repository"""
    
    with open(f'variables/{name}.pkl', 'wb') as output:
        pickle.dump(variable, output)      



def load_initial_model(dataset):
    """Function that helps keep the same initial model for all the simulations
    If an initial has never been used before creates one that will be saved in 
    variables"""
    
    from models import MultinomialLogisticRegression
    from models import LSTM_Shakespeare
    
    if dataset=="MNIST-iid" or dataset=="MNIST-shard":
        m_initial=MultinomialLogisticRegression()
    
        if os.path.exists("variables/model_MNIST_0.pth"):
            print(f"initial model for {dataset} already exists")
            m_initial_dic=torch.load("variables/model_MNIST_0.pth")
            m_initial.load_state_dict(m_initial_dic)
    
            
        else:
            print(f"initial model for {dataset} does not exist")
            torch.save(m_initial.state_dict(), "variables/model_MNIST_0.pth")
            
    if dataset=="shakespeare":
        m_initial=LSTM_Shakespeare()
        
        if os.path.exists("variables/model_shakespeare_0.pth"):
            print(f"initial model for {dataset} already exists")
            m_initial_dic=torch.load("variables/model_shakespeare_0.pth")
            m_initial.load_state_dict(m_initial_dic)
    
            
        else:
            print(f"initial model for {dataset} does not exist")
            torch.save(m_initial.state_dict(), "variables/model_shakespeare_0.pth")
        
    return m_initial




def exist(file_name):
    """check if a file exists"""
    
    return os.path.exists(file_name)
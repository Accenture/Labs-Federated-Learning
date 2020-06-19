#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn


#MULTINOMIAL LOGISTIC REGRESSION FOR MNIST
class MultinomialLogisticRegression(torch.nn.Module):
    def __init__(self):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784,10)

    def forward(self, x):
        outputs = self.linear(x.view(-1,784))
        return outputs


def loss_MNIST(predictions,labels):
    
    predictions_soft=F.log_softmax(predictions,dim=1)

    loss=F.nll_loss(predictions_soft,labels.view(-1))
    
    return loss


class LSTM_Shakespeare(torch.nn.Module):
    def __init__(self):
        super(LSTM_Shakespeare,self).__init__()
        self.n_characters=100
        self.hidden_dim=100
        self.n_layers=2
        self.len_seq=80
        self.batch_size=100
        self.embed_dim=8
        
        self.embed=torch.nn.Embedding(self.n_characters,self.embed_dim)
        
        self.lstm=torch.nn.LSTM(self.embed_dim,self.hidden_dim,self.n_layers,batch_first=True)
        
        self.fc=nn.Linear(self.hidden_dim, self.n_characters)

        
    def forward(self,x):

        embed_x=self.embed(x)
        output,_=self.lstm(embed_x)
        output=self.fc(output[:,-1])
        return output
        

    def init_hidden(self, batch_size):

        self.hidden=(torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim))





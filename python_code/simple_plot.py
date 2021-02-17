#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# In[2]: 
import pickle
fair=pickle.load(open(f'hist/acc/MNIST-iid_FedAvg_FL_5_600_5_200_0.001.pkl', 'rb'))
none=pickle.load(open(f'hist/acc/MNIST-iid_FedAvg_plain_1_5_600_5_200_0.001.pkl', 'rb'))
additive=pickle.load(open(f'hist/acc/MNIST-iid_FedAvg_disguised_1.0_0.001_1_5_600_5_200_0.001_1.pkl', 'rb'))   

# In[2]: 
import numpy as np
fair=[np.mean(acc_t) for acc_t in fair]
none=[np.mean(acc_t) for acc_t in none]
additive=[np.mean(acc_t) for acc_t in additive]


# In[2]: 
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fair,label="Only Fair")
plt.plot(none,label="Plain")
plt.plot(additive, label=r"Disguised $\sigma$ $\gamma=1$")
plt.ylim(85,92)
plt.legend()
plt.title("MNIST-iid FedAvg E=5")
plt.ylabel("Accuracy")
plt.xlabel("# rounds")
plt.savefig("plots/simple_experiment.png")


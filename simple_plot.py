#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[2]: 
import subprocess

rc = subprocess.call("./generate_3_scenarios.sh", shell=True)


# In[2]: 
import pickle
fair=pickle.load(open(f'variables/MNIST-iid_FedAvg_acc_fair_5_600_5_300_0.0005.pkl', 'rb'))
none=pickle.load(open(f'variables/MNIST-iid_FedAvg_acc_none_5_600_5_300_0.0005.pkl', 'rb'))
additive=pickle.load(open(f'variables/MNIST-iid_FedAvg_acc_add_1.0_0.001_5_600_5_300_0.0005.pkl', 'rb'))   

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
plt.ylim(85,90)
plt.legend()
plt.title("MNIST-iid FedAvg E=5")
plt.ylabel("Accuracy")
plt.xlabel("# rounds")
plt.savefig("plots/simple_experiments.png")


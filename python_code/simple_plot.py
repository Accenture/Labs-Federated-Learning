#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# In[2]:
import pickle

acc_fair = pickle.load(open(f"hist/acc/MNIST-iid_FedAvg_FL_5_600_5_50_0.001.pkl", "rb"))
acc_plain = pickle.load(
    open(f"hist/acc/MNIST-iid_FedAvg_plain_1_5_600_5_50_0.001.pkl", "rb")
)
acc_disg = pickle.load(
    open(
        f"hist/acc/MNIST-iid_FedAvg_disguised_1.0_0.001_1_5_600_5_50_0.001_1.pkl",
        "rb",
    )
)

loss_fair = pickle.load(
    open(f"hist/loss/MNIST-iid_FedAvg_FL_5_600_5_50_0.001.pkl", "rb")
)
loss_plain = pickle.load(
    open(f"hist/loss/MNIST-iid_FedAvg_plain_1_5_600_5_50_0.001.pkl", "rb")
)
loss_disg = pickle.load(
    open(
        f"hist/loss/MNIST-iid_FedAvg_disguised_1.0_0.001_1_5_600_5_50_0.001_1.pkl",
        "rb",
    )
)


import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
fig.suptitle("MNIST-iid FedAvg E=5")

axes[0].plot([np.mean(acc_t) for acc_t in acc_fair], label="Only Fair")
axes[0].plot([np.mean(acc_t) for acc_t in acc_plain], label="Plain")
axes[0].plot(
    [np.mean(acc_t) for acc_t in acc_disg], label=r"Disguised $\sigma$ $\gamma=1$"
)
axes[0].set_ylim(70, 92)
axes[0].legend()
axes[0].set_ylabel("Accuracy")
axes[0].set_xlabel("# rounds")


axes[1].plot([np.mean(loss_t) for loss_t in loss_fair])
axes[1].plot([np.mean(loss_t) for loss_t in loss_plain])
axes[1].plot([np.mean(loss_t) for loss_t in loss_disg])
axes[1].set_ylabel("Loss")
axes[1].set_xlabel("# rounds")
axes[1].set_yscale("log")

fig.savefig("plots/simple_experiment.png")
plt.show()

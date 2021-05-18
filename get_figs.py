#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append("./python_code")


# In[2]:
"""PLOTS FOR THE CORE OF THE PAPER"""
from python_code.functions_plot import plot_fig_1_half

plot_fig_1_half("loss", 20)
plot_fig_1_half("acc", 20)


# In[2]:
"""PLOTS FOR THE APPENDIX"""
from python_code.functions_plot import plot_metric_history
from itertools import product
import matplotlib.pyplot as plt

for solver, n_epochs in product(["FedAvg", "FedProx"], [5, 20]):
    plot_metric_history(solver, n_epochs, "acc", False)
    plt.close()
    plot_metric_history(solver, n_epochs, "loss", True)
    plt.close()

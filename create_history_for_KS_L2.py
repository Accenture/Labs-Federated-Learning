#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./python_code')

# In[2]:
"""Figure 1"""
from python_code.functions_create_history_metric import KS_L2_history_fig1

#FedAvg and FedProx for E=20
KS_L2_history_fig1(20)

#FedAvg and Fedprox for E=5
KS_L2_history_fig1(5)


# In[2]:
"""Figure 2"""
from python_code.functions_create_history_metric import KS_L2_history_fig2

#FedAvg and FedProx for E=20
KS_L2_history_fig2(20)

#FedAvg and Fedprox for E=5
KS_L2_history_fig2(5)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./python_code')

# In[2]: 
"""Accuracies"""
from python_code.functions_plot_fig1_accuracy import figure_1_accuracy_params
from python_code.functions_plot_fig1_accuracy import plot_metric_for_all

#Accuracy 20 epochs
n_epochs=20
FA_iid,FP_iid,FA_shard,FP_shard,FA_shak,FP_shak=figure_1_accuracy_params(n_epochs)
plot_metric_for_all(FA_iid,FA_shard,FA_shak,FP_iid,FP_shard,FP_shak, y_label="Accuracy",
    save_name=f"accuracy_{n_epochs}")

n_epochs=5
FA_iid,FP_iid,FA_shard,FP_shard,FA_shak,FP_shak=figure_1_accuracy_params(n_epochs)
plot_metric_for_all(FA_iid,FA_shard,FA_shak,FP_iid,FP_shard,FP_shak, y_label="Accuracy",
    save_name=f"accuracy_{n_epochs}")



# In[2]: 
"""KS and L2"""
from python_code.functions_plot_fig_1_2_KS_L2 import KS_L2_single_freerider
from python_code.functions_plot_fig_1_2_KS_L2 import plot_KS_L2

#L2 20 epochs
FA_iid_L2,FP_iid_L2,FA_shard_L2,FP_shard_L2,FA_shak_L2,FP_shak_L2=KS_L2_single_freerider("L2",20)
plot_KS_L2(FA_iid_L2,FA_shard_L2,FA_shak_L2,FP_iid_L2,FP_shard_L2,FP_shak_L2, y_label="L2",save_name="fig_1_L2_20",fig=1,legend_idx=2)

#Ks 20 epochs
FA_iid_KS,FP_iid_KS,FA_shard_KS,FP_shard_KS,FA_shak_KS,FP_shak_KS=KS_L2_single_freerider("KS",20)
plot_KS_L2(FA_iid_KS,FA_shard_KS,FA_shak_KS,FP_iid_KS,FP_shard_KS,FP_shak_KS, y_label="KS",save_name="fig_1_KS_20",fig=1,legend_idx=4)

#L2 5 epochs
FA_iid_L2,FP_iid_L2,FA_shard_L2,FP_shard_L2,FA_shak_L2,FP_shak_L2=KS_L2_single_freerider("L2",5)
plot_KS_L2(FA_iid_L2,FA_shard_L2,FA_shak_L2,FP_iid_L2,FP_shard_L2,FP_shak_L2, y_label="L2",save_name="fig_1_L2_5",fig=1,legend_idx=2)

#Ks 5 epochs
FA_iid_KS,FP_iid_KS,FA_shard_KS,FP_shard_KS,FA_shak_KS,FP_shak_KS=KS_L2_single_freerider("KS",5)
plot_KS_L2(FA_iid_KS,FA_shard_KS,FA_shak_KS,FP_iid_KS,FP_shard_KS,FP_shak_KS, y_label="KS",save_name="fig_1_KS_5",fig=1,legend_idx=4)

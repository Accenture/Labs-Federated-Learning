#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./python_code')

# In[2]: 
"""Accuracies"""
from python_code.functions_plot_fig2_accuracy import params_accuracy_plot_many_freerider
from python_code.functions_plot_fig2_accuracy import plot_metric_for_all

#FedAvg 20 epochs
FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,FA_shak_45=params_accuracy_plot_many_freerider("FedAvg",20)
plot_metric_for_all(FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,
    FA_shak_45,y_label="Accuracy",save_name=f"accuracy_20_FedAvg")

#FedProx 20 epochs
FP_iid_5,FP_iid_45,FP_shard_5,FP_shard_45,FP_shak_5,FP_shak_45=params_accuracy_plot_many_freerider("FedProx",20)
plot_metric_for_all(FP_iid_5,FP_iid_45,FP_shard_5,FP_shard_45,FP_shak_5,
    FP_shak_45,y_label="Accuracy",save_name=f"accuracy_20_FedProx")

#FedAvg 5 epochs
FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,FA_shak_45=params_accuracy_plot_many_freerider("FedAvg",5)
plot_metric_for_all(FA_iid_5,FA_iid_45,FA_shard_5,FA_shard_45,FA_shak_5,
    FA_shak_45,y_label="Accuracy",save_name=f"accuracy_5_FedAvg",hist=False)

#FedProx 5 epochs
FP_iid_5,FP_iid_45,FP_shard_5,FP_shard_45,FP_shak_5,FP_shak_45=params_accuracy_plot_many_freerider("FedProx",5)
plot_metric_for_all(FP_iid_5,FP_iid_45,FP_shard_5,FP_shard_45,FP_shak_5,
    FP_shak_45,y_label="Accuracy",save_name=f"accuracy_5_FedProx",hist=False)


# In[2]: 
"""KS and L2 tests"""
from python_code.functions_plot_fig_1_2_KS_L2 import plot_KS_L2
from python_code.functions_plot_fig_1_2_KS_L2 import KS_L2_many_freerider

#FedAvg L2 20 epochs
FA_iid_L2_5,FA_iid_L2_45,FA_shard_L2_5,FA_shard_L2_45,FA_shak_L2_5,FA_shak_L2_45=KS_L2_many_freerider("L2",20,"FedAvg")
plot_KS_L2(FA_iid_L2_5,FA_shard_L2_5,FA_shak_L2_5,FA_iid_L2_45,FA_shard_L2_45,FA_shak_L2_45, 
    y_label="L2 - log scale",save_name="fig_2_L2_FA_20",legend_idx=5)

#FedAvg KS 20 epochs
FA_iid_KS_5,FA_iid_KS_45,FA_shard_KS_5,FA_shard_KS_45,FA_shak_KS_5,FA_shak_KS_45=KS_L2_many_freerider("KS",20,"FedAvg")
plot_KS_L2(FA_iid_KS_5,FA_shard_KS_5,FA_shak_KS_5,FA_iid_KS_45,FA_shard_KS_45,FA_shak_KS_45, 
    y_label="KS - log scale",save_name="fig_2_KS_FA_20",legend_idx=4)

#FedProx L2 20 epochs
FA_iid_L2_5,FA_iid_L2_45,FA_shard_L2_5,FA_shard_L2_45,FA_shak_L2_5,FA_shak_L2_45=KS_L2_many_freerider("L2",20,"FedProx")
plot_KS_L2(FA_iid_L2_5,FA_shard_L2_5,FA_shak_L2_5,FA_iid_L2_45,FA_shard_L2_45,FA_shak_L2_45, 
    y_label="L2 - log scale",save_name="fig_2_L2_FP_20",legend_idx=5)

#FedProx KS 20 epochs
FA_iid_KS_5,FA_iid_KS_45,FA_shard_KS_5,FA_shard_KS_45,FA_shak_KS_5,FA_shak_KS_45=KS_L2_many_freerider("KS",20,"FedProx")
plot_KS_L2(FA_iid_KS_5,FA_shard_KS_5,FA_shak_KS_5,FA_iid_KS_45,FA_shard_KS_45,FA_shak_KS_45, 
    y_label="KS - log scale",save_name="fig_2_KS_FP_20",legend_idx=4)

#FedAvg L2 5 epochs
FA_iid_L2_5,FA_iid_L2_45,FA_shard_L2_5,FA_shard_L2_45,FA_shak_L2_5,FA_shak_L2_45=KS_L2_many_freerider("L2",5,"FedAvg")
plot_KS_L2(FA_iid_L2_5,FA_shard_L2_5,FA_shak_L2_5,FA_iid_L2_45,FA_shard_L2_45,FA_shak_L2_45, 
    y_label="L2 - log scale",save_name="fig_2_L2_FA_5",legend_idx=4)

#FedAvg Ks 5 epochs
FA_iid_KS_5,FA_iid_KS_45,FA_shard_KS_5,FA_shard_KS_45,FA_shak_KS_5,FA_shak_KS_45=KS_L2_many_freerider("KS",5,"FedAvg")
plot_KS_L2(FA_iid_KS_5,FA_shard_KS_5,FA_shak_KS_5,FA_iid_KS_45,FA_shard_KS_45,FA_shak_KS_45, 
    y_label="KS - log scale",save_name="fig_2_KS_FA_5",legend_idx=4)
 
#FedProx L2 5 epochs
FA_iid_L2_5,FA_iid_L2_45,FA_shard_L2_5,FA_shard_L2_45,FA_shak_L2_5,FA_shak_L2_45=KS_L2_many_freerider("L2",5,"FedProx")
plot_KS_L2(FA_iid_L2_5,FA_shard_L2_5,FA_shak_L2_5,FA_iid_L2_45,FA_shard_L2_45,FA_shak_L2_45, 
    y_label="L2 - log scale",save_name="fig_2_L2_FP_5",legend_idx=4)

#FedProx KS 5 epochs
FA_iid_KS_5,FA_iid_KS_45,FA_shard_KS_5,FA_shard_KS_45,FA_shak_KS_5,FA_shak_KS_45=KS_L2_many_freerider("KS",5,"FedProx")
plot_KS_L2(FA_iid_KS_5,FA_shard_KS_5,FA_shak_KS_5,FA_iid_KS_45,FA_shard_KS_45,FA_shak_KS_45, 
    y_label="KS - log scale",save_name="fig_2_KS_FP_5",legend_idx=4)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from plots_func.Fig_CIFAR10 import plot_fig_CIFAR10_alpha_effect_both

n_SGD = 100
p = 0.1
mu = 0.0
plot_fig_CIFAR10_alpha_effect_both(n_SGD, p, mu)


from plots_func.Fig_CIFAR10_N_and_m import plot_fig_CIFAR10_N_and_m_both
from plots_func.Fig_CIFAR10_N_and_m import plot_fig_CIFAR10_N_and_m_one

alpha = 0.01

l_lr = [0.1, 0.05, 0.05, 0.05]
l_p = [0.1, 0.1, 0.05, 0.2]

plot_fig_CIFAR10_N_and_m_both(alpha, l_lr, l_p)

metric = "loss"
smooth = True
plot_fig_CIFAR10_N_and_m_one(metric, alpha, l_lr, l_p, smooth)


from plots_func.Fig_similarity_effect import plot_fig_similarity

alpha = 0.01
lr = 0.01
n_SGD = 100
p = 0.1
plot_fig_similarity(alpha, n_SGD, p, lr)


from plots_func.Fig_regularization import plot_regularization

alpha = 0.01
lr = 0.05
n_SGD = 100
p = 0.1
mu = 0.1
plot_regularization(alpha, n_SGD, p, mu, lr)


from plots_func.distribution_CIFAR import distribution_samples

list_alpha = [10, 0.1, 0.01, 0.001]
n_classes = 10
n_clients = 100
distribution_samples(list_alpha, n_classes, n_clients)

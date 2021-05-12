#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from plots_func.Fig_MNIST_shard import plot_fig_alg2

dataset = "MNIST_shard"
sampling = "clustered_1"
n_SGD = 50
seed = 0
lr = 0.01
decay = 1.0
p = 0.1
mu = 0.0
n_iter_plot = 200

plot_fig_alg2(dataset, sampling, n_SGD, seed, lr, decay, p, mu, n_iter_plot)


from plots_func.Fig_CIFAR10 import plot_fig_CIFAR10_alpha_effect_one

metric = "loss"
smooth = True
plot_fig_CIFAR10_alpha_effect_one(metric, n_SGD, p, mu, smooth)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print(
    "algo - dataset - epochs - noise type - multiplicator - power - n_freeriders - redo"
)


"""FEDAVG AND FEDPROX WIHTOUT THE HISTORY"""
l_algo = ["FedAvg", "FedProx"]
l_dataset = ["shakespeare", "MNIST-iid", "MNIST-shard", "CIFAR-10"]
l_epochs = [5, 20]
l_noise_type = [
    "FL 1 1",
    "plain 1 1",
    "disguised 1 1",
    "disguised 1 0.5",
    "disguised 1 2",
    "disguised 3 1",
    "disguised 3 0.5",
    "disguised 3 2",
]
l_fr = [1, 5, 45]
l_force = ["False"]


text_file = open("experiments.txt", "w")

from itertools import product

for algo, dataset, epochs, noise_type, n_fr, force in product(
    l_algo, l_dataset, l_epochs, l_noise_type, l_fr, l_force
):

    string = f"{algo} {dataset} {epochs} {noise_type} {n_fr} {force}\n"

    text_file.write(string)


"""HISTORY FOR FEDAVG"""
l_algo = ["FedAvg"]
l_dataset = ["shakespeare", "MNIST-iid", "CIFAR-10", "MNIST-shard"]
l_epochs = [5, 20]
l_noise_type = [f"many{i} 1 1" for i in range(30)]
l_fr = [45]
l_force = [False]


from itertools import product

for algo, dataset, epochs, noise_type, n_fr, force in product(
    l_algo, l_dataset, l_epochs, l_noise_type, l_fr, l_force
):

    string = f"{algo} {dataset} {epochs} {noise_type} {n_fr} {force}\n"

    text_file.write(string)


"""HISTORY FOR FEDPROX"""
l_algo = ["FedProx"]
l_dataset = ["shakespeare", "MNIST-iid", "MNIST-shard"]
l_epochs = [5, 20]
l_noise_type = [f"many{i} 1 1" for i in range(30)]
l_fr = [45]
l_force = [False]


from itertools import product

for algo, dataset, epochs, noise_type, n_fr, force in product(
    l_algo, l_dataset, l_epochs, l_noise_type, l_fr, l_force
):

    string = f"{algo} {dataset} {epochs} {noise_type} {n_fr} {force}\n"

    text_file.write(string)


text_file.close()

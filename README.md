# Clustered Sampling: Low-Variance and Improved Representativity for Clients Selection in Federated Learning

This paper can be found [here](http://proceedings.mlr.press/v139/fraboni21a.html).

## Download the dependencies

This work is done using PyTorch 1.4.0 .


## Datasets

Datasets are creaeted from CIFAR10 and MNIST. These datasets are not included in this repository but are automatically downloaded using `torchvision` library when running a simulation. A wide range of partitioning is available to create balanced and unbalanced datasets with different level of heterogeneity. More details in the paper We note that we always create datasets composed of 100 clients. For CIFAR, we partition the dataset using a Dirichlet distribution with $\alpha$ =\{0.001, 0.01, 0.1, 10\}. The code creating these datasets can be found in `./py_func/read_db.py`. 


## Initial Model

We detail in the paper the different models used. To have a fair comparison between different simulations, we allow the user to input the seed used to initialize the model. In all of our experiments, we use `seed=0`.


## Required Experiments

We run a wide range of experiments in this work. `experiments.txt`, `experiments_appendix.txt`, and `experiemnts_all.txt` are three txt files containing the experiments used in this work. `experiments.txt` contains all the experiments needed to obtain identical figures as the 2 given in the paper; `experiments_appendix.txt` contains all the experiments needed to obtain identical figures as the one given in the appendix; `experiments_all.txt` contains all the simulations run to produce this work. We test learning rates ranging from lr=\{0.001, 0.005, 0.01, 0.05, 0.1\}, number of SGD N=\{10, 100, 500\}, number of sampled clients m=\{5, 10, 20\}, different similarity measures for clustered sampling of Algorithm 2 (Arccos, L2, L1), different regularization parameters. More details in the paper.


## Running an experiment

Running an experiment requires to use `FL.py`. This code takes as input:
- The `dataset` used.
- The `sampling` scheme used. Either `random` for MD sampling, `clustered_1` and `clustered_2` for clustered sampling with Algorithm 1 and 2, or `FedAvg` for the initial sampling scheme proposed in FedAvg.
- The similarity measure `sim_type` used for the clients representative gradients. With `clustered_2` put either `cosine`, `L2` or `L1` and, with other sampling, put `any`. 
- The `seed` used to initialize the training model. We use `0` in all our experiments. 
- The number of SGD run locally `n_SGD` used. 
- The learning rate `lr` used. 
- The learning rate `decay` used after each SGD. We consider no decay in our experiments, `decay=1`.
- The percentage of clients sampled `p`. We consider 100 clients in all our datasets and use thus `p` = \{0.05, 0.1, 0.2\}.
- `force` a boolean equal to True when a simulation has already been run but needs to be rerun.
- The local loss function regularization parameter `mu`. Leaving this field empty gives no local regularization, `mu=0`.

Every experiment saves by default the training loss, the testing accuracy, and the sampled clients at every iteration in the folder `saved_exp_info`. The global model and local models histories can also be saved.


## Plotting the figures
`plots_paper.py` plots the figures in the paper and `plots_appendix.py` plots the ones in the appendix. We note that these codes work only if the FL experiments needed for them have already been run.








## A General Theory for Federated Optimization with Asynchronous and Heterogeneous Clients Updates

This paper can be found [here](https://arxiv.org/abs/2206.10189).

### Download the Dependencies

This work is done using Python 3.9, PyTorch 1.11.0, and Torchvision 0.12.0.


### `FL.py` and its Inputs

`FL.py` is the file used to launch any asynchronous experiment with specificities resulting from its input parameters described hereafter:
- `dataset_name`: the dataset name considered composed of up to two elements separated by underscore "_": the public dataset considered (`MNIST`, `CIFAR10`, `CIFAR100`, `Shakespeare`) and the dirichlet distribution parameter as a float. 
- `opt_scheme`: the optimization scheme(`FL`, `Async`, `FedFix-X`, `FedBuff-X`) followed by the type of aggregation weight policy (`weight` or `identical`). The `X` stands in `FedFix-X` for the amount of time the server waits before creating the new global model, and in `FedBuff-X` for the amount of clients to wait before creating the new global model.
- `time_scenario`, more details in our [paper](https://arxiv.org/abs/2206.10189). The string `F-X` where the fastest client is `X`% faster than the slowest.
- `P_type`: the clients importance considered `unfiorm` in all our experiments.
- `T`: the amount of time for given to the experiment.
- `n_SGD`: the amount of local SGD run by each client.
- `B`: the amount of data samples considered for each SGD.
- `lr_g`: the server aggregation learning rate set to 1 in our experiments.
- `lr_l`: the local learning rate used by clients to perform their local work.
- `M`: the amount of participating clients.
- `seed`: the seed used to initialize the global model.

MNIST, CIFAR10, and CIFAR100 are downloaded with torchvision. Shakespeare is downloaded with [Leaf](https://github.com/TalwalkarLab/leaf), which we add as a submodule of our repository in the `data` folder.

`list_exp_to_run.py` generates the `.txt` for which each row gives the information of one experiment.

All the experiments needed to run all the experiments used in this work run to obtain our paper can be found in the `.txt` files generated with `list_experiments.py`.



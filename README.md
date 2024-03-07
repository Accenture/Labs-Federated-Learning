## Sequential Informed Federated Unlearning: Efficient and Provable Client Unlearning in Federated Optimization

This paper can be found [here](https://arxiv.org/abs/2211.11656).

### Download the Dependencies

This work is done using Python 3.9 and PyTorch 1.11.0.


### `FU.py` and its Inputs

`FU.py` is the function used for any FU simulation with specificities resulting from its input parameters described hereafter:
- `dataset_name`: the dataset name considered composed of up to three elements separated by underscore "_":
  the public dataset considered (`MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, `celeba`), 
  the dirichlet distribution parameter as a float,
  and the string `backdoored` if each client's data is watermarked,
  We note that no dirichlet distribution follows the dataset CelebA as its partitioning is independent of this work.
  For example, `CIFAR10_0.1_backdoored` clients have 100 watermarked data points sampled from CIFAR10 
  with class ratio randomly obtained with a Dirichlet distribution of parameter 0.1. Other examples, `celeba`, `MNIST_0._backdoored`, `CIFAR100_1.`
- `unlearn_scheme`: the 7 unlearning schemes considered `SIFU`, `scratch`, `fine-tuning`, `DP_X`, `last`, `FedEraser` and `FedAccum`.
    We note that the training with every client is identical for `SIFU`, `scratch`, `last`, and `fine-tuning`, which we thus obtain with `train`.
    Also, the `X` in `DP_X` is the clipping constant of a client's contribution, e.g. `DP_0.2`.
- `model`: the model architecture considered, `default` for MNIST and `CNN` for every other dataset,
- `forgetting`: the string giving the associated unlearning request saved in `policy.py` and set to `P70` for Figure 2, and `P9` for Figure 3,
- `T`: the maximal allowed ammount of aggregation rounds, set to 10 000 in our experiments,
- `B`: the amount of data samples considered for each SGD (batch size),
- `n_SGD`: the amount of local SGD ran by each client between each aggregation round,
- `n_SGD_cali`: only relevant for `FedEraser`, the number of local SGD steps used for retraining, defaults to half of n_SGD
- `delta_t`: only relevant for `FedEraser`, the amount of time between two calibration steps,
- `lr_g`: the server aggregation learning rate set to 1 in our experiments,
- `lr_l`: the local learning rate used by clients to perform their local work,
- `M`: the amount of participating clients set to 100 in our experiments,
- `n_sampled`: the amount of sampled clients at each epoch,
- `clip`: the gradient clipping used in training,
- `epsilon`: the unlearning budget for `SIFU`, `last`, and `DP_X`,
- `sigma`: the noise perturbation parameter for `SIFU`,
- `stop_acc`: the stopping accuracy of the training or retraining for the global model,
- `seed`: the seed used to initialize the global model,
- `iter_min`: the minimal amount of iterations before stopping the training or retraining for the global model,
- `compute_diff`: used to compute additional informations about gradient norm, required for `FedEraser` and `FedAccum`,
- `device`: the device used for the experiment.


### Running the Experiments
Running the experiments is done by calling the function `FU.py` with the desired input parameters.
Before unlearning, it is necessary to first train the initial model by using the argument `train` for `unlearn_scheme`.
For instance, to run the unlearning with SIFU on CIFAR10 presented in Figure 2, one can launch the experiments with the script SIFU_example.sh.
Note that the script uses parallelization that might need to be removed if not supported by the system.
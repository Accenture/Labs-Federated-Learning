## Sequential Informed Federated Unlearning: Efficient and Provable Client Unlearning in Federated Optimization

This paper can be found [here](future_url).

### Download the Dependencies

This work is done using Python 3.9 and PyTorch 1.11.0.


### `FU.py` and its Inputs

`FU.py` is the function used for any FU simulation with specificities resulting from its input parameters described hereafter:
- `dataset_name`: the dataset name considered composed of up to three elements separated by underscore "_":
  the public dataset considered (`MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, `celeba`), 
  the dirichlet distribution parameter as a float,
  and the string `backdoored` if each client's data is watermarked.
  We note that no dirichlet distribution follows the dataset CelebA as its partitioning is independent of this work. 
  For example, `CIFAR10_0.1_backdoored` clients have 100 watermarked data points sampled from CIFAR10 
  with class ratio randomly obtained with a Dirichlet distribution of parameter 0.1. Other examples, `celeba`, `MNIST_0._backdoored`, `CIFAR100_1.`
- `unlearn_scheme`: the 6 unlearning schemes considered `SIFU`, `scratch`, `fine-tuning`, `DP_X`, `last`, and `FedAccum`.
    We note that the training with every client is identical for `SIFU`, `scratch`, `last`, and `fine-tuning`, which we thus obtain with `train`.
    Also, the `X` in `DP_X` is the clipping constant of a client's contribution, e.g. `DP_0.2`.
- `forgetting`: the string giving the associated unlearning request saved in `policy.py` and set to `P9` for Figure 2,  
- `n_SGD`: the amount of local SGD run by each client,
- `B`: the amount of data samples considered for each SGD,
- `lr_global`: the server aggregation learning rate set to 1 in our experiments,
- `lr_local`: the local learning rate used by clients to perform their local work,
- `M`: the amount of participating clients set to 100 in our experiments,
- `n_sampled`: the amount of sampled clients,
- `epsilon`: the unlearning budget for `SIFU`, `last`, and `DP_X`,
- `sigma`: the noise perturbation parameter for `SIFU`,
- `stop_acc`: the stopping accuracy of the training or retraining for the global model, and
- `seed`: the seed used to initialize the global model.
# Free-rider Attacks on Model Aggregation in Federated Learning

This paper can be found [here](http://proceedings.mlr.press/v130/fraboni21a.html).

## Download the dependencies
This work is done using PyTorch 1.4.0 .


## Datasets
The datasets used for MNIST iid, MNIST non-iid, CIFAR-10 and Shakespeare can be downloaded using the python code in the `data` folder.
To better distinguish MNIST iid from MNIST non-iid, these two datasets are respectively called `MNIST-iid` and `MNIST-shard`.
More details in the paper about how the datasets are built.


## Initial Models
In `variables` are saved the initial models used for the comparisons between free-riding and federated learning for each dataset:
`model_MNIST_0.pth` for MNIST-iid and MNIST non-iid, `model_CIFAR-10_0.pth` for CIFAR-10, and `model_shakespeare_0.pth` for Shakespeare.


## Simple plot
Since the computation of Figures 1 and 2 requires many experiemnts to be performed, we propose `simple_experiment.sh` computing a plot for FedAvg, MNIST-iid, 5 epochs and 300 iterations. This code runs three experiments: FedAvg with fair clients, one plain freerider, and one disguised freerider for $\sigma$ and $\gamma=1$. 
The plot is saved in `plots` as `simple_experiment.png`.


## Figure 1 and 2, and the associated plots in Appendix

### Running the experiments
`free-riding.py` is the code needed to run the experiments needed for the paper's plots with one or many free-riders (5 and 45 in the paper). The code take as input eight arguments in the following order:
- `algo`: the federated learning algorithm used. Either the string "FedAvg" or "FedProx".
- `dataset`: the federated learning dataset used for the experiment. Either the string “MNIST-iid”, “MNIST-shard”, "CIFAR-10" or “shakespeare”.
- `epochs`: the number of epochs run by the fair clients at each iteration. Only 5 and 20 are used in the paper.
- `experiment type`. The string “FL” to run federated learning with no attackers and the dataset saved initial model, “plain” to run plain free-riding, “disguised” to run disguised free-riding, or “manyXX” to run federated learning with a random initialization where XX is the experiment number. In our paper, XX ranges from 0 to 29.
- `coef`: the scalar multiplying the standard deviation obtained with the paper's heuristic. 1 and 3 used in the paper.
- `power`: the $\gamma$ in $\varphi(t)$. 0.5, 1 or 2 used for the paper experiments.
- `n_freeriders`. Number of free-riders used for the experiment. In the paper we use 1, 5 or 45. 
- `redo`: Boolean needed to rerun an experiment.

`coef` and `power` are only taken into account if `experiment_type` has "disguised" as an input. These fields still need to be filled for any other `experiment_type` but will not be taken into account.

Each experiment saves the loss and the accuracy history in respectively `hist/acc` and `hist/loss`, the model at the end of the training in `saved_models/final`, and the history of the global model at every iteration in `saved_models/hist`.

### Experiments run for the paper

In `txt_to_run_experiments`, the code `generate_txt_files_experiments.py` creates, once run, `experiments.txt` including at each row the info of an experiment to parse as an input to `free-riding.py`.

### Plotting the figures
Figure 1, Figure 2 and their annexes' derivatives are obtained with `get_figs.py`. These code will run without error only if all the simulations with parameters in `experiments.txt` are run.

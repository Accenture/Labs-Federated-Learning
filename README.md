# Free-rider Attacks on Model Aggregation in Federated Learning

## Download the dependencies
This work is done using PyTorch 1.4.0 .

## Datasets
The datasets used for MNIST iid, MNIST non-iid, and Shakespeare are in the `data` folder.  To better distinguish MNIST iid from MNIST non-iid, these two datasets are respectively called `MNIST-iid` and `MNIST-shard` in all this repository.

## Initial Models
The initial models used for the comparison between federated learning and free-riding can be found in `variables` as `model_MNIST_0.pth` for MNIST-iid and MNIST non-iid and as `model_shakespeare_0.pth` for Shakespeare.

## Simple plot
As a lot of simulations are needed to get Figure 1 and Figure 2, we propose `simple_plot.py` computing a plot for FedAvg, MNIST-iid, E=5 and 300 iterations. This code run three experiments: FedAvg with fair clients, with one plain freerider and with one disguised freerider for $\sigma$ and $\gamma=1$. The plot is saved in `plots` as `simple_experiements.png`.

## Figure 1 and 2, and the associated plots in Appendix

### Running the experiments
`one_freerider.py` and `many_freeriders.py`are the two code files needed to run all the experiments needed for these plots. They are respectively for running experiments with only one free-rider and with many free-riders (5 and 45 in the paper). They both take as input nine arguments in the following order:
- `algo`: the federated learning algorithm used. Either the string "FedAvg" or "FedProx".
- `dataset`: the federated learning dataset used for the experiment. Either the string “MNIST-iid”, “MNIST-shard” or “shakespeare”.
- `epochs`: the number of epochs run by the fair clients at each iteration. Either 5 or 20.
-  `experiment type`. Either the string “FL”, to run the federated learning algorithm for the saved initialization, “none” to run plain free-riding, “add” to run disguised free-riding, or “manyXX” to run federated learning with a random initialization where XX is the experiment number. In our paper, XX ranges from 0 to 29.
- `noise_shape`. Either the string `linear` or `exp` for exponential. In the paper we are only using `linear`.
- `coef`: the coefficient multiplying the standard deviation obtained with the heuristic in the paper. Either 1 or 3.
- `power`: the $\gamma$ in $\varphi(t)$. In the paper experiments either 0.5,1 or 2.
- `n_freeriders`. The number of free-riders used for the experiment. In the paper we use 1, 5 or 45. `one_freerider.py` only works with 1 and `many_freerider.py` only works with 5 and 45. 
- `redo`: a Boolean equal to True to rerun the experiments with the same values for all the fields above.

Each experiment saves the loss and accuracy history in the folder `variables`, the global model at each iteration in `saved_models` and the final server model in `models_s`.
In the folder `txt_to_run_experiments` can be found the .txt files listing all the experiments settings to run for each python file. Each row in the txt file is for one experiment.

### Plotting the figures
To plot Figure 1 and Figure 2, the code to use are respectively `get_fig1.py` and `get_fig2.py`. To plot Figure 1 and all the associated plot, all the simulations with the settings in the .txt file `single_freerider.txt` and `single_freerider_for_confidence_interval.txt` need to be run. Similarly for Figure 2,  all the simulations with the settings in the .txt file `multiple_freerider.txt` and `multiple_freerider_for_confidence_interval.txt` need to be run.

Before running `get_fig1.py` or `get_fig2.py`, it is important to run `create_history_for_KS_L2_plot.py`. This code creates the metric history for the KS and L2 test at each iteration by using all the global model saved in `saved_models`. Other metric can be computed with this history.


## Figure 3 and 4
### Running the experiments
`improve_disguise.py` runs the experiments used for Figure 3 and 4. This code takes the following three arguments:
- `n_gaussians`: the number of Gaussian used in the Gaussian mixture model,
- `cycle`: the number of iterations before the free-rider computes a new standard deviation with the heuristic,
- `outliers`: Boolean equal to True if the free-rider is generating some outliers in its parameters distribution.

In the folder `txt_to_run_experiments` can be found `improve_disguised.txt`, the file listing the four experiments settings used to create Figure 4. Each experiment saves in the folder `fig4` the loss history, accuracy history, global model history and local model history of each client 

### Plotting the Figures
`get_fig_3_4.py` plots the Figure 3 and 4 using the files obtained with the four experiments run with `improve_disguise.py`.

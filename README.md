# On the Impact of Client Sampling on Federated Learning Convergence

This paper can be found here.

## Download the Dependencies

This work is done using PyTorch 1.7.1.


## Synthetic Experiments

Figure 1 can directly be obtained by running `python plot_figures.py --plot quadratic`.  Figure 3 and 4 in Appendix E.1 are automatically generated alongside Figure 1.

## Shakespeare Experiments

### Datasets

[Leaf](https://github.com/TalwalkarLab/leaf) is a submodule of our repository in the `data` folder. Thanks to this repository, we download a portion of the federated dataset Shakespeare with the command 
`./preprocess.sh - s niid - -sf 0.2 - k 0 - t sample - -tf 0.8` 
from which we create 4 sub federated datasets with 80, 40, 20, 10 clients respectively called `Shakespeare`, `Shakespeare2`, `Shakespeare3`, and `Shakespeare4`. The download and creations of these datasets is automatically done when running a simulation with `FL.py`.

More information about the Shakespeare dataset can be found [here](https://arxiv.org/abs/1812.01097).


### `FL.py` and its Inputs

`FL.py` is the function used for any FL simulation. An FL simulation specificity will come from `Fl.py` input parameters. We list all the parameters that can be changed with their associated default value and description:
- `dataset` = "Shakespeare", the dataset name considered,
- `sampling` = "MD", the client sampling scheme chosen to select clients during the learning process (with "Full" every client is considered, with "Uniform" clients are uniformly sampled without replacement, with "MD" clients are selected with a Multinomial Distribution),
- `n_SGD` = 10, the amount of local SGD run by each client,
- `lr_local` = 1.5, the local learning rate used by clients to perform their local work,
- `lr_global` = 1.0, the aggregation learning rate,
- `n_sampled` = 10, the amount of sampled clients,
- `batch_size` = 64, the amount of data samples considered for each SGD,
- `mu` = 0, the local loss function regularization term ([More information](https://arxiv.org/abs/1812.06127))
- `n_iter` = 100, the amount of server optimization step, 
- `seed`, the seed used to initialize the global model,
- `device`, a boolean indicating whether to use GPU or CPU,
- `decay`, the local learning rate decay applied at every FL optimization step, and 
- `importance_type` = "ratio", the importance given to a client in the global loss function ("ratio" gives clients an importance proportionnal to their data amount while "uniform" gives clients identical importance).



### Needed Experiments in This Work

Running `list_experiments.py` generates `Shak_main.txt` with all the parameters that need to be changed from the default ones in `FL.py` to obtain Figure 2. Each line of the generated `.txt` file contains the values of the changed parameters which we list here: `dataset`, `sampling`, `n_SGD`, `lr_local`, `lr_global`, `n_sampled`, `batch_size`, `n_iter`, `mu`, `importance_type`, `decay`, and `seed`.

When an experiment is finished running, the line associated to this simulation is removed from `Shak_main.txt` when regenerating this file with `list_experiments.py`.



### Running experiments

Once `Shak_main.txt` is empty. Running `python plot_figures.py --plot shak_paper` provides Figure 2 and Figure 5 in Appendix E.2.


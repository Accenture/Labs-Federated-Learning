#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Experiment_Names:
    def __init__(self, args: dict, verbose: bool = True):
        self.__dict__.update(args)

        # If training for the first time, force all the clients to be considered
        if self.unlearn_scheme == "train":
            self.forgetting = "P0"

        self.file_name, self.file_name_train = self.name()
        if verbose:
            print("file_name:", self.file_name)
            print("file_name_train:", self.file_name_train)
        self.exp_string = self.string_training()

    def name(self) -> str:
        """file_name of the experiment based on input parameters"""

        file_name = "FL_"

        params = [
            self.dataset_name,
            self.unlearn_scheme,
            # "train",
            self.opti,
            self.forgetting,
            self.P_type,
            self.T,
            self.n_SGD,
            self.B,
            self.lr_g,
            self.lr_l,
            self.lambd,
            self.dropout,
        ]
        params = [str(p) for p in params]
        prefix = ["", "", "", "", "", "T", "K", "B", "lr_g", "lr_l",
                  # "M", "m", "p",
                  "lam", "dr"]
        file_name += "_".join([p + q for p, q in zip(prefix, params)])
        file_name += f"_s{self.seed}"

        file_name_train = file_name

        if self.unlearn_scheme != "train":
            file_name += "_unlearn"

        if self.unlearn_scheme in ["finetuning", "last", "scratch"]:
            file_name_train = file_name_train.replace(self.forgetting, "P0")
            file_name_train = file_name_train.replace(self.unlearn_scheme, "train")
            file_name_train += "_last"
        elif self.unlearn_scheme.split("-")[0] == "PGD":
            file_name_train = file_name_train.replace(self.unlearn_scheme, "FedAccum")

        return file_name, file_name_train

    def string_training(self) -> str:
        """create the string including all the arguments for the experiment"""
        string = (
            f"--dataset_name {self.dataset_name} --exp_name {self.exp_name} "
            f"--unlearn_scheme {self.unlearn_scheme} --opti {self.opti} "
            f"--P_type {self.P_type} --T {self.T} --n_SGD {self.n_SGD} "
            f"--B {self.B} --lr_g {self.lr_g} --lr_l {self.lr_l} "
            f"--seed {self.seed} --forgetting {self.forgetting} "
            f"--lambd {self.lambd} "
            f"--dropout {self.dropout} "
        )

        if self.save_unlearning_models:
            string += "--save_unlearning_models "

        string += "\n"
        return string



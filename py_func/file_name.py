#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_file_name(
    dataset: str,
    sampling: str,
    n_iter: int,
    n_SGD: int,
    batch_size: int,
    lr_global: float,
    lr_local: float,
    n_sampled: float,
    mu: float,
    importances: bool,
    decay: float,
    seed: int,
):
    """
    file name used to save the different experiments results
    """

    file_name = (
        f"{dataset}_{sampling}_{importances}_T{n_iter}_E{n_SGD}"
        f"_B{batch_size}_lr_g{lr_global}_lr_l{lr_local}"
    )

    if sampling != "Full":
        file_name += f"_m{n_sampled}_s{seed}"

    if mu > 0.0:
        file_name += f"_mu{mu}"

    if decay < 1.0:
        file_name += f"_d{decay}"

    return file_name

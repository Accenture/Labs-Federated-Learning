import os
import random
import shutil

from numpy.random import shuffle
import numpy as np


def create_a_split(seed:int):

    path_train_s = path_train + f"_s{seed}"
    path_test_s = path_test + f"_s{seed}"
    if not os.path.exists(f"data/{path_train_s}"):

        os.mkdir(path_train_s)
        os.mkdir(path_test_s)

        list_clients = os.listdir(path_origin)
        print(list_clients)

        for i, client in enumerate(list_clients):
            os.mkdir(f"{path_train_s}/{client}")
            os.mkdir(f"{path_test_s}/{client}")
            list_datapoints = os.listdir(f"{path_origin}/{client}")
            list_datapoints = [dp for dp in list_datapoints if dp[-4:] != ".csv"]

            np.random.seed(42 + seed)
            shuffle(list_datapoints)

            train_data = list_datapoints[:int(0.8 * len(list_datapoints))]
            test_data = list_datapoints[int(0.8 * len(list_datapoints)):]

            for data in train_data:
                shutil.copytree(f"{path_origin}/{client}/{data}", f"{path_train_s}/{client}/{data}")
            for data in test_data:
                shutil.copytree(f"{path_origin}/{client}/{data}", f"{path_test_s}/{client}/{data}")


path_origin = "data/prostate"
path_train = "data/prostate_train"
path_test = "data/prostate_test"

for s in range(5):

    create_a_split(s)

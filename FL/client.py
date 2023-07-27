import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module, functional

from copy import deepcopy
import math
import random

from FL.read_db import get_dataloaders
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Clients:
    def __init__(self, dataset: str, M: int, B: int):
        self.dls_train, self.dls_test = get_dataloaders(dataset, B, M)
        self.M = len(self.dls_train)
        self.B = B

    def __len__(self):
        return self.M

    def client_dataset(self, i: float, ds_type: str):
        if ds_type == "train":
            return self.dls_train[i]
        elif ds_type == "test":
            return self.dls_test[i]

    def n_samples(self):
        """returns a list of the amount of each client's amount of data"""
        return [len(self.client_dataset(i, "train").dataset) for i in range(self.M)]

    def pred_i_batch(self, i: int, model: Module,
                     loss_f: functional, ds_type: str) -> torch.tensor:
        """returns loss value for random batch of client i evaluated on model + loss_f"""

        iterations = next(iter(self.client_dataset(i, ds_type)))
        features, labels = iterations["image"], iterations["label"]
        predictions = model(features.to(device))

        return loss_f(predictions, labels.to(device))


    def local_work(
        self,
        working_clients: list[str],
        model: Module,
        loss_f: functional,
        opti_type: torch.optim,
        optimizer_params: dict,
        n_SGD: int,
    ) -> (list, list, list):
        local_models, loss_trains, loss_tests = [], [], []

        for i in range(len(self)):

            # CREATE A CLIENT LOCAL MODEL AND PREPARE IT FOR TRAINING
            local_model = deepcopy(model)
            local_model.train().to(device)
            local_models.append(local_model)

            # COMPUTE LOSS OF CLIENT i ON A RANDOM BATCH OF ITS TESTING DATA
            loss_tests.append(
                self.pred_i_batch(i, local_model, loss_f, "test").data.item()
            )

            # COMPUTE LOSS OF CLIENT i ON A RANDOM BATCH OF ITS TRAINING DATA
            # ALSO COMPUTE THE LOCAL WORK OF working_clients
            if i in working_clients:
                optimizer = opti_type(local_model.parameters(), **optimizer_params)

                for k in range(n_SGD):
                    optimizer.zero_grad()

                    batch_loss = self.pred_i_batch(i, local_model, loss_f, "train")
                    batch_loss.backward()

                    if k == 0:
                        loss_trains.append(batch_loss.data.item())

                    optimizer.step()

            else:
                loss_trains.append(
                    self.pred_i_batch(i, local_model, loss_f, "train").data.item()
                )

        return local_models, loss_trains, loss_tests

    def loss_clients(self, model: Module,
                     loss_f: torch.nn.functional, ds_type: str):

        model.eval().to(device)

        losses = [0] * len(self)

        for i in range(len(self)):

            for iteration in iter(self.client_dataset(i, ds_type)):
                features, labels = iteration["image"], iteration["label"]

                with torch.no_grad():
                    predictions = model(features.to(device))
                loss_batch = loss_f(predictions, labels.to(device)).data.item()
                losses[i] += loss_batch * len(features)

            losses[i] /= len(self.client_dataset(i, ds_type).dataset)

        return losses
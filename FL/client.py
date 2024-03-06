import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module, functional

from copy import deepcopy

from FL.read_db import get_dataloaders



class Client:
    def __init__(self, dl_train: DataLoader, dl_test: DataLoader,
                 dl_train_eval: DataLoader, dl_test_eval: DataLoader,
                 device: torch.device, compute_grad=False):
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.device = device
        self.dl_train_eval = dl_train_eval
        self.dl_test_eval = dl_test_eval

    def local_work(self, model: Module, loss_f: functional, lr_l: float,
                   n_SGD: int, lambd: float, clip: float, compute_grad: bool):
        # print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model.train().to(self.device)

        optimizer = optim.SGD(model.parameters(),
                              lr=lr_l,
                              momentum=0.9,
                              weight_decay=lambd)

        if compute_grad:
            # Copy the model parameters before doing the gradient descent steps
            old_param = torch.cat([p.flatten() for p in model.parameters()]).detach().clone()

        for _ in range(n_SGD):
            optimizer.zero_grad()
            features, labels = next(iter(self.dl_train))
            predictions = model(features.to(self.device))

            batch_loss = loss_f(predictions, labels.to(self.device))
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        if compute_grad:
            # Compute the distance between the old and new parameters
            new_param = torch.cat([p.flatten() for p in model.parameters()])
            return new_param - old_param
        


    def loss_acc(self, model: Module, loss_f: torch.nn.functional, ds_type: str) -> (float, float):
        """Compute loss and acc of `model` on `client_data`"""


        model.eval().to(self.device)
        loss, correct = 0, 0

        for features, labels in iter(self.dl_train_eval):

            with torch.no_grad():
                predictions = model(features.to(self.device))

            # print(predictions)
            # print(labels.to(self.device))
            # print(loss_batch)
            # print(loss_f)
            loss_batch = loss_f(predictions, labels.to(self.device)).item()

            loss += loss_batch * len(features)

            _, predicted = predictions.max(1, keepdim=True)
            correct_batch = torch.sum(predicted.view(-1) == labels.to(self.device)).item()
            correct += correct_batch

        loss /= len(self.dl_train_eval.dataset)
        accuracy = 100 * correct / len(self.dl_train_eval.dataset)
        
        return loss, accuracy


class Clients:
    def __init__(self, dataset: str, M: int, B: int, device="cpu"):
        dls_train, dls_test = get_dataloaders(dataset, B, M)
        dls_train_eval, dls_test_eval = get_dataloaders(dataset, 2 * 10**3, M)

        self.clients = [
            Client(dl_train_i, dl_test_i, dl_train_eval_i, dl_test_eval_i, device)
            for dl_train_i, dl_test_i, dl_train_eval_i, dl_test_eval_i
            in zip(dls_train, dls_test, dls_train_eval, dls_test_eval)
        ]

        self.dataset = dataset
        self.M = M
        self.B = B

    def __getitem__(self, i: int):
        return self.clients[i]

    def local_work(
        self,
        working_clients: list[str],
        model: Module,
        loss_f: functional,
        lr_l: float,
        n_SGD: int,
        lambd: float,
        clip: float,
        compute_grad: bool = False,
    ):

        local_models = []
        local_grads = []
        for idx, i in enumerate(working_clients):
            local_models.append(deepcopy(model))
            # print("\n", "\n", "local model device", next(local_models[0].parameters()).device, "\n", "\n", next(model.parameters()).device)
            grad = self.clients[i].local_work(local_models[idx], loss_f, lr_l, n_SGD,
                                       lambd, clip, compute_grad)
            if compute_grad:
                local_grads.append(grad)
        if compute_grad:   
            return local_models, local_grads
        else:
            return local_models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

from copy import deepcopy

from FL.create_model import load_model
from FL.branch import Branch
from FL.client import Clients

import random
from FL.privacy import psi, get_std
import pickle

from variables import policies, CONFIG_SIFU, CONFIG_PGD
import wandb

from FL.experiment import Experiment_Names

from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def noise_model(model: Module, sigma: float):
    """Aggregates the clients models to create the new model"""

    model.to(device)
    if sigma > 0:
        for w_k in model.parameters():
            noise = torch.normal(0.0, sigma, size=w_k.shape).to(device)
            w_k.data.add_(noise)


def vector_params_detached(model: Module) -> torch.Tensor:
    return nn.utils.parameters_to_vector(deepcopy(model).parameters()).detach()


class Server:

    def __init__(self, args: dict):

        self.dataset_name = args["dataset_name"]
        self.seed = args["seed"]
        self.unlearn_scheme = args["unlearn_scheme"]
        self.opti = args["opti"]
        self.T = args["T"]
        self.M = args["M"]
        self.lr_g = args["lr_g"]
        self.lr_l = args["lr_l"]
        self.P_type = args["P_type"]
        self.n_SGD = args["n_SGD"]
        self.lambd = args["lambd"]
        self.dropout = args["dropout"]
        self.seed = args["seed"]

        exp = Experiment_Names(args, verbose=True)
        # NAME OF THE EXPERIMENT ASSOCIATED TO THIS SEQUENCE OF SERVER AGGREG.
        self.file_name = exp.file_name
        # NAME OF THE EXPERIMENT USED TO OBTAIN THE INITIAL UNLEARNING MODEL
        self.file_name_train = exp.file_name_train

        self.policy = policies[args["forgetting"]]

        # INITIAL MODEL AND OPTIMIZATION ELEMENTS USED FOR THE TRAINING
        self.model_0, self.loss_f, self.optimizer, self.optimizer_params \
            = load_model(
                self.dataset_name, self.opti, self.lr_l,
                self.dropout, "default", self.seed
            )
        self.g_model = deepcopy(self.model_0)

        self.working_clients = [i for i in range(self.M)]
        self.P_kept = None

        # BEST MODELS FOR SIFU and SIFU improved
        self.best_SIFU = {
            key: [deepcopy(self.model_0) for _ in range(self.M)]
            for key in CONFIG_SIFU.keys()
        }
        self.best_SIFU_improved = {
            key: [deepcopy(self.model_0) for _ in range(self.M)]
            for key in CONFIG_SIFU.keys()
        }
        # THRESHOLDS TO SAVE THE GLOBAL MODELS
        self.psi_star = {
            key: psi(CONFIG_SIFU[key]["epsilon"], CONFIG_SIFU[key]["delta"],
                     CONFIG_SIFU[key]["sigma"])
            for key in CONFIG_SIFU.keys()
        }

        # MODEL WITH WEIGHT=0 FOR FedAccum
        self.saved_FedAccum = [deepcopy(self.model_0) for _ in range(self.M)]

        # INITIALISE THE HISTORY USED TO SAVE MEASURES ON THE DATASET
        empty = np.zeros((2, self.T + 2, self.M))
        # self.acc = deepcopy(empty)
        self.loss = np.zeros((2, self.T + 2, self.M))
        self.metric = np.zeros((self.T + 2, self.M))

        vec_g = vector_params_detached(self.model_0)
        self.metric_univariate = [
            deepcopy((vec_g - vec_g).detach()).to(device)
            for i in range(self.M)
        ]

        # 0 WEIGHT MODEL FOR FUKD
        zero_weight_model = deepcopy(self.model_0).to(device)
        for w_k in zero_weight_model.parameters():
            w_k.data.mul_(0)
        self.models_FUKD = [
            deepcopy(zero_weight_model) for _ in range(self.M)
        ]

        # Amount of aggregations since the start of the learning process
        self.t = 0

    def clients_importance(self, n_samples: list[int]):
        if self.P_type == "uniform":
            self.P_kept = np.array([1. / self.M] * self.M)
        elif self.P_type == "ratio":
            self.P_kept = np.array(n_samples) / sum(n_samples)

    def forget_Wr(self, clients: Clients):

        print(f"forgetting {self.policy}")

        if len(self.policy) == 0:
            print("no client to unlearn")
            return
        elif len(self.policy) >= 2:
            raise Exception("More than one client to unlearn. "
                            "Code not adapted for this application")

        # STOP CONSIDERING CLIENTS IN W_r AND SET THEIR IMPORTANCE TO 0
        for e in self.policy:
            self.working_clients.remove(e)
            self.P_kept[e] = 0.0
        self.P_kept /= sum(self.P_kept)

        print("forgetting with", self.unlearn_scheme)

        # GLOBAL MODEL TO START THE UNLEARNING WITH
        if self.unlearn_scheme.split("-")[0] in ["SIFU", "SIFUimp", "finetuning",
                                                 "FedAccum", "PGD", "FUKD", "last"]:
            # LOAD THE MODEL SAVED DURING 'train' FOR THIS UNLEARNING SCHEME
            print("Loading Global Model with file name", self.file_name_train)
            path_file = f"saved_exp_info/models/{self.file_name_train}.pth"
            self.g_model.load_state_dict(
                torch.load(path_file, map_location="cpu")
            )
            print("Done.")

        elif self.unlearn_scheme == "scratch":
            # UNLEARNING STARTS WITH THE INITIAL MODEL
            self.g_model = deepcopy(self.model_0)
        else:
            raise Exception("unlearning scheme not covered")

        # MODIFICATION TO THE UNLEARNING INITIAL MODEL
        if self.unlearn_scheme.split("-")[0] in ["SIFU", "SIFUimp"]:
            torch.manual_seed(self.seed)
            config = self.unlearn_scheme.split("-")[1]
            sigma = CONFIG_SIFU[config]["sigma"]
            noise_model(self.g_model, sigma)
        elif self.unlearn_scheme == "last":
            # GET THE METRIC ON THE LAST TRAINED MODEL OF THE CLIENT TO FORGET
            std = get_std(np.max(self.metric[self.r - 1]), self.epsilon, self.delta)
            print(std)
            noise_model(self.g_model, std, 42 + self.seed)
        elif self.unlearn_scheme.split("-")[0] == "PGD":
            # CREATE w_ref THE INITIAL GLOBAL MODEL AND REFERENCE FOR PROJECTION
            w_ref = deepcopy(self.g_model).to(device)

            # ESTIMATE DELTA (SEE PAPER FOR MORE DETAILS, HALIMI ET AL. 2022)
            l_delta = np.zeros(10)
            for s in range(10):
                random_model, _, _, _ = load_model(
                    self.dataset_name, self.opti, self.lr_l,
                    self.dropout, "default", 42 + s, verbose=False
                )
                l_delta[s] = torch.norm(
                    vector_params_detached(random_model)
                    - vector_params_detached(self.g_model)
                )
            delta = np.mean(l_delta) / 3

            # THE CLIENT PERFORMS ITS ASCENTS UNTIL DICE SCORE IS BELOW tau
            optimizer = self.optimizer(self.g_model.parameters(),
                                       **self.optimizer_params)

            # GET THE PARAMETERS FOR THE  UNLEARNING
            params = CONFIG_PGD[self.unlearn_scheme.split("-")[1]]
            n_SGD, tau = params["n_SGD"], params["tau"]
            self.g_model.to(device)

            # PERFORM UNLEARNING ON UNLEARNT DATA TO HAVE THE TRUE INITIAL MODEL
            for k in range(n_SGD):
                optimizer.zero_grad()

                # BATCH LOSS FOR THE GRADIENT ASCENT
                batch_loss = - clients.pred_i_batch(
                    self.policy[0], self.g_model, self.loss_f, "train"
                )
                print(k, batch_loss)

                # COMPUTE THE ASCENT ONLY IF THE DICE SCORE IS SUPERIOR TO tau
                if -batch_loss > 1 - tau:
                    break

                batch_loss.backward()
                optimizer.step()

                # COMPUTE THE NORM DIFFERENCE
                norm_diff = torch.norm(
                    vector_params_detached(self.g_model)
                    - vector_params_detached(w_ref)
                )
                # PROJECT IF THE NORM IS BIGGER THAN delta
                if norm_diff > delta:
                    for w_k, w_ref_k in \
                            zip(self.g_model.parameters(), w_ref.parameters()):
                        w_k.data.mul_(delta/norm_diff)
                        w_k.data.add_((1 - delta/norm_diff) * w_ref_k)

        elif self.unlearn_scheme in ["fine-tuning", "FedAccum", "scratch"]:
            # NOTHING TO DO
            pass

    def aggregation(self, P: list[float], model: Module,
                    contributions: list[Module]) -> Module:

        if len(P) != len(contributions):
            raise Exception("Dimension issue")

        # IMPORTANCE OF THE PREVIOUS GLOBAL MODEL IN THE NEW ONE
        for w_k in model.parameters():
            w_k.data.mul_(1 - self.lr_g * np.sum(P))

        # ADD THE CONTRIBUTIONS OF THE PARTICIPANTS
        for p_i, rec_i in zip(P, contributions):
            for new_w_k, r_w_ik in zip(model.parameters(), rec_i.parameters()):
                new_w_k.data.add_(self.lr_g * p_i * r_w_ik)

    def aggregation_round(self, local_models: list[Module]) -> Module:
        """Aggregates the clients models to create the new model"""

        P = self.P_kept[np.where(self.P_kept > 0)]
        received_models = [local_models[i] for i in np.where(self.P_kept > 0)[0]]

        self.aggregation(P, self.g_model, received_models)

        self.t += 1

    def compute_metric(self, local_models: list[Module]):
        """Compute Psi as in the paper for every client"""

        # SIFU
        coef_regu = (1 - self.lr_l * self.lambd) ** self.n_SGD
        self.metric[self.t] = coef_regu * self.metric[self.t - 1]

        vec_g = vector_params_detached(self.g_model)
        # print(vec_g)

        for l, model_l in zip(self.working_clients, local_models):
            vec_l = vector_params_detached(model_l)

            coef_norm_l = self.lr_g * self.P_kept[l] / (1 - self.P_kept[l])

            self.metric[self.t, l] += coef_norm_l * torch.norm(vec_g - vec_l, p=2)

            self.metric_univariate[l] = coef_regu * self.metric_univariate[l]
            self.metric_univariate[l] += coef_norm_l * torch.abs(vec_g - vec_l)

    def keep_best_model(self, local_models: list[Module]):

        # SAVE THE BEST MODEL FOR SIFU WITH OUR METRIC.
        # IF METRIC BELOW THRESHOLD, CURRENT = BEST
        self.compute_metric(local_models)
        for key, i in product(self.best_SIFU.keys(), self.working_clients):
            if self.metric[self.t, i] <= self.psi_star[key]:
                self.best_SIFU[key][i] = deepcopy(self.g_model)

        # SAVE THE BEST MODEL FOR SIFU IMPROVED
        for key, i in product(self.best_SIFU_improved.keys(), self.working_clients):

            std_key_i = get_std(
                self.metric_univariate[i],
                CONFIG_SIFU[key]["epsilon"],
                CONFIG_SIFU[key]["delta"]
            )

            if torch.max(std_key_i) <= CONFIG_SIFU[key]["sigma"]:
                print(key, i, self.t)
                self.best_SIFU_improved[key][i] = deepcopy(self.g_model)

        # SUBSTRACT THE CONTRIBUTION OF THE CLIENT FOR FUKD
        for model_FUKD, local_model in zip(self.models_FUKD, local_models):
            for w_k_unlearn, w_k in zip(model_FUKD.parameters(),
                                        local_model.parameters()):
                # print(w_k_unlearn)
                # print(w_k)
                with torch.no_grad():
                    w_k_unlearn.add_(-w_k/self.M)

        # CREATE THE UPDATED NEW MODEL FOR FedAccum and PGD
        for i in self.working_clients:
            P_without_i = deepcopy(self.P_kept)
            P_without_i[i] = 0
            P_without_i = P_without_i[np.where(P_without_i > 0)] / sum(P_without_i)

            contributions_without_i = deepcopy(local_models)
            del contributions_without_i[i]

            self.saved_FedAccum[i].to(device)

            self.aggregation(P_without_i, self.saved_FedAccum[i],
                             contributions_without_i)

        # CREATE THE MODEL FOR PGD
        for i in self.working_clients:
            remaining_clients = deepcopy(self.working_clients)
            remaining_clients.remove(i)


    def loss_acc_global(self, loss_train, loss_test) -> (float, float):

        # loss_all = clients.loss_clients(self.g_model, self.loss_f, ds_type)
        # print(loss_all)

        self.loss[0, self.t] = loss_train
        self.loss[1, self.t] = loss_test

        loss_mean_train = np.dot(self.P_kept, self.loss[0, self.t])
        loss_mean_test = np.dot(self.P_kept, self.loss[1, self.t])
        # var_loss = np.dot(self.P_kept, (loss_mean - self.loss[self.t]) ** 2) ** 0.5

        log = {
            "t": self.t,
            "loss_all_train": loss_mean_train,
            "loss_all_test": loss_mean_test
        }
        for i, loss_i in enumerate(loss_train):
            log[f"loss_C{i}_train"] = loss_i

        for i, loss_i in enumerate(loss_test):
            log[f"loss_C{i}_test"] = loss_i
        wandb.log(log)

        print(
            f"====> n: {self.t} "
            f"TRAIN: {loss_mean_train:.3} "
            f"TEST:+- {loss_mean_test:.3}"
        )
        return loss_mean_train, loss_mean_test

    def metric_save(self, metric: str):
        path_file = f"saved_exp_info/{metric}/{self.file_name}.pkl"
        # array_variable = getattr(self, metric)
        # array_variable = array_variable[np.where(array_variable>0)]
        with open(path_file, "wb") as output:
            pickle.dump(getattr(self, metric), output)

    def save_last_model(self):
        base = f"saved_exp_info/models/{self.file_name}"
        path_file = f"{base}_last.pth"
        torch.save(self.g_model.state_dict(), path_file)

    def save_unlearning_models(self):

        base = f"saved_exp_info/models/{self.file_name}.pth"

        # SIFU SIFUimp
        for i, key in product(range(self.M), self.best_SIFU.keys()):
            path_SIFU_key_i = base.replace("train", f"SIFU-{key}")
            path_SIFU_key_i = path_SIFU_key_i.replace("P0", f"P{i}")
            torch.save(self.best_SIFU[key][i].state_dict(), path_SIFU_key_i)

            path_SIFUimp_key_i = path_SIFU_key_i.replace("SIFU", "SIFUimp")
            torch.save(self.best_SIFU_improved[key][i].state_dict(),
                       path_SIFUimp_key_i)

        # FedAccum and PGD
        for i, model_FA_i in enumerate(self.saved_FedAccum):
            # path_FedAccum_i = f"{base}_FedAccum_P{i}.pth"
            path_FedAccum_i = base.replace("train", f"FedAccum")
            path_FedAccum_i = path_FedAccum_i.replace("P0", f"P{i}")
            torch.save(model_FA_i.state_dict(), path_FedAccum_i)

        # FUKD
        for i, model_FUKD in enumerate(self.models_FUKD):
            for w_k_FUKD, w_k in zip(model_FUKD.parameters(),
                                        self.g_model.parameters()):
                with torch.no_grad():
                    w_k_FUKD.add_(w_k)
            path_FUKD_i = base.replace("train", "FUKD")
            path_FUKD_i = path_FUKD_i.replace("P0", f"P{i}")
            torch.save(model_FUKD.state_dict(), path_FUKD_i)

        # last, fine-tuning
        self.save_last_model()

        # scratch -> do nothing

    def save_metrics_and_models(self, save_unlearning: bool = False):

        if self.unlearn_scheme == "train" and save_unlearning:
            self.save_unlearning_models()
            self.metric_save("metric")
        else:
            self.save_last_model()

        self.metric_save("loss")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy


def get_tau_i(n_clients: int, time_scenario: str, verbose=True):
    """returns every client tau_i"""

    np.random.seed(42)
    P_slower = float(time_scenario.split("-")[1])
    tau_i = np.array([1 - i * P_slower/100 / (n_clients - 1) for i in range(n_clients)])
    return tau_i


def clients_time_schedule(
    n_clients: np.array,
    opt_scheme: str,
    time_scenario: str,
    T: int,
    verbose=True
):

    tau_i = get_tau_i(n_clients, time_scenario, True)

    if opt_scheme == "FL":

        # an iteration takes the time of the slowest client
        n_aggreg = int(T / max(tau_i))

        # every opt. step takes the time of the slowest client
        step_times = np.array([i * max(tau_i) for i in range(n_aggreg + 1)])

        # every client works on the current global model
        schedule_global = np.array([list(range(n_aggreg))] * n_clients).T

    elif opt_scheme == "Async":

        # time at which a client returns its work
        return_times = deepcopy(tau_i)

        # loop to simulate the flow of clients update until T
        schedule_global = []
        step_times = [0]

        # latest sent model index for every client
        latest_sent_models = [0] * n_clients

        # current opt. step and its associated time
        n = 0

        while np.min(return_times) <= T:

            # Next incoming client (with earliest return time)
            idx_client = np.argmin(return_times)

            # Update the current time
            step_times.append(return_times[idx_client])

            # New return time for the client that participated
            return_times[idx_client] = step_times[-1] + tau_i[idx_client]

            # global model idx the client performed its local work on
            v = - np.ones( n_clients)
            v[idx_client] = latest_sent_models[idx_client]
            schedule_global.append(v)

            # New opt. step index
            n += 1

            # New global model received by the participating client
            latest_sent_models[idx_client] = n

        n_aggreg = n
        schedule_global = np.array(schedule_global)

    elif opt_scheme.split("-")[0] == "FedFix":

        delta_t = float(opt_scheme.split("-")[1])

        # time at which each client returns its work
        return_times = deepcopy(tau_i)

        # Simulate the flow of updates
        n_aggreg = int(T / delta_t)
        schedule_global = - np.ones((n_aggreg, n_clients))
        step_times = np.array([i * delta_t for i in range(n_aggreg)])

        # Latest sent model index for every client
        latest_sent_models = np.zeros(n_clients)

        for n in range(n_aggreg):

            # Indices of the participating clients
            idx_clients = np.where(return_times <= (n + 1) * delta_t)[0]

            # New return time for the participating clients
            return_times[idx_clients] = (n + 1) * delta_t + tau_i[idx_clients]

            # idx of the global model received by participating clients
            schedule_global[n , idx_clients] = latest_sent_models[idx_clients]
            latest_sent_models[idx_clients] = n + 1

    elif opt_scheme.split("-")[0] == "FedBuff":

        # amount of clients to wait before aggreg.
        c = int(opt_scheme.split("-")[1])

        # time at which each client returns its work
        return_times = deepcopy(tau_i)

        # loop to simulate the flow of clients update until T
        schedule_global = []
        step_times = [0]

        # latest sent model index for every client
        latest_sent_models = [0] * n_clients

        # current opt. step
        n = 0

        while np.min(return_times) <= T:
            # Next c incoming client (with earliest return time)
            idx_client = np.argpartition(return_times, c)[:c]

            # Update the current time
            step_times.append(np.max(return_times[idx_client]))

            # New return time for the client that participated
            for i in idx_client:
                return_times[i] = step_times[-1] + tau_i[i]

            # global model idx the client performed its local work on
            v = - np.ones(n_clients)
            for i in idx_client:
                v[i] = latest_sent_models[i]
            schedule_global.append(v)

            # New opt. step index
            n += 1

            # New global model received by the participating client
            for i in idx_client:
                latest_sent_models[i] = n

        n_aggreg = n
        schedule_global = np.array(schedule_global)

    if verbose:
        print("Number of server aggregations:", n_aggreg)

    step_times = np.array(step_times)
    schedule_global = np.array(schedule_global).astype(int)

    return step_times, schedule_global

import torch

import numpy as np
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Branch:
    def __init__(self, branch: list[list[int, int]] = [[0, 0]]):
        self.branch = deepcopy(branch)

    def __call__(self):
        return self.branch

    def cut(self, metric: np.array, r: int, W_r: list[int], psi_star: float):
        """Cut the branch to only keep nodes forgetting W_r.
        If they all forget W_r then add the previous flow"""

        # NO CUT NEEDED FOR THE BRANCH. NO CLIENT TO FORGET
        if len(W_r) == 0:
            return

        for i, (zeta_s, T_s) in enumerate(self.branch):
            if max(metric[zeta_s, T_s, W_r]) > psi_star:
                self.branch = self.branch[:i]
                self.branch.append([zeta_s, -42])
                return

        self.branch.append([r - 1, -42])

    def get_T(self, metric: np.array, zeta: int, W_r: list[int], psi_star: float) -> int:

        psi_Sr = np.max(metric[zeta, :, W_r], axis=0)

        indices = np.where(psi_Sr > psi_star)[0]

        if len(indices) > 0:
            return np.min(indices) - 1
        else:
            return np.max(np.where(psi_Sr > 0)[0])

    def update(self, metric: np.array, r: int, W_r: list[int], psi_star: float):

        # THE BRANCH REMAINS THE INITIALIZATION
        if r == 0:
            return

        self.cut(metric, r, W_r, psi_star)

        zeta_r = self.branch[-1][0]
        T_r = self.get_T(metric, zeta_r, W_r, psi_star)
        if [zeta_r, T_r] == self.branch[-2]:
            self.branch = self.branch[:-1]
        else:
            self.branch[-1][1] = T_r

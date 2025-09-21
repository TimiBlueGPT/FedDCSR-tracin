# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from collections import OrderedDict
from collections import defaultdict
import copy

class ClientAttributor:
    def __init__(self):
        self.scores = defaultdict(float)  # cid -> score

    def add_score(self, cid, score: float):
        self.scores[cid] += float(score)

    def dump_topk(self, k=20):
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:k]

class Server(object):
    def __init__(self, args, init_global_params):
        self.args = args
        self.global_params = self._clone_param_list(init_global_params)
        self.latest_progress_direction = None
        self.latest_client_scores = {}
        self.attributor = ClientAttributor()
        if args.method == "FedDCSR":
            self.global_reps = None
        




    def aggregate_params(self, clients, random_cids):
        """Sums up parameters of models shared by all active clients at each
        epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        # Record the model parameter aggregation results of each branch
        # separately
        prev_global_params = self._clone_param_list(self.global_params)
        num_branchs = len(self.global_params)
        print(num_branchs)
        new_global_params = [None] * num_branchs
        for branch_idx in range(num_branchs):
            client_params_sum = None
            for c_id in random_cids:
                # Obtain current client's parameters
                current_client_params = clients[c_id].get_params()[branch_idx]
                # Sum it up with weights
                if client_params_sum is None:
                    client_params_sum = OrderedDict(
                        (key, current_client_params[key]
                         * clients[c_id].train_weight)
                        for key in current_client_params.keys())
                else:
                    for key in client_params_sum.keys():
                        client_params_sum[key] += (
                                clients[c_id].train_weight
                                * current_client_params[key])
            if client_params_sum is None:
                client_params_sum = self._clone_param_branch(
                    prev_global_params[branch_idx])
            new_global_params[branch_idx] = client_params_sum

        self.global_params = new_global_params
        progress_direction = self._compute_progress_direction(
            prev_global_params, self.global_params)
        self.latest_progress_direction = progress_direction
        self._update_progress_scores(clients, random_cids, progress_direction)

    def aggregate_reps(self, clients, random_cids):
        """Sums up representations of user sequences shared by all active
        clients at each epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        # Record the user sequence aggregation results of each branch
        # separately
        client_reps_sum = None
        for c_id in random_cids:
            # Obtain current client's user sequence representations
            current_client_reps = clients[c_id].get_reps_shared()
            # Sum it up with weights
            if client_reps_sum is None:
                client_reps_sum = current_client_reps * \
                    clients[c_id].train_weight
            else:
                client_reps_sum += clients[c_id].train_weight * \
                    current_client_reps
        self.global_reps = client_reps_sum

    def choose_clients(self, n_clients, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]

    def add_eval_score(self, cid, score):
        """Accumulates attribution score reported by a client."""
        if score is None:
            return
        self.attributor.add_score(cid, score)

    def get_global_params(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_params

    def get_global_reps(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_reps


    @staticmethod
    def _clone_param_branch(branch):
        if branch is None:
            return None
        cloned = OrderedDict()
        for key, value in branch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    @classmethod
    def _clone_param_list(cls, params):
        if params is None:
            return None
        return [cls._clone_param_branch(branch) for branch in params]

    @staticmethod
    def _compute_progress_direction(prev_params, new_params):
        if (prev_params is None) or (new_params is None):
            return None
        progress_direction = []
        for prev_branch, new_branch in zip(prev_params, new_params):
            branch_direction = OrderedDict()
            if (prev_branch is None) or (new_branch is None):
                progress_direction.append(branch_direction)
                continue
            for key, new_value in new_branch.items():
                prev_value = prev_branch[key]
                branch_direction[key] = -(new_value - prev_value)
            progress_direction.append(branch_direction)
        return progress_direction

    def _update_progress_scores(self, clients, random_cids, progress_direction):
        if (progress_direction is None) or (len(random_cids) == 0):
            self.latest_client_scores = {}
            return

        round_scores = {}
        for c_id in random_cids:
            client_grads = clients[c_id].get_grads()
            score = 0.0
            for branch_grad, branch_direction in zip(client_grads, progress_direction):
                if branch_grad is None:
                    continue
                for key, grad_val in branch_grad.items():
                    if key not in branch_direction:
                        continue
                    score += float(torch.sum(grad_val * branch_direction[key]))
            round_scores[c_id] = score
            self.attributor.add_score(c_id, score)

        self.latest_client_scores = round_scores

    def get_latest_progress_scores(self):
        return self.latest_client_scores

    def get_progress_direction(self):
        return self.latest_progress_direction

    def get_topk_contributors(self, k=20):
        return self.attributor.dump_topk(k)
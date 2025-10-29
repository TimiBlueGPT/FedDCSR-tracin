# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from collections import OrderedDict
from collections import defaultdict

class ClientAttributor:
    def __init__(self):
        self.scores = defaultdict(float)

    def add_score(self, cid, score: float):
        self.scores[cid] += float(score)

    def current_score(self, cid, score: float):
        self.scores[cid] = float(score)

    @staticmethod
    def _softmax(scores_dict, temperature: float):
        print(scores_dict)
        if not scores_dict:
            return {}
        if temperature == 0:
            raise ValueError("Temperature for softmax normalization must be non-zero.")
        exp_scores = {
            cid: math.exp(score / temperature) for cid, score in scores_dict.items()
        }
        total = sum(exp_scores.values())
        if total == 0:
            return {cid: 0.0 for cid in scores_dict}
        return {cid: val / total for cid, val in exp_scores.items()}

    def dump_topk(self, k=20, T=1500.0):
        softmax_scores = self._softmax(self.scores, T)
        return sorted(softmax_scores.items(), key=lambda x: x[1], reverse=True)[:k]

class Server(object):
    def __init__(self, args, init_global_params):
        self.args = args
        self.global_params = init_global_params
        self.attributor = ClientAttributor()
        if args.method == "VeriFRL_Fed":
            self.global_reps = None
        




    def aggregate_params(self, clients, random_cids):
        num_branchs = len(self.global_params)
        print(num_branchs)
        for branch_idx in range(num_branchs):
            client_params_sum = None
            for c_id in random_cids:
                current_client_params = clients[c_id].get_params()[branch_idx]
                if client_params_sum is None:
                    client_params_sum = dict((key, value
                                              * clients[c_id].train_weight)
                                             for key, value
                                             in current_client_params.items())
                else:
                    for key in client_params_sum.keys():
                        client_params_sum[key] += clients[c_id].train_weight \
                            * current_client_params[key]
            self.global_params[branch_idx] = client_params_sum

    def aggregate_reps(self, clients, random_cids):
        client_reps_sum = None
        for c_id in random_cids:
            current_client_reps = clients[c_id].get_reps_shared()
            if client_reps_sum is None:
                client_reps_sum = current_client_reps * \
                    clients[c_id].train_weight
            else:
                client_reps_sum += clients[c_id].train_weight * \
                    current_client_reps
        self.global_reps = client_reps_sum

    def choose_clients(self, n_clients, ratio=1.0):
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]

    def add_eval_score(self, cid, score):
        if score is None:
            return
        self.attributor.add_score(cid, score)

    def set_current_score(self, cid, score):
        if score is None:
            return
        self.attributor.current_score(cid, score)

    def get_global_params(self):
        return self.global_params

    def get_global_reps(self):
        return self.global_reps

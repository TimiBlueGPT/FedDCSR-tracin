# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import SeqDataloader
from utils.io_utils import ensure_dir
from utils.influence_utils import merge_influence_records

class Client:
    def __init__(self, model_fn, c_id, args, adj, train_dataset, valid_dataset, test_dataset):
        # Used for computing the mask in self-attention module
        self.num_items = train_dataset.num_items
        self.domain = train_dataset.domain
        # Used for computing the positional embeddings
        self.max_seq_len = args.max_seq_len
        self.trainer = model_fn(args, self.num_items, self.max_seq_len)
        self.model = self.trainer.model
        self.method = args.method
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = args.id if len(args.id) > 1 else "0" + args.id
        if args.method == "FedDCSR":
            self.z_s = self.trainer.z_s
            self.z_g = self.trainer.z_g
        self.c_id = c_id
        self.args = args
        self.adj = adj
        self.branch_influence_log = {"train": [], "valid": [], "test": []}
        self.train_dataloader = SeqDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = SeqDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = SeqDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0
        # Model evaluation results
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.init_global_params = copy.deepcopy(self.get_params())


    def train_epoch(self, round, args, global_params=None):
        """Trains one client with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
            global_params: Global model parameters used in `FedProx` method.
        """
        self.trainer.model.train()
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            for batch_idx, (_, sessions) in enumerate(self.train_dataloader, start=1):
                if ("Fed" in args.method) and args.mu:
                    batch_output = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args,
                        global_params=global_params)
                else:
                    batch_output = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args)
                if isinstance(batch_output, tuple):
                    batch_loss, batch_branch_scores = batch_output
                    if batch_branch_scores:
                        self._log_branch_scores(
                            phase="train",
                            scores=batch_branch_scores,
                            round_idx=round,
                            batch_idx=batch_idx,
                        )
                else:
                    batch_loss = batch_output
                loss += batch_loss
                step = batch_idx

            gc.collect()

        if getattr(self.trainer, "enable_influence", False):
            records = self.trainer.consume_train_influence()
            aggregated = merge_influence_records(records)
            if aggregated:
                self.branch_influence_log["train"].append(aggregated)
                self._log_branch_scores(
                    phase="train",
                    scores=aggregated,
                    round_idx=round,
                    summary=True,
                )

        logging.info("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.epochs, self.c_id, loss / step))
        return self.n_samples_train



    def evaluation(self, mode="valid"):
        """Evaluates one client with its own valid/test data for one epoch.

        Args:
            mode: `valid` or `test`.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()
        if (self.method == "FedDCSR") or ("VGSAN" in self.method):
            self.trainer.model.graph_convolution(self.adj)
        pred = []
        for batch_idx, (_, sessions) in enumerate(dataloader, start=1):
            batch_output = self.trainer.test_batch(sessions)
            if isinstance(batch_output, tuple):
                predictions, batch_branch_scores = batch_output
                if batch_branch_scores:
                    self._log_branch_scores(
                        phase=mode,
                        scores=batch_branch_scores,
                        batch_idx=batch_idx,
                    )
            else:
                predictions = batch_output
            pred = pred + predictions
        gc.collect()
        if getattr(self.trainer, "enable_influence", False):
            records = self.trainer.consume_eval_influence()
            aggregated = merge_influence_records(records)
            if aggregated:
                key = "valid" if mode == "valid" else "test"
                self.branch_influence_log[key].append(aggregated)
                self._log_branch_scores(
                    phase=key,
                    scores=aggregated,
                    summary=True,
                )
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = self.cal_test_score(pred)
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}

    def _log_branch_scores(self, phase, scores, round_idx=None, batch_idx=None,
                           summary=False):
        """Emit a formatted logging message for branch influence values."""
        if not scores:
            return
        formatted_pairs = ", ".join(
            f"{key}={value:.4f}" for key, value in sorted(scores.items())
        )
        context_parts = [f"client {self.c_id}"]
        if round_idx is not None:
            context_parts.append(f"round {round_idx}")
        if batch_idx is not None:
            label = "summary" if summary else f"batch {batch_idx}"
            context_parts.append(label)
        elif summary:
            context_parts.append("summary")
        context = ", ".join(context_parts)
        logging.info(
            "[%s] branch influence - %s",
            phase,
            formatted_pairs if context == "" else f"{context}: {formatted_pairs}"
        )

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest epoch.
        """
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}

    @ staticmethod
    def cal_test_score(predictions):
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # `pred` indicates the rank of groundtruth items in the recommendation
        # list
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
        return MRR/valid_entity, NDCG_5 / valid_entity, \
            NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / \
            valid_entity, HR_10 / valid_entity


    def get_grads(self):
        """Returns gradients of the shared model parameters.

        Gradients are computed as the difference between the current model
        parameters and the global parameters that were used to initialise this
        client's model at the beginning of the round (stored in
        ``self.init_global_params``).
        """
        assert hasattr(self, "init_global_params"), \
            "`init_global_params` missing. Did you forget to call `set_global_params`?"
        current_params = self.get_params()
        grads = []
        for branch_idx in range(len(current_params)):
            branch_grads = {}
            for key in current_params[branch_idx]:
                branch_grads[key] = (current_params[branch_idx][key]
                                     - self.init_global_params[branch_idx][key])
            grads.append(branch_grads)
        return grads

    def get_client_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone())
            else:
                grads.append(torch.zeros_like(param))
        return grads

    def get_params(self):
        """Returns the model parameters that need to be shared between clients.
        """
        if self.method == "FedDCSR":
            return copy.deepcopy([self.model.encoder_s.state_dict()])
        elif "VGSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "SASRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "VSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "ContrastVAE" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "CL4SRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "DuoRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])

    def get_reps_shared(self):
        """Returns the user sequence representations that need to be shared
        between clients.
        """
        assert (self.method == "FedDCSR")
        return copy.deepcopy(self.z_s[0].detach())

    def get_branch_influence(self, mode=None):
        """Return branch influence statistics collected on the client."""
        if mode is None:
            return copy.deepcopy(self.branch_influence_log)
        return copy.deepcopy(self.branch_influence_log.get(mode, []))

    def set_global_params(self, global_params):
        """Assign the local shared model parameters with global model
        parameters.
        """
        assert (self.method in ["FedDCSR", "FedVGSAN", "FedSASRec", "FedVSAN",
                                "FedContrastVAE", "FedCL4SRec", "FedDuoRec"])
        self.init_global_params = copy.deepcopy(global_params)
        if self.method == "FedDCSR":
            self.model.encoder_s.load_state_dict(global_params[0])
        elif self.method == "FedVGSAN":
            self.model.encoder.load_state_dict(global_params[0])
            # self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedSASRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedVSAN":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedContrastVAE":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedCL4SRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedDuoRec":
            self.model.encoder.load_state_dict(global_params[0])

    def set_global_reps(self, global_rep):
        """Copy global user sequence representations to local.
        """
        assert (self.method == "FedDCSR")
        self.z_g[0] = copy.deepcopy(global_rep)

    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                for domain
                                                 in self.args.domains]),
                                        self.method + "_" + self.model_id)
        ensure_dir(method_ckpt_path, verbose=True)
        ckpt_filename = os.path.join(
            method_ckpt_path, "client%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                             for domain in self.args.domains]),
                                     self.method + "_" + self.model_id,
                                     "client%d.pt" % self.c_id)
        try:
            checkpoint = torch.load(ckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)

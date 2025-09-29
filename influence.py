import copy
import json
import logging
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


class InfluenceCalculator:
    """Offline influence function estimator for FedDCSR clients."""

    def __init__(self, clients, args):
        self.clients = clients
        self.args = args

    def run(self):
        if not self.clients:
            logging.warning("No clients found. Skip influence calculation.")
            return
        if self.args.method != "FedDCSR":
            logging.info(
                "Influence calculation is currently implemented only for FedDCSR.")
            return

        base_dir = os.path.dirname(self.clients[0].get_checkpoint_path())
        round_dirs = self._discover_round_checkpoints(base_dir)
        if not round_dirs:
            logging.warning(
                "No round-level checkpoints were found under %s. "
                "Skipping influence estimation.",
                base_dir,
            )
            return

        results = {}
        for round_id, round_name in round_dirs:
            round_scores = {}
            for client in self.clients:
                ckpt_path = client.get_checkpoint_path(round_idx=round_id)
                if not os.path.exists(ckpt_path):
                    continue
                client_scores = self._compute_scores_for_checkpoint(
                    client, ckpt_path)
                if client_scores:
                    round_scores[str(client.c_id)] = client_scores
            if round_scores:
                results[round_name] = round_scores

        if not results:
            logging.warning("Influence scores could not be computed.")
            return

        output_path = os.path.join(base_dir, "influence_scores.json")
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        logging.info("Saved influence scores to %s", output_path)

    def _discover_round_checkpoints(self, base_dir: str) -> List[Tuple[int, str]]:
        if not os.path.isdir(base_dir):
            return []
        round_dirs: List[Tuple[int, str]] = []
        for entry in os.listdir(base_dir):
            path = os.path.join(base_dir, entry)
            if not os.path.isdir(path):
                continue
            if not entry.startswith("round_"):
                continue
            try:
                round_id = int(entry.split("_")[-1])
            except ValueError:
                continue
            round_dirs.append((round_id, entry))
        round_dirs.sort(key=lambda item: item[0])
        return round_dirs

    def _branch_parameters(self, client) -> Dict[str, List[torch.nn.Parameter]]:
        if client.method != "FedDCSR":
            return {}
        return {
            "shared": [
                param for param in client.model.encoder_s.parameters()
                if param.requires_grad
            ],
            "exclusive": [
                param for param in client.model.encoder_e.parameters()
                if param.requires_grad
            ],
        }

    @staticmethod
    def _flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.empty(0)
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    @staticmethod
    def _unflatten_like(vector: torch.Tensor,
                        params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        views: List[torch.Tensor] = []
        pointer = 0
        for param in params:
            numel = param.numel()
            views.append(vector[pointer:pointer + numel].view_as(param))
            pointer += numel
        return views

    def _compute_scores_for_checkpoint(self, client, ckpt_path: str) -> Dict[str, float]:
        device = client.trainer.device
        original_state = copy.deepcopy(client.trainer.model.state_dict())
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            client.trainer.model.load_state_dict(checkpoint)

            branch_params = self._branch_parameters(client)
            if not branch_params:
                return {}

            val_grads = self._compute_validation_gradients(client, branch_params)
            train_batches = self._sample_train_batches(client)
            if not train_batches:
                return {}
            train_grads = self._compute_train_gradients(
                client, branch_params, train_batches)

            inv_hvps = {}
            for branch_name, params in branch_params.items():
                grad_vector = train_grads.get(branch_name)
                if grad_vector is None:
                    inv_hvps[branch_name] = None
                    continue
                inv_hvps[branch_name] = self._approximate_inverse_hvp(
                    client, params, train_batches, grad_vector)

            scores = {}
            for branch_name in branch_params.keys():
                val_vector = val_grads.get(branch_name)
                inv_hvp_vector = inv_hvps.get(branch_name)
                if val_vector is None or inv_hvp_vector is None:
                    continue
                # Influence = - grad_val^T H^{-1} grad_train
                scores[branch_name] = float(
                    torch.dot(-val_vector, inv_hvp_vector).item())
            return scores
        finally:
            client.trainer.model.load_state_dict(original_state)
            client.trainer.model.zero_grad()

    def _compute_validation_gradients(self, client,
                                      branch_params: Dict[str, List[torch.nn.Parameter]]
                                      ) -> Dict[str, Optional[torch.Tensor]]:
        batches = (tuple(np.array(x, dtype=np.int64) for x in sessions)
                   for _, sessions in client.valid_dataloader)
        return self._accumulate_gradients(
            client, branch_params, batches, mode="valid")

    def _compute_train_gradients(self, client,
                                 branch_params: Dict[str, List[torch.nn.Parameter]],
                                 train_batches: List[Tuple[np.ndarray, ...]]
                                 ) -> Dict[str, Optional[torch.Tensor]]:
        return self._accumulate_gradients(
            client, branch_params, train_batches, mode="train")

    def _accumulate_gradients(self, client,
                              branch_params: Dict[str, List[torch.nn.Parameter]],
                              batches: Iterable[Tuple[np.ndarray, ...]],
                              mode: str) -> Dict[str, Optional[torch.Tensor]]:
        model = client.trainer.model
        prev_mode = model.training
        if mode == "valid":
            model.eval()
        else:
            model.train()

        accumulators: Dict[str, List[torch.Tensor]] = {}
        for branch_name, params in branch_params.items():
            accumulators[branch_name] = [torch.zeros_like(param)
                                         for param in params]

        seen_batch = False
        for sessions in batches:
            seen_batch = True
            model.zero_grad()
            if mode == "valid":
                loss = client.trainer.compute_validation_loss(
                    sessions, client.adj, client.num_items)
            else:
                loss = client.trainer.compute_loss(
                    sessions, client.adj, client.num_items, client.args,
                    include_prox=False, update_state=False)
            loss.backward()
            for branch_name, params in branch_params.items():
                for idx, param in enumerate(params):
                    if param.grad is None:
                        continue
                    accumulators[branch_name][idx] += param.grad.detach().clone()

        gradients: Dict[str, Optional[torch.Tensor]] = {}
        if seen_batch:
            for branch_name, tensors in accumulators.items():
                gradients[branch_name] = self._flatten_tensors(tensors)
        else:
            for branch_name in branch_params.keys():
                gradients[branch_name] = None

        if prev_mode:
            model.train()
        else:
            model.eval()
        model.zero_grad()
        return gradients

    def _sample_train_batches(self, client) -> List[Tuple[np.ndarray, ...]]:
        dataset = client.train_dataloader.dataset
        total_samples = len(dataset)
        if total_samples == 0:
            return []

        ratio = max(0.0, float(self.args.influence_train_sample_ratio))
        sample_count = int(total_samples * ratio) if ratio > 0 else total_samples
        sample_count = max(1, sample_count)
        max_samples = getattr(self.args, "influence_max_train_samples", None)
        if isinstance(max_samples, int) and max_samples > 0:
            sample_count = min(sample_count, max_samples, total_samples)
        else:
            sample_count = min(sample_count, total_samples)

        indices = random.sample(range(total_samples), sample_count)
        batch_size = min(self.args.batch_size, sample_count)

        batches: List[Tuple[np.ndarray, ...]] = []
        for start in range(0, sample_count, batch_size):
            current_indices = indices[start:start + batch_size]
            sessions = []
            for idx in current_indices:
                _, session = dataset[idx]
                sessions.append(copy.deepcopy(session))
            if not sessions:
                continue
            zipped = list(zip(*sessions))
            batches.append(tuple(np.array(x, dtype=np.int64) for x in zipped))
        return batches

    def _approximate_inverse_hvp(self, client, params,
                                 train_batches: List[Tuple[np.ndarray, ...]],
                                 vector: torch.Tensor) -> Optional[torch.Tensor]:
        if vector.numel() == 0:
            return torch.zeros_like(vector)
        damping = float(getattr(self.args, "influence_hvp_damping", 0.01))
        scale = float(getattr(self.args, "influence_hvp_scale", 25.0))
        iters = int(getattr(self.args, "influence_hvp_iterations", 10))

        estimate = vector.clone()
        model = client.trainer.model
        prev_mode = model.training
        model.train()

        for _ in range(max(1, iters)):
            hv = self._hvp(client, params, train_batches, estimate)
            if hv is None:
                estimate = None
                break
            estimate = vector + (1 - damping) * estimate - hv / scale

        if prev_mode:
            model.train()
        else:
            model.eval()
        model.zero_grad()
        return estimate

    def _hvp(self, client, params,
             train_batches: List[Tuple[np.ndarray, ...]],
             vector: torch.Tensor) -> Optional[torch.Tensor]:
        if not train_batches:
            return None
        vector = vector.to(params[0].device) if params else vector
        vec_tensors = self._unflatten_like(vector, params)

        hv_sum: Optional[torch.Tensor] = None
        for sessions in train_batches:
            client.trainer.model.zero_grad()
            loss = client.trainer.compute_loss(
                sessions, client.adj, client.num_items, client.args,
                include_prox=False, update_state=False)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_dot = torch.zeros(1, device=vector.device)
            for grad_tensor, vec in zip(grads, vec_tensors):
                grad_dot = grad_dot + torch.sum(grad_tensor * vec)
            hv = torch.autograd.grad(grad_dot, params, retain_graph=False)
            flat_hv = self._flatten_tensors([h.detach() for h in hv])
            hv_sum = flat_hv if hv_sum is None else hv_sum + flat_hv

        if hv_sum is None:
            return None
        return hv_sum / len(train_batches)
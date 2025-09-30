import json
import logging
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch


def _is_shared_param(name):
    shared_prefixes = ("item_emb_s", "pos_emb_s", "GNN_encoder_s",
                       "encoder_s", "LayerNorm_s")
    return name.startswith(shared_prefixes)


def _is_exclusive_param(name):
    exclusive_prefixes = ("item_emb_e", "pos_emb_e", "GNN_encoder_e",
                          "encoder_e", "LayerNorm_e")
    return name.startswith(exclusive_prefixes)


def _collect_params(model):
    params = []
    branch_indices = {"shared": [], "exclusive": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        idx = len(params) - 1
        if _is_shared_param(name):
            branch_indices["shared"].append(idx)
        if _is_exclusive_param(name):
            branch_indices["exclusive"].append(idx)
    return params, branch_indices


def _zeros_like_params(params):
    return [torch.zeros_like(param) for param in params]


def _grad_or_zeros(grads, params):
    result = []
    for grad, param in zip(grads, params):
        if grad is None:
            result.append(torch.zeros_like(param))
        else:
            result.append(grad)
    return result


def _prepare_validation_batch_for_disen(sessions, num_items, seed):
    seqs_np = np.asarray(sessions[0], dtype=np.int64)
    targets_np = np.asarray(sessions[1], dtype=np.int64)
    batch_size, seq_len = seqs_np.shape
    pad_token = num_items

    ground_truths, ground_masks = [], []
    js_neg_list, contrast_aug_list = [], []
    rng = random.Random(seed)

    for idx in range(batch_size):
        seq = seqs_np[idx].tolist()
        pad_len = 0
        while pad_len < len(seq) and seq[pad_len] == pad_token:
            pad_len += 1
        effective_seq = seq[pad_len:]
        if not effective_seq:
            effective_seq = [pad_token]
            pad_len = max(0, seq_len - 1)

        shifted = effective_seq[1:] + [int(targets_np[idx])]
        ground_seq = [pad_token] * pad_len + shifted
        mask = [0] * pad_len + [1] * len(shifted)

        js_seq = [pad_token] * pad_len + effective_seq
        aug_seq_items = effective_seq.copy()
        rng.shuffle(aug_seq_items)
        aug_seq = [pad_token] * pad_len + aug_seq_items

        if len(ground_seq) != seq_len:
            ground_seq = (ground_seq + [pad_token] * seq_len)[:seq_len]
        if len(mask) != seq_len:
            mask = (mask + [0] * seq_len)[:seq_len]
        if len(js_seq) != seq_len:
            js_seq = (js_seq + [pad_token] * seq_len)[:seq_len]
        if len(aug_seq) != seq_len:
            aug_seq = (aug_seq + [pad_token] * seq_len)[:seq_len]

        ground_truths.append(ground_seq)
        ground_masks.append(mask)
        js_neg_list.append(js_seq)
        contrast_aug_list.append(aug_seq)

    return (seqs_np,
            np.asarray(ground_truths, dtype=np.int64),
            np.asarray(ground_masks, dtype=np.int64),
            np.asarray(js_neg_list, dtype=np.int64),
            np.asarray(contrast_aug_list, dtype=np.int64))


def _compute_validation_gradients(client, params, args, seed_offset):
    trainer = client.trainer
    model = trainer.model
    device = trainer.device
    model.train()
    if (client.method == "FedDCSR") or ("VGSAN" in client.method):
        model.graph_convolution(client.adj)

    grads = _zeros_like_params(params)
    num_batches = 0

    for batch_idx, (_, sessions) in enumerate(client.valid_dataloader):
        prepared_sessions = _prepare_validation_batch_for_disen(
            sessions, client.num_items, seed_offset + batch_idx)
        tensor_sessions = [torch.LongTensor(x).to(device)
                           for x in prepared_sessions]

        seq, ground, ground_mask, js_neg, contrast_aug = tensor_sessions
        result, result_exclusive, mu_s, logvar_s, z_s, mu_e, logvar_e, z_e, \
            neg_z_e, aug_z_e = model(seq, neg_seqs=js_neg, aug_seqs=contrast_aug)
        loss = trainer.disen_vgsan_loss_fn(
            result, result_exclusive, mu_s, logvar_s, mu_e, logvar_e,
            ground, z_s, trainer.z_g[0] if hasattr(trainer, "z_g") else None,
            z_e, neg_z_e, aug_z_e, ground_mask, client.num_items, trainer.step)


        batch_grads = torch.autograd.grad(
            loss, params, allow_unused=True, retain_graph=True,create_graph=True)
        batch_grads = _grad_or_zeros(batch_grads, params)
        grads = [g_acc + g_curr.detach() for g_acc, g_curr in zip(grads, batch_grads)]
        num_batches += 1

    if num_batches == 0:
        return None, 0

    grads = [g / num_batches for g in grads]
    return grads, num_batches


def _compute_train_loss_and_grads(client, params, args):
    trainer = client.trainer
    model = trainer.model
    device = trainer.device
    model.train()
    if (client.method == "FedDCSR") or ("VGSAN" in client.method):
        model.graph_convolution(client.adj)

    target_samples = max(1, int(client.n_samples_train
                                * args.influence_train_ratio))
    batch_size = client.train_dataloader.batch_size
    max_batches = min(client.train_dataloader.num_batch,
                      math.ceil(target_samples / batch_size))

    losses = []
    for batch_idx, (_, sessions) in enumerate(client.train_dataloader):
        seqs = [torch.LongTensor(x).to(device) for x in sessions]
        seq, ground, ground_mask, js_neg_seqs, contrast_aug_seqs = seqs
        result, result_exclusive, mu_s, logvar_s, z_s, mu_e, logvar_e, z_e, \
            neg_z_e, aug_z_e = model(
                seq, neg_seqs=js_neg_seqs, aug_seqs=contrast_aug_seqs)
        loss = trainer.disen_vgsan_loss_fn(
            result, result_exclusive, mu_s, logvar_s, mu_e, logvar_e,
            ground, z_s, trainer.z_g[0] if hasattr(trainer, "z_g") else None,
            z_e, neg_z_e, aug_z_e, ground_mask, client.num_items, trainer.step)
        losses.append(loss)
        if batch_idx + 1 >= max_batches:
            break

    if not losses:
        return None, None, None

    train_loss = torch.stack(losses).mean()
    train_grads = torch.autograd.grad(
        train_loss, params, create_graph=True, retain_graph=True,
        allow_unused=True)
    train_grads = _grad_or_zeros(train_grads, params)
    train_grads_detached = [grad.detach().clone() for grad in train_grads]
    return train_loss, train_grads, train_grads_detached


def _approx_inverse_hvp(train_grads, params, vector_list, args):
    damping = args.hvp_damping
    scale = args.hvp_scale if args.hvp_scale != 0 else 1.0
    num_iter = max(1, args.hvp_iterations)

    def hvp_fn(vec):
        hv = torch.autograd.grad(
            train_grads, params, grad_outputs=vec,create_graph=True, retain_graph=True,
            allow_unused=True)
        return _grad_or_zeros(hv, params)

    estimate = [v.clone().detach() for v in vector_list]
    for _ in range(num_iter):
        hv = hvp_fn(estimate)
        estimate = [v + (1 - damping) * est - hv_i / scale
                    for v, est, hv_i in zip(vector_list, estimate, hv)]
    return [est.detach() for est in estimate]


def _branch_dot_product(grad_list, indices):
    if not indices:
        return 0.0
    grad_vector = torch.cat([grad_list[idx].reshape(-1) for idx in indices])
    return grad_vector


def compute_influence_for_clients(clients, args):
    if not clients:
        logging.info("No clients provided for influence computation.")
        return

    logging.info("Computing influence scores across checkpoints...")
    influence_results = defaultdict(dict)

    for client in clients:
        base_dir = client.get_method_checkpoint_path()
        if not os.path.exists(base_dir):
            logging.warning("Checkpoint directory %s does not exist for client %d.",
                            base_dir, client.c_id)
            continue
        round_dirs = [d for d in os.listdir(base_dir)
                      if d.startswith("round_")]
        if not round_dirs:
            logging.warning("No round checkpoints found for client %d in %s.",
                            client.c_id, base_dir)
            continue

        params, branch_indices = _collect_params(client.trainer.model)
        if not params:
            logging.warning("No trainable parameters collected for client %d.",
                            client.c_id)
            continue

        for round_dir in sorted(round_dirs):
            try:
                round_idx = int(round_dir.split("_")[-1])
            except ValueError:
                logging.warning("Invalid checkpoint directory name %s; skipping.",
                                round_dir)
                continue
            ckpt_path = os.path.join(base_dir, round_dir,
                                     "client%d.pt" % client.c_id)
            if not os.path.exists(ckpt_path):
                logging.warning("Checkpoint %s does not exist; skipping.",
                                ckpt_path)
                continue
            if not client.load_params_from_path(ckpt_path):
                continue

            valid_grads, num_valid_batches = _compute_validation_gradients(
                client, params, args, seed_offset=args.seed + round_idx)
            if num_valid_batches == 0 or valid_grads is None:
                logging.warning("Failed to compute validation gradients for client %d in round %d.",
                                client.c_id, round_idx)
                continue

            train_loss, train_grads, train_grads_detached = \
                _compute_train_loss_and_grads(client, params, args)
            if train_loss is None:
                logging.warning("Failed to compute train gradients for client %d in round %d.",
                                client.c_id, round_idx)
                continue

            inv_hvp = _approx_inverse_hvp(train_grads, params, valid_grads, args)

            branch_scores = {}
            for branch_name, indices in branch_indices.items():
                if not indices:
                    continue
                branch_train = _branch_dot_product(train_grads_detached, indices)
                branch_inv_hvp = _branch_dot_product(inv_hvp, indices)
                if branch_train.numel() == 0 or branch_inv_hvp.numel() == 0:
                    continue
                score = float(-(branch_train * branch_inv_hvp).sum().item())
                branch_scores[branch_name] = score

            influence_results[round_dir]["client_%d" % client.c_id] = branch_scores

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not influence_results:
        logging.info("No influence scores were computed.")
        return

    output_dir = clients[0].get_method_checkpoint_path()
    output_path = os.path.join(output_dir, "influence_scores.json")
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(influence_results, outfile, indent=2)
        logging.info("Influence scores saved to %s", output_path)
    except IOError:
        logging.warning("Failed to save influence scores to %s", output_path)
# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm
from utils.train_utils import EarlyStopping, LRDecay, Return_best
import numpy as np
from utils.influence_utils import compute_influence_for_clients

JAIN_HISTORY = {"MRR": [], "HR @10": [], "NDCG @10": []}
DELTA_JAIN_HISTORY = {"MRR": [], "HR @10": [], "NDCG @10": []}


def temporal_stability_index_adj_diff(values):
    """按相邻差分法计算 Temporal Stability Index (TSI)，不除以均值。"""
    values = np.array(values, dtype=float)
    if len(values) < 2:
        return 1.0
    diffs = np.abs(np.diff(values))
    mean_diff = np.mean(diffs)
    tsi = 1 - mean_diff
    return np.clip(tsi, 0, 1)


def evaluation_logging(eval_logs, round_id, weights, mode="valid"):
    """
    计算平均指标、Jain's Fairness Index、CV、Entropy、TSI、ΔJain（公平性变化率）
    """

    # 日志头
    if mode == "valid":
        logging.info(f"Epoch {round_id} Validation:")
    else:
        logging.info(f"Epoch {round_id} Test:")

    metric_names = list(eval_logs.values())[0].keys()
    sum_dict = {m: 0.0 for m in metric_names}
    sum2_dict = {m: 0.0 for m in metric_names}
    avg_eval_log = {m: 0.0 for m in metric_names}

    # 累加各 domain 指标
    for domain, metrics in eval_logs.items():
        for m in metric_names:
            val = metrics[m]
            sum_dict[m] += val
            sum2_dict[m] += val ** 2
            avg_eval_log[m] += val * weights[domain]

    # ---- 计算 Jain / CV / Entropy ----
    numbers = len(eval_logs)
    jain_index, cv_index, entropy_index = {}, {}, {}
    for m in metric_names:
        mean = sum_dict[m] / numbers
        std = np.sqrt(sum2_dict[m] / numbers - mean ** 2)
        jain_index[m] = (sum_dict[m] ** 2) / (numbers * sum2_dict[m]) if sum2_dict[m] != 0 else 0
        cv_index[m] = std / mean if mean != 0 else 0

        vals = np.array([eval_logs[d][m] for d in eval_logs])
        vals = np.clip(vals, 1e-12, None)
        p = vals / np.sum(vals)
        entropy = -np.sum(p * np.log(p))
        entropy_index[m] = entropy / np.log(numbers)

    # ---- 更新 Jain 历史并计算 TSI、ΔJain ----
    tsi_index, delta_jain = {}, {}
    for m in metric_names:
        # 更新 Jain 历史
        JAIN_HISTORY.setdefault(m, []).append(jain_index[m])

        # Temporal Stability
        tsi_index[m] = temporal_stability_index_adj_diff(JAIN_HISTORY[m])

        # ΔJain: 当前轮与上一轮差
        if len(JAIN_HISTORY[m]) >= 2:
            diff = jain_index[m] - JAIN_HISTORY[m][-2]
        else:
            diff = 0.0
        delta_jain[m] = diff
        DELTA_JAIN_HISTORY.setdefault(m, []).append(diff)

    # ---- 输出日志 ----
    for m in metric_names:
        logging.info(f"{m}: {avg_eval_log[m]:.4f}")
    for m in metric_names:
        logging.info(
            f"Jain's Fairness {m}: {jain_index[m]:.4f} | "
            f"ΔJain({m}): {delta_jain[m]:+.5f} | "
            f"CV({m}): {cv_index[m]:.4f} | "
            f"Entropy({m}): {entropy_index[m]:.4f} | "
            f"TSI({m}): {tsi_index[m]:.4f}"
        )

    for domain, metrics in eval_logs.items():
        logging.info(
            "%s: " % domain + "\t".join([f"{m}: {metrics[m]:.4f}" for m in metric_names])
        )


    return avg_eval_log


def load_and_eval_model(n_clients, clients, args):
    eval_logs = {}
    for c_id in tqdm(range(n_clients), ascii=True):
        clients[c_id].load_params()
        eval_log = clients[c_id].evaluation(mode="test")
        eval_logs[clients[c_id].domain] = eval_log
    weights = dict((client.domain, client.test_weight) for client in clients)
    evaluation_logging(eval_logs, 0, weights, mode="test")


def run_fl(clients, server, args):
    n_clients = len(clients)
    if args.do_eval:
        load_and_eval_model(n_clients, clients, args)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_dir, patience=args.es_patience, verbose=True)
        lr_decay = LRDecay(args.lr, args.decay_epoch,
                           args.optimizer, args.lr_decay,
                           patience=args.ld_patience, verbose=True)
        return_best = Return_best()
        for round in range(1, args.epochs + 1):
            arr = np.array([2, 1, 3, 0])
            random_cids = server.choose_clients(n_clients, args.frac)

            for c_id in range(4):
                logging.info(clients[c_id].train_weight)

            for c_id in tqdm(arr, ascii=True):
                if "Fed" in args.method:

                    clients[c_id].set_global_params(server.get_global_params())
                    if args.method == "VeriFRL_Fed":
                        clients[c_id].set_global_reps(server.get_global_reps())


                clients[c_id].train_epoch(
                    round, args, global_params=server.global_params)
                """
                if hasattr(server, "add_eval_score"):
                    server.add_eval_score(c_id, clients[c_id].latest_eval_score)
                """

                if hasattr(server, "set_current_score"):
                    server.set_current_score(c_id, clients[c_id].latest_eval_score)

            topk_scores = server.attributor.dump_topk()
            logging.info(f"Normalized TracIn scores: {topk_scores}")
            temp_d = dict(topk_scores)
            """for c_id in tqdm(random_cids, ascii=True):
                clients[c_id].train_weight= clients[c_id].train_weight * 0.9 + temp_d[c_id] * 0.1"""

            if "Fed" in args.method:
                server.aggregate_params(clients, random_cids)
                if args.method == "VeriFRL_Fed":
                    server.aggregate_reps(clients, random_cids)
            if args.ckpt_interval and round % args.ckpt_interval == 0:
                for c_id in range(n_clients):
                    clients[c_id].save_round_checkpoint(round)

            if round % args.eval_interval == 0:
                eval_logs = {}
                for c_id in tqdm(range(n_clients), ascii=True):
                    if "Fed" in args.method:
                        clients[c_id].set_global_params(
                            server.get_global_params())
                    if c_id in random_cids:
                        eval_log = clients[c_id].evaluation(mode="valid")
                    else:
                        eval_log = clients[c_id].get_old_eval_log()
                    eval_logs[clients[c_id].domain] = eval_log

                weights = dict((client.domain, client.valid_weight)
                               for client in clients)
                avg_eval_log = evaluation_logging(
                    eval_logs, round, weights, mode="valid")


                early_stopping(avg_eval_log, clients, eval_logs)
                if early_stopping.early_stop:
                    logging.info("Early stopping")
                    break

                lr_decay(round, avg_eval_log, clients)
                """for c_id in tqdm(random_cids, ascii=True):
                    return_best(eval_logs[clients[c_id].domain],clients[c_id])"""
                #return_best(eval_logs[clients[2].domain], clients[2])
        load_and_eval_model(n_clients, clients, args)
        if args.method == "VeriFRL_Fed" and args.compute_influence:
            compute_influence_for_clients(clients, args)

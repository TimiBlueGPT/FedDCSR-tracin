# -*- coding: utf-8 -*-
import logging
import torch
from torch.optim.optimizer import Optimizer



class MyAdagrad(Optimizer):


    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1,
                 weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay,
                        init_accu_value=init_accu_value,
                        weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["sum"] = torch.ones(p.data.size()).type_as(
                    p.data) * init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with "
                            "sparse gradients")
                    grad = grad.add(group["weight_decay"], p.data)

                clr = group["lr"] / \
                    (1 + (state["step"] - 1) * group["lr_decay"])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state["sum"].add_(make_sparse(grad_values.pow(2)))
                    std = state["sum"]._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state["sum"].addcmul_(1, grad, grad)
                    std = state["sum"].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss


def get_optimizer(name, parameters, lr, l2=0):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name in ["adagrad", "myadagrad"]:
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1,
                         weight_decay=l2)
    elif name == "adam":
        return torch.optim.Adam(parameters, weight_decay=l2, lr=lr,
                                betas=(0.9, 0.98))
    elif name == "adamax":
        return torch.optim.Adamax(parameters, weight_decay=l2)
    elif name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))




def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad


class EarlyStopping:

    def __init__(self, checkpoint_path, patience=5, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def is_increase(self, score):
        if score["MRR"] > self.best_score["MRR"] + self.delta:
            return True
        else:
            return False

    def __call__(self, score, clients):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(clients)
        elif not self.is_increase(score):
            self.counter += 1
            logging.info(
                f"Early Stopping counter: {self.counter} "
                f"out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(clients)
            self.counter = 0

    def save_checkpoint(self, clients):
        if self.verbose:
            logging.info("Validation score increased.  Saving model ...")
        for client in clients:
            client.save_params()


class LRDecay:

    def __init__(self, lr, decay_epoch, optimizer, lr_decay, patience=1,
                 verbose=False, delta=0):
        self.patience = patience
        self.current_lr = lr
        self.decay_epoch = decay_epoch
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.verbose = verbose
        self.latest_score = None
        self.delta = delta

    def is_increase(self, score):
        if score["MRR"] > self.latest_score["MRR"] + self.delta:
            return True
        else:
            return False

    def __call__(self, round, score, clients):
        if self.latest_score:
            if round > self.decay_epoch and not self.is_increase(score) \
                    and self.optimizer in ["sgd", "adagrad", "adadelta",
                                           "adam"]:
                self.counter += 1
                logging.info(
                    f"Learning rate decay counter: {self.counter} "
                    f"out of {self.patience}")
                if self.counter >= self.patience:
                    self.current_lr = self.current_lr * self.lr_decay
                    for client in clients:
                        client.trainer.update_lr(self.current_lr)
                    logging.info("Learning rate decay")
                    self.counter = 0
            else:
                self.counter = 0
        self.latest_score = score

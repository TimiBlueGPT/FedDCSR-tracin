from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Concatenate a collection of tensors into a single 1-D tensor."""
    filtered: List[torch.Tensor] = []
    device = None
    for tensor in tensors:
        if tensor is None:
            continue
        if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            continue
        if device is None:
            device = tensor.device
        filtered.append(tensor.reshape(-1))
    if not filtered:
        return torch.tensor([], device=device or torch.device("cpu"))
    return torch.cat(filtered)


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]],
                   params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten gradients while preserving alignment with their parameters."""
    if not params:
        return torch.tensor([], device=torch.device("cpu"))
    flat_parts: List[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat_parts.append(grad.reshape(-1))
    return torch.cat(flat_parts) if flat_parts else torch.tensor([], device=params[0].device)


def _unflatten_like(vector: torch.Tensor,
                    params: Sequence[torch.nn.Parameter]) -> Tuple[torch.Tensor, ...]:
    """Split a flat vector into tensors that match the provided parameters."""
    if vector.numel() == 0:
        return tuple(torch.zeros_like(param) for param in params)
    splits: List[torch.Tensor] = []
    offset = 0
    for param in params:
        numel = param.numel()
        piece = vector[offset: offset + numel].reshape_as(param)
        splits.append(piece)
        offset += numel
    return tuple(splits)


def _hessian_vector_product(loss: torch.Tensor,
                            params: Sequence[torch.nn.Parameter],
                            vector: torch.Tensor,
                            damping: float) -> torch.Tensor:
    """Compute Hessian-vector product with optional damping."""
    if vector.numel() == 0:
        return vector
    vec_split = _unflatten_like(vector, params)
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    grad_dot = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for grad, v in zip(grads, vec_split):
        if grad is None:
            continue
        grad_dot = grad_dot + (grad * v).sum()
    hvps = torch.autograd.grad(
        grad_dot,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    flat_hvp = _flatten_grads(hvps, params)
    if damping:
        flat_hvp = flat_hvp + damping * vector
    return flat_hvp


def _conjugate_gradient(matvec: Callable[[torch.Tensor], torch.Tensor],
                        rhs: torch.Tensor,
                        max_iter: int,
                        tol: float) -> torch.Tensor:
    """Solve ``Ax = rhs`` using the conjugate gradient method."""
    if rhs.numel() == 0:
        return rhs
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    if torch.sqrt(rs_old) <= tol:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if torch.abs(denom) < 1e-12:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) <= tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


def compute_branch_influence(train_loss: torch.Tensor,
                             branch_params: Dict[str, Tuple[torch.nn.Parameter, ...]],
                             eval_loss: Optional[torch.Tensor] = None,
                             cg_damping: float = 0.0,
                             cg_max_iter: int = 50,
                             cg_tol: float = 1e-5,
                             retain_graph: bool = False) -> Dict[str, float]:
    """Estimate per-branch influence scores using a conjugate-gradient solver."""
    if not isinstance(branch_params, dict):
        raise TypeError("`branch_params` must be a dictionary of parameter tuples.")

    influence: Dict[str, float] = {}
    branch_items: List[Tuple[str, Tuple[torch.nn.Parameter, ...]]] = list(branch_params.items())
    eval_loss = train_loss if eval_loss is None else eval_loss

    for idx, (name, param_group) in enumerate(branch_items):
        params = tuple(
            p for p in param_group
            if p is not None and p.requires_grad and p.dtype in (torch.float16, torch.float32, torch.float64)
        )
        if not params:
            influence[name] = 0.0
            continue

        grads_train = torch.autograd.grad(
            train_loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        flat_train = _flatten_grads(grads_train, params)
        if flat_train.numel() == 0:
            influence[name] = 0.0
            continue

        need_retain_eval = retain_graph or (eval_loss is train_loss) or (idx < len(branch_items) - 1)
        grads_eval = torch.autograd.grad(
            eval_loss,
            params,
            retain_graph=need_retain_eval,
            allow_unused=True,
        )
        flat_eval = _flatten_grads(grads_eval, params)

        rhs = flat_train.detach()

        def matvec(vec: torch.Tensor) -> torch.Tensor:
            return _hessian_vector_product(train_loss, params, vec, cg_damping).detach()

        solution = _conjugate_gradient(matvec, rhs, max_iter=cg_max_iter, tol=cg_tol)
        if flat_eval.numel() == 0:
            score = torch.tensor(0.0, device=train_loss.device)
        else:
            score = -torch.dot(flat_eval.detach(), solution)
        influence[name] = float(score.detach().cpu())
    return influence


def merge_influence_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of per-branch influence scores by averaging."""
    if not records:
        return {}
    merged: Dict[str, List[float]] = {}
    for record in records:
        for key, value in record.items():
            merged.setdefault(key, []).append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in merged.items()}
from typing import Dict, Iterable, List, Tuple

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


def compute_branch_influence(loss: torch.Tensor,
                             branch_params: Dict[str, Tuple[torch.nn.Parameter, ...]],
                             retain_graph: bool = False) -> Dict[str, float]:
    """Estimate per-branch influence scores via gradient norms.

    The function approximates the classic influence function by measuring the
    norm of the loss gradient restricted to each parameter subset
    (i.e. branch). Although this omits the inverse Hessian term, the gradient
    magnitude still reflects each branch's relative contribution to the current
    objective and provides a stable proxy for monitoring.

    Args:
        loss: Scalar tensor representing the objective.
        branch_params: Mapping from branch name to the parameters belonging to
            that branch.
        retain_graph: Whether to retain the autograd graph after computing the
            gradients. This should be ``True`` when further backpropagation is
            required (e.g. during training).

    Returns:
        A dictionary that maps branch names to scalar influence scores.
    """
    if not isinstance(branch_params, dict):
        raise TypeError("`branch_params` must be a dictionary of parameter tuples.")

    influence: Dict[str, float] = {}
    branch_names: List[str] = list(branch_params.keys())
    for idx, name in enumerate(branch_names):
        params = tuple(p for p in branch_params[name] if p is not None and p.requires_grad)
        if not params:
            influence[name] = 0.0
            continue
        need_retain = retain_graph or (idx < len(branch_names) - 1)
        grads = torch.autograd.grad(loss, params, retain_graph=need_retain, allow_unused=True)
        flat_grad = _flatten_tensors(grads)
        if flat_grad.numel() == 0:
            score = torch.tensor(0.0, device=loss.device)
        else:
            score = torch.norm(flat_grad, p=2)
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
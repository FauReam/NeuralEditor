"""Utility functions for model editing (ROME / MEMIT).

These helpers compute the statistics needed for rank-one updates
without modifying model weights directly.
"""

from typing import Any

import torch
from torch import Tensor


def compute_covariance(
    keys: list[Tensor],
    lam: float = 1.0,
) -> Tensor:
    """Compute (λI + K^T K)^{-1} for ROME update.

    Args:
        keys: List of key vectors [hidden_dim], collected from calibration data.
        lam: Regularization coefficient (higher = more conservative edit).

    Returns:
        Covariance matrix [hidden_dim, hidden_dim].
    """
    device = keys[0].device
    hidden_dim = keys[0].shape[0]

    # Stack to [num_samples, hidden_dim]
    K = torch.stack(keys, dim=0)
    # K^T K
    cov = K.T @ K  # [hidden_dim, hidden_dim]
    # Add regularization
    cov = cov + lam * torch.eye(hidden_dim, device=device)
    # Invert
    return torch.linalg.inv(cov)


def locate_mlp_layer(model: Any, layer_idx: int) -> torch.nn.Module:
    """Locate the MLP module at a specific layer.

    Supports common architectures (Qwen2, Llama, Mistral).
    """
    # Try common attribute paths
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture for layer access")

    layer = layers[layer_idx]
    # Try common MLP module names
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, name):
            return getattr(layer, name)

    raise ValueError(f"Cannot locate MLP in layer {layer_idx}")


def get_mlp_down_proj(mlp_module: torch.nn.Module) -> torch.nn.Linear:
    """Get the down-projection linear layer from an MLP module."""
    # Try common names across architectures
    for name in ["down_proj", "c_proj", "dense_4h_to_h", "w2"]:
        if hasattr(mlp_module, name):
            return getattr(mlp_module, name)
    raise ValueError("Cannot locate down-projection in MLP module")


def get_mlp_up_proj(mlp_module: torch.nn.Module) -> torch.nn.Linear:
    """Get the up-projection / gate-projection for computing keys."""
    for name in ["up_proj", "c_fc", "dense_h_to_4h", "w1"]:
        if hasattr(mlp_module, name):
            return getattr(mlp_module, name)
    raise ValueError("Cannot locate up-projection in MLP module")


def compute_rome_update(
    W: Tensor,
    k_star: Tensor,
    v_star: Tensor,
    C: Tensor,
) -> Tensor:
    """Compute rank-one update for ROME.

    Args:
        W: Original weight matrix [out_dim, in_dim].
        k_star: Subject key vector [in_dim].
        v_star: Target value vector [out_dim].
        C: Precomputed covariance matrix [in_dim, in_dim].

    Returns:
        Updated weight matrix [out_dim, in_dim].
    """
    # W_new = W_old + (v* - W_old @ k*) @ (C^{-1} @ k*)^T / (k*^T @ C^{-1} @ k*)
    wk = W @ k_star  # [out_dim]
    ck = C @ k_star  # [in_dim]
    denom = k_star @ ck  # scalar

    if abs(denom) < 1e-8:
        raise ValueError("Denominator too small; increase lam or check k_star")

    delta = (v_star - wk).unsqueeze(1) @ (ck / denom).unsqueeze(0)  # [out_dim, in_dim]
    return W + delta


def compute_memit_updates(
    W: Tensor,
    keys: list[Tensor],
    values: list[Tensor],
    C: Tensor,
) -> Tensor:
    """Compute batch update for MEMIT (multiple facts at once).

    Uses the same covariance but handles multiple (k_i, v_i) pairs.
    """
    delta = torch.zeros_like(W)
    for k, v in zip(keys, values):
        wk = W @ k
        ck = C @ k
        denom = k @ ck
        if abs(denom) < 1e-8:
            continue
        delta += (v - wk).unsqueeze(1) @ (ck / denom).unsqueeze(0)
    return W + delta


def generate_edit_target(
    tokenizer: Any,
    model: Any,
    prompt: str,
    target_new: str,
    layer_idx: int = 15,
    token_idx: int = -1,
) -> Tensor:
    """Run a forward pass to extract the target activation at edit position.

    This is a simplified version; production use requires careful position tracking.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of [batch, seq, hidden] for each layer
    hidden = outputs.hidden_states[layer_idx]  # [1, seq, hidden]
    # Extract the hidden state at the target token position
    return hidden[0, token_idx, :].cpu()

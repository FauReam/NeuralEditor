"""ROME and MEMIT model editing framework.

Implements rank-one model editing for precise, surgical intervention
into transformer MLP layers. Designed for single-GPU (4070) use.

References:
    ROME: Mitchell et al. 2022 (arXiv:2202.05262)
    MEMIT: Meng et al. 2023 (arXiv:2210.07229)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.edit_utils import (
    compute_covariance,
    compute_rome_update,
    compute_memit_updates,
    generate_edit_target,
    get_mlp_down_proj,
    get_mlp_up_proj,
    locate_mlp_layer,
)


@dataclass
class EditRequest:
    """A single edit request for model editing."""
    subject: str              # e.g. "牵手"
    relation: str | None      # e.g. "在恋爱中是一种"
    target: str               # e.g. "常见的亲密表达方式"
    layer_idx: int = 15       # Which layer to edit (mid-layer usually best)
    lam: float = 5.0          # Covariance regularization (higher = safer)


@dataclass
class EditState:
    """Serializable state of applied edits."""
    edits: list[dict[str, Any]]

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.edits, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "EditState":
        with open(path, "r", encoding="utf-8") as f:
            edits = json.load(f)
        return cls(edits)


class BaseEditor:
    """Base class for model editors."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.edits: list[dict[str, Any]] = []
        self._original_weights: dict[tuple[int, str], torch.Tensor] = {}

    def save_original(self, layer_idx: int, layer_name: str = "down_proj") -> None:
        """Backup original weight before editing."""
        mlp = locate_mlp_layer(self.model, layer_idx)
        proj = get_mlp_down_proj(mlp)
        key = (layer_idx, layer_name)
        if key not in self._original_weights:
            self._original_weights[key] = proj.weight.data.clone().cpu()

    def restore(self, layer_idx: int, layer_name: str = "down_proj") -> None:
        """Restore a specific layer to its original weight."""
        key = (layer_idx, layer_name)
        if key in self._original_weights:
            mlp = locate_mlp_layer(self.model, layer_idx)
            proj = get_mlp_down_proj(mlp)
            proj.weight.data = self._original_weights[key].to(proj.weight.device)
            print(f"Restored layer {layer_idx}")

    def restore_all(self) -> None:
        """Restore all edited layers."""
        for (layer_idx, layer_name), weight in self._original_weights.items():
            mlp = locate_mlp_layer(self.model, layer_idx)
            proj = get_mlp_down_proj(mlp)
            proj.weight.data = weight.to(proj.weight.device)
        print("All layers restored to original.")


class ROMEEditor(BaseEditor):
    """Rank-One Model Editor for single-fact updates."""

    def __init__(self, model: Any, tokenizer: Any):
        super().__init__(model, tokenizer)
        self._covariance_cache: dict[int, torch.Tensor] = {}

    def _collect_keys(
        self,
        layer_idx: int,
        calibration_prompts: list[str],
    ) -> list[torch.Tensor]:
        """Collect key vectors from calibration data at specified layer."""
        keys = []
        mlp = locate_mlp_layer(self.model, layer_idx)
        up_proj = get_mlp_up_proj(mlp)

        self.model.eval()
        for prompt in calibration_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                hidden = self.model.get_input_embeddings()(inputs.input_ids)
                # Run through model up to the target layer
                # Simplified: use output_hidden_states
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get activation before MLP at layer_idx
                pre_mlp = outputs.hidden_states[layer_idx]  # [1, seq, hidden]
                # Average over sequence positions for calibration
                k = up_proj(pre_mlp[0].mean(dim=0))  # [hidden]
                keys.append(k.cpu())
        return keys

    def _precompute_covariance(
        self,
        layer_idx: int,
        calibration_prompts: list[str],
        lam: float = 5.0,
    ) -> torch.Tensor:
        """Precompute and cache covariance matrix for a layer."""
        if layer_idx in self._covariance_cache:
            return self._covariance_cache[layer_idx]

        print(f"Precomputing covariance for layer {layer_idx}...")
        keys = self._collect_keys(layer_idx, calibration_prompts)
        C = compute_covariance(keys, lam=lam)
        self._covariance_cache[layer_idx] = C
        return C

    def apply(
        self,
        request: EditRequest,
        calibration_prompts: list[str],
    ) -> None:
        """Apply a single ROME edit.

        Args:
            request: What fact to change.
            calibration_prompts: Diverse prompts to estimate key distribution.
        """
        self.save_original(request.layer_idx)

        # 1. Precompute covariance
        C = self._precompute_covariance(request.layer_idx, calibration_prompts, request.lam)

        # 2. Compute subject key (k_star)
        prompt = f"{request.subject}{request.relation or ''}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            mlp = locate_mlp_layer(self.model, request.layer_idx)
            up_proj = get_mlp_up_proj(mlp)
            pre_mlp = outputs.hidden_states[request.layer_idx]
            k_star = up_proj(pre_mlp[0, -1, :]).cpu()  # Last token

        # 3. Compute target value (v_star)
        # Simplified: use the target text's hidden representation
        target_prompt = f"{request.target}"
        v_star = generate_edit_target(
            self.tokenizer, self.model, target_prompt, request.target,
            layer_idx=request.layer_idx,
        ).to(self.model.device)

        # 4. Apply rank-one update
        mlp = locate_mlp_layer(self.model, request.layer_idx)
        down_proj = get_mlp_down_proj(mlp)
        W = down_proj.weight.data  # [out_dim, in_dim]

        W_new = compute_rome_update(W, k_star.to(W.device), v_star, C.to(W.device))
        down_proj.weight.data = W_new

        # Record
        self.edits.append({
            "method": "ROME",
            "subject": request.subject,
            "target": request.target,
            "layer_idx": request.layer_idx,
            "lam": request.lam,
        })
        print(f"ROME edit applied at layer {request.layer_idx}: "
              f"'{request.subject}' -> '{request.target}'")


class MEMITEditor(BaseEditor):
    """Massive Editing in Transformer for batch updates."""

    def __init__(self, model: Any, tokenizer: Any):
        super().__init__(model, tokenizer)
        self._covariance_cache: dict[int, torch.Tensor] = {}

    def _collect_keys(
        self,
        layer_idx: int,
        calibration_prompts: list[str],
    ) -> list[torch.Tensor]:
        """Same as ROME but collects more diverse keys."""
        keys = []
        mlp = locate_mlp_layer(self.model, layer_idx)
        up_proj = get_mlp_up_proj(mlp)

        self.model.eval()
        for prompt in calibration_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                pre_mlp = outputs.hidden_states[layer_idx]
                k = up_proj(pre_mlp[0].mean(dim=0))
                keys.append(k.cpu())
        return keys

    def _precompute_covariance(
        self,
        layer_idx: int,
        calibration_prompts: list[str],
        lam: float = 5.0,
    ) -> torch.Tensor:
        if layer_idx in self._covariance_cache:
            return self._covariance_cache[layer_idx]

        print(f"Precomputing covariance for layer {layer_idx}...")
        keys = self._collect_keys(layer_idx, calibration_prompts)
        C = compute_covariance(keys, lam=lam)
        self._covariance_cache[layer_idx] = C
        return C

    def apply_batch(
        self,
        requests: list[EditRequest],
        calibration_prompts: list[str],
    ) -> None:
        """Apply multiple edits simultaneously using MEMIT.

        More stable than sequential ROME when editing related concepts.
        """
        if not requests:
            return

        # Group by layer
        by_layer: dict[int, list[EditRequest]] = {}
        for req in requests:
            by_layer.setdefault(req.layer_idx, []).append(req)

        for layer_idx, reqs in by_layer.items():
            self.save_original(layer_idx)
            C = self._precompute_covariance(layer_idx, calibration_prompts, reqs[0].lam)

            keys = []
            values = []

            for req in reqs:
                # Subject key
                prompt = f"{req.subject}{req.relation or ''}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    mlp = locate_mlp_layer(self.model, layer_idx)
                    up_proj = get_mlp_up_proj(mlp)
                    pre_mlp = outputs.hidden_states[layer_idx]
                    k = up_proj(pre_mlp[0, -1, :]).cpu()
                    keys.append(k)

                # Target value
                target_prompt = f"{req.target}"
                v = generate_edit_target(
                    self.tokenizer, self.model, target_prompt, req.target,
                    layer_idx=layer_idx,
                ).cpu()
                values.append(v)

            # Batch update
            mlp = locate_mlp_layer(self.model, layer_idx)
            down_proj = get_mlp_down_proj(mlp)
            W = down_proj.weight.data

            W_new = compute_memit_updates(
                W,
                [k.to(W.device) for k in keys],
                [v.to(W.device) for v in values],
                C.to(W.device),
            )
            down_proj.weight.data = W_new

            for req in reqs:
                self.edits.append({
                    "method": "MEMIT",
                    "subject": req.subject,
                    "target": req.target,
                    "layer_idx": layer_idx,
                    "lam": req.lam,
                })

            print(f"MEMIT batch edit applied at layer {layer_idx}: "
                  f"{len(reqs)} facts updated")

    def save_state(self, path: str | Path) -> None:
        """Save edit log (not weights — edits are in-place)."""
        EditState(self.edits).to_json(path)

    @classmethod
    def load_state(cls, path: str | Path) -> EditState:
        return EditState.from_json(path)

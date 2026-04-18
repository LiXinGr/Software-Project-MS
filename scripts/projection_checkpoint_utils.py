from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scripts.train_projection_head import ProjectionHead


_PROJECTION_PREFIXES = (
    "extractor.projection.",
    "module.extractor.projection.",
    "projection.",
    "module.projection.",
)


def _extract_projection_state_dict(checkpoint: dict[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "standalone_projection_head"

    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise KeyError("Checkpoint does not contain a readable model state dict.")

    for prefix in _PROJECTION_PREFIXES:
        projection_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if projection_state:
            return projection_state, "gluefactory_training_checkpoint"

    if any(key.startswith("mlp.") for key in state):
        return state, "raw_projection_state_dict"

    raise KeyError(
        "Could not find projection-head weights in checkpoint. "
        "Expected `model_state_dict` or `extractor.projection.*` keys."
    )


def _infer_projection_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, list[int], int]:
    weight_items = []
    for key, value in state_dict.items():
        if key.startswith("mlp.") and key.endswith(".weight") and value.ndim == 2:
            layer_idx = int(key.split(".")[1])
            weight_items.append((layer_idx, value))

    if not weight_items:
        raise KeyError("Projection state dict is missing `mlp.*.weight` tensors.")

    weight_items.sort(key=lambda item: item[0])
    input_dim = int(weight_items[0][1].shape[1])
    output_dim = int(weight_items[-1][1].shape[0])
    hidden_dims = [int(weight.shape[0]) for _, weight in weight_items[:-1]]
    return input_dim, hidden_dims, output_dim


def load_projection_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, checkpoint_type = _extract_projection_state_dict(checkpoint)

    config = checkpoint.get("config", {})
    if not config and isinstance(checkpoint.get("conf"), dict):
        config = checkpoint["conf"].get("model", {}).get("extractor", {})

    inferred_input_dim, inferred_hidden_dims, inferred_output_dim = _infer_projection_dims(
        state_dict
    )
    hidden_dims = config.get("hidden_dims")
    if hidden_dims is None:
        hidden_dim = config.get("hidden_dim")
        hidden_dims = inferred_hidden_dims if hidden_dim is None else [int(hidden_dim)]

    return {
        "checkpoint_type": checkpoint_type,
        "state_dict": state_dict,
        "input_dim": int(config.get("input_dim", inferred_input_dim)),
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "output_dim": int(config.get("output_dim", inferred_output_dim)),
    }


def build_projection_model(
    checkpoint_path: str | Path,
    device: torch.device,
    eval_mode: bool = True,
) -> tuple[ProjectionHead, dict[str, Any]]:
    artifact = load_projection_checkpoint(checkpoint_path)
    model = ProjectionHead(
        input_dim=artifact["input_dim"],
        hidden_dims=artifact["hidden_dims"],
        output_dim=artifact["output_dim"],
    )
    model.load_state_dict(artifact["state_dict"])
    if eval_mode:
        model.eval()
    model = model.to(device)
    return model, artifact

"""Paper-aligned PIVOT inference hooks for LLaVA-1.5 (Eqs. 4–8)."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_mip_neurons(neuron_map_file: str) -> dict[int, dict[str, list[float]]]:
    with open(neuron_map_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    mip = data.get("mip_neurons", data.get("SENSORS", data))
    return {int(layer): values for layer, values in mip.items()}


def mip_map_sha256(groups) -> str:
    map_data = {
        int(layer): {"indices": data["indices"], "scores": [float(s) for s in data["scores"]]}
        for layer, data in groups.items()
    }
    encoded = json.dumps(map_data, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_group(layers, groups, name) -> None:
    if not groups:
        raise ValueError(f"The {name} neuron map is empty.")
    for layer_idx, data in groups.items():
        if not 0 <= layer_idx < len(layers):
            raise ValueError(f"{name} layer {layer_idx} is outside the model depth.")
        width = layers[layer_idx].mlp.down_proj.in_features
        indices, scores = data["indices"], data["scores"]
        if (not indices or len(indices) != len(scores)
                or any(type(i) is not int or not 0 <= i < width for i in indices)
                or len(indices) != len(set(indices))
                or any(not math.isfinite(float(s)) for s in scores)):
            raise ValueError(f"Invalid indices or scores for {name} layer {layer_idx}.")
        if name == "MIP" and not any(float(s) != 0 for s in scores):
            raise ValueError(f"MIP layer {layer_idx} has zero L1 score norm.")


def add_to_neurons(h, indices, delta):
    updated = h.clone()
    shifted = h.index_select(-1, indices).float() + delta
    updated.index_copy_(-1, indices, shifted.to(h.dtype))
    return updated


class MIPSaliencyProbe:
    """Share the same MIP reinforcement and saliency hooks with calibration."""

    def __init__(self, model: Any, mip_neurons: dict[int, dict[str, list[float]]],
                 *, enable_mip_feedback: bool = True, allow_decode: bool = False) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.mip_neurons = dict(sorted(mip_neurons.items()))
        self.layers = get_layers(model)
        validate_group(self.layers, self.mip_neurons, "MIP")
        self.enable_mip_feedback = enable_mip_feedback
        self.allow_decode = allow_decode
        self.hooks = []
        self.phi_harm = 0.0
        self.layer_rhos = {}

    def attach(self) -> None:
        self.close()
        self.phi_harm = 0.0
        self.layer_rhos = {}

        try:
            for layer_idx in self.mip_neurons:
                down_proj = self.layers[layer_idx].mlp.down_proj
                self.hooks.append(down_proj.register_forward_pre_hook(self._make_hook(layer_idx)))
        except BaseException:
            self.close()
            raise

    def _make_hook(self, layer_idx):
        torch = self.torch
        data = self.mip_neurons[layer_idx]
        parameters = None

        def pre_hook(_module, inputs):
            nonlocal parameters
            h = inputs[0]
            is_prefill = layer_idx not in self.layer_rhos
            if h.ndim != 3 or h.shape[0] != 1:
                raise RuntimeError(f"MIP layer {layer_idx}: expected a single sample.")
            if not is_prefill and (not self.allow_decode or h.shape[1] != 1):
                raise RuntimeError(f"MIP layer {layer_idx}: expected one prefill for one sample.")
            previous_peak = self.layer_rhos.get(layer_idx - 1, 0.0)
            reinforce = self.enable_mip_feedback and previous_peak != 0.0
            if not is_prefill and not reinforce:
                return None
            if parameters is None or parameters[0].device != h.device:
                indices = torch.tensor(data["indices"], device=h.device, dtype=torch.long)
                scores = torch.tensor(data["scores"], device=h.device, dtype=torch.float32)
                weights = scores / scores.abs().sum()
                parameters = (indices, weights)
            indices, weights = parameters
            # Measure incoming activations before applying reinforcement at this layer.
            if is_prefill:
                observed = h.float()
                numerator = (observed.index_select(-1, indices) * weights).sum(dim=-1)
                saliency = numerator / (observed.norm(p=2, dim=-1) + 1e-6)
                if not torch.isfinite(saliency).all().item():
                    raise ValueError(f"MIP layer {layer_idx}: non-finite token saliency.")
                self.layer_rhos[layer_idx] = saliency.max().item()
                if len(self.layer_rhos) == len(self.mip_neurons):
                    self.finish()

            if reinforce:
                h = add_to_neurons(h, indices, previous_peak * weights)
            return (h, *inputs[1:])

        return pre_hook

    def finish(self) -> None:
        missing = self.mip_neurons.keys() - self.layer_rhos.keys()
        if missing:
            raise RuntimeError(f"No prefill saliency captured for MIP layers {sorted(missing)}.")
        self.phi_harm = math.fsum(self.layer_rhos.values())

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model"):
        return model.model.model.layers
    return model.layers


def load_pivot_config(neuron_map_file: str, threshold_file: str, model_key: str,
                      *, t_neuron_map_file: str | None = None) -> dict[str, Any]:
    mip_neurons = load_mip_neurons(neuron_map_file)
    turning_file = t_neuron_map_file or neuron_map_file
    turning = json.loads(Path(turning_file).read_text(encoding="utf-8"))
    turning = turning.get("t_neurons", turning) if t_neuron_map_file else turning["t_neurons"]
    report = json.loads(Path(threshold_file).read_text(encoding="utf-8"))
    if "threshold" in report:
        threshold = report["threshold"]
        expected_hash = report.get("mip_map_sha256")
        if expected_hash is not None and expected_hash != mip_map_sha256(mip_neurons):
            raise ValueError("Calibration threshold was computed with a different MIP neuron map.")
    else:
        threshold = report[model_key]["mip_threshold"]
    return {
        "mip_neurons": mip_neurons,
        "t_neurons": {int(layer): data for layer, data in turning.items()},
        "mip_threshold": float(threshold),
    }


class PIVOTController(MIPSaliencyProbe):
    """Add the paper's prefill-only Turning Intervention to MIP sensing."""

    def __init__(self, model: Any, mip_neurons: dict[int, dict[str, list[float]]],
                 t_neurons: dict[int, dict[str, list[float]]], mip_threshold: float,
                 *, enable_mip_feedback: bool = True, enable_intervention: bool = True) -> None:
        super().__init__(model, mip_neurons, enable_mip_feedback=enable_mip_feedback, allow_decode=True)
        self.t_neurons = dict(sorted(t_neurons.items()))
        if self.t_neurons or enable_intervention:
            validate_group(self.layers, self.t_neurons, "T")
        if any(layer <= max(self.mip_neurons) for layer in self.t_neurons):
            raise ValueError("All T layers must follow the last MIP layer.")
        if not math.isfinite(mip_threshold):
            raise ValueError("The MIP threshold must be finite.")
        self.mip_threshold = mip_threshold
        self.enable_intervention = enable_intervention
        self.prefilled_t_layers: set[int] = set()

    @classmethod
    def from_files(cls, model: Any, neuron_map_file: str, threshold_file: str,
                   model_key: str = "llava-v1.5-7b", *, t_neuron_map_file: str | None = None,
                   **kwargs: Any) -> "PIVOTController":
        config = load_pivot_config(neuron_map_file, threshold_file, model_key,
                                   t_neuron_map_file=t_neuron_map_file)
        return cls(model=model, **config, **kwargs)

    def attach(self) -> None:
        self.prefilled_t_layers.clear()
        super().attach()
        if self.enable_intervention:
            try:
                for layer_idx in self.t_neurons:
                    down_proj = self.layers[layer_idx].mlp.down_proj
                    hook = down_proj.register_forward_pre_hook(self._make_turning_hook(layer_idx))
                    self.hooks.append(hook)
            except BaseException:
                self.close()
                raise

    def _make_turning_hook(self, layer_idx):
        torch = self.torch
        data = self.t_neurons[layer_idx]

        def pre_hook(_module, inputs):
            if layer_idx in self.prefilled_t_layers:
                return None
            self.prefilled_t_layers.add(layer_idx)
            self.finish()
            delta_phi = max(0.0, self.phi_harm - self.mip_threshold)
            if delta_phi == 0.0:
                return None
            h = inputs[0]
            indices = torch.tensor(data["indices"], device=h.device, dtype=torch.long)
            scores = torch.tensor(data["scores"], device=h.device, dtype=torch.float32)
            weights = torch.nn.functional.normalize(scores, p=1, dim=0)
            boost = 4.2 * math.tanh(150 * delta_phi) * layer_idx * weights
            return (add_to_neurons(h, indices, boost), *inputs[1:])

        return pre_hook


def cleanup_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_with_pivot(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    pivot: PIVOTController,
    images: Any | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
    max_new_tokens: int = 512,
    decode_mode: str = "full",
) -> tuple[str, float]:
    import torch

    if model is not pivot.model:
        raise ValueError("The PIVOT controller must be attached to the model being generated from.")
    try:
        pivot.attach()
        with torch.inference_mode():
            if images is None:
                output_ids = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
        pivot.finish()
        if decode_mode == "slice":
            decode_ids = output_ids[0, input_ids.shape[1] :]
        elif decode_mode == "full":
            decode_ids = output_ids[0]
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")
        text = tokenizer.decode(decode_ids, skip_special_tokens=True).strip()
        return text, pivot.phi_harm
    finally:
        pivot.close()
        cleanup_cuda()


def generate_plain(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    images: Any | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
    max_new_tokens: int = 512,
    decode_mode: str = "full",
) -> str:
    import torch

    with torch.inference_mode():
        if images is None:
            output_ids = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        else:
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
    if decode_mode == "slice":
        decode_ids = output_ids[0, input_ids.shape[1] :]
    elif decode_mode == "full":
        decode_ids = output_ids[0]
    else:
        raise ValueError(f"Unknown decode_mode: {decode_mode}")
    return tokenizer.decode(decode_ids, skip_special_tokens=True).strip()

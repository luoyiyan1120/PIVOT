"""Locate MIP neurons with contrastive probing on LLaVA.

This implements the MIP localization step used in the paper:

1. Estimate benign activation statistics from TextVQA.
2. Compare harmful prompts with their context-legitimized rewrites.
3. Score MLP neurons by robust activation contrast normalized by benign RMS.
4. Merge top neurons from multiple harmful probing datasets.

The script saves the resulting neuron map to a user-specified local file.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_END_TOKEN = None
DEFAULT_IM_START_TOKEN = None
IMAGE_TOKEN_INDEX = None
Image = None
conv_templates = None
get_model_name_from_path = None
load_pretrained_model = None
process_images = None
requests = None
tokenizer_image_token = None
torch = None
tqdm = None


def load_runtime_dependencies() -> None:
    """Import heavy LLaVA dependencies only when localization is requested."""
    global DEFAULT_IMAGE_TOKEN
    global DEFAULT_IM_END_TOKEN
    global DEFAULT_IM_START_TOKEN
    global IMAGE_TOKEN_INDEX
    global Image
    global conv_templates
    global get_model_name_from_path
    global load_pretrained_model
    global process_images
    global requests
    global tokenizer_image_token
    global torch
    global tqdm

    import requests as requests_module
    import torch as torch_module
    from PIL import Image as image_module
    from tqdm import tqdm as tqdm_module

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN as llava_default_image_token,
        DEFAULT_IM_END_TOKEN as llava_default_im_end_token,
        DEFAULT_IM_START_TOKEN as llava_default_im_start_token,
        IMAGE_TOKEN_INDEX as llava_image_token_index,
    )
    from llava.conversation import conv_templates as llava_conv_templates
    from llava.mm_utils import (
        get_model_name_from_path as llava_get_model_name_from_path,
        process_images as llava_process_images,
        tokenizer_image_token as llava_tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model as llava_load_pretrained_model

    DEFAULT_IMAGE_TOKEN = llava_default_image_token
    DEFAULT_IM_END_TOKEN = llava_default_im_end_token
    DEFAULT_IM_START_TOKEN = llava_default_im_start_token
    IMAGE_TOKEN_INDEX = llava_image_token_index
    Image = image_module
    conv_templates = llava_conv_templates
    get_model_name_from_path = llava_get_model_name_from_path
    load_pretrained_model = llava_load_pretrained_model
    process_images = llava_process_images
    requests = requests_module
    tokenizer_image_token = llava_tokenizer_image_token
    torch = torch_module
    tqdm = tqdm_module


def disable_llava_torch_init() -> None:
    from llava.utils import disable_torch_init

    disable_torch_init()


class NeuronTracer:
    """Capture MLP intermediate activations before down projection."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.activations: dict[int, Any] = {}
        self.hooks = []

    def register_hooks(self, target_layers: list[int]) -> None:
        self.activations = {}
        self.hooks = []
        layers = get_layers(self.model)

        def get_activation(layer_idx: int):
            def pre_hook(_module: Any, input_tuple: Any) -> None:
                intermediate_state = input_tuple[0]
                self.activations[layer_idx] = intermediate_state[0, -1, :].detach().cpu()

            return pre_hook

        for layer_idx in target_layers:
            module_to_hook = layers[layer_idx].mlp.down_proj
            self.hooks.append(module_to_hook.register_forward_pre_hook(get_activation(layer_idx)))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


def get_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model"):
        return model.model.model.layers
    return model.layers


def parse_layers(layer_spec: str) -> list[int]:
    layers: list[int] = []
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in handle if line.strip()]
        else:
            data = json.load(handle)
    if isinstance(data, dict):
        return list(data.values())
    return data


def load_image(image_file: str, base_folder: str = "") -> Any | None:
    try:
        if image_file.startswith(("http://", "https://")):
            response = requests.get(image_file, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        image_path = Path(image_file)
        if base_folder and not image_path.is_absolute():
            image_path = Path(base_folder) / image_path
        if not image_path.exists():
            return None
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def prepare_input(
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    prompt_text: str,
    image: Any,
    conv_mode: str,
) -> tuple[Any, Any]:
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates[conv_mode].copy()
    if model.config.mm_use_im_start_end:
        user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    return input_ids.to(model.device), image_tensor


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_conv_mode(model_name: str) -> str:
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    if "v1" in model_name.lower():
        return "llava_v1"
    return "llava_v0"


def forward_and_capture(
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    tracer: NeuronTracer,
    prompt_text: str,
    image: Any,
    conv_mode: str,
) -> dict[int, Any]:
    input_ids, image_tensor = prepare_input(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        prompt_text=prompt_text,
        image=image,
        conv_mode=conv_mode,
    )
    with torch.inference_mode():
        model(input_ids, images=image_tensor)
    return {layer: activation.float().clone() for layer, activation in tracer.activations.items()}


def ensure_stat_buffers(buffers: dict[int, Any], layers: list[int], activations: dict[int, Any]) -> None:
    for layer in layers:
        if layer not in buffers:
            buffers[layer] = torch.zeros_like(activations[layer], dtype=torch.float32, device="cpu")


def main() -> None:
    args = parse_args()
    if len(args.top_k_list) != len(args.malicious_datasets):
        raise ValueError("--top-k-list must have the same length as --malicious-datasets.")

    load_runtime_dependencies()
    disable_llava_torch_init()

    target_layers = parse_layers(args.target_layers)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
        use_flash_attn=False,
    )
    conv_mode = args.conv_mode or infer_conv_mode(model_name)

    tracer = NeuronTracer(model)
    tracer.register_hooks(target_layers)

    benign_data = load_json_or_jsonl(args.benign_dataset)
    if args.benign_limit > 0:
        benign_data = benign_data[: args.benign_limit]

    benign_count = 0
    sum_x: dict[int, Any] = {}
    sum_x2: dict[int, Any] = {}

    for item in tqdm(benign_data, desc="Benign TextVQA passes"):
        image = load_image(str(item.get("image", "")), args.benign_image_folder)
        question = item.get("question", "")
        if image is None or not question:
            continue
        try:
            activations = forward_and_capture(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                tracer=tracer,
                prompt_text=question,
                image=image,
                conv_mode=conv_mode,
            )
            ensure_stat_buffers(sum_x, target_layers, activations)
            ensure_stat_buffers(sum_x2, target_layers, activations)
            for layer in target_layers:
                sum_x[layer] += activations[layer]
                sum_x2[layer] += activations[layer] ** 2
            benign_count += 1
        except Exception:
            continue
        finally:
            cleanup()

    if benign_count == 0:
        raise ValueError("No benign passes completed successfully. Check TextVQA image paths.")

    benign_mu = {}
    benign_std = {}
    benign_rms = {}
    for layer in target_layers:
        mean_x = sum_x[layer] / benign_count
        mean_x2 = sum_x2[layer] / benign_count
        variance = mean_x2 - mean_x**2
        benign_mu[layer] = mean_x
        benign_std[layer] = torch.sqrt(torch.clamp(variance, min=0.0))
        benign_rms[layer] = torch.sqrt(torch.clamp(mean_x2, min=0.0))

    merged_neurons: dict[tuple[int, int], float] = defaultdict(float)
    for dataset_idx, mal_path in enumerate(args.malicious_datasets):
        top_k = args.top_k_list[dataset_idx]
        mal_data = load_json_or_jsonl(mal_path)
        if args.malicious_limit > 0:
            mal_data = mal_data[: args.malicious_limit]

        mal_count = 0.0
        sum_robust_diff: dict[int, Any] = {}
        focus_start_idx = max(0, len(mal_data) - args.focus_tail_count)

        for idx, item in enumerate(tqdm(mal_data, desc=f"Malicious passes {dataset_idx + 1}")):
            image_path = item.get("image") or item.get("image_SD_TYPO", "")
            original_prompt = item.get("Original_Question") or item.get("Rephrased Question", "")
            legitimized_prompts = item.get("Legitimized_Prompts_List", [])
            if not image_path or not original_prompt or not legitimized_prompts:
                continue

            image = load_image(str(image_path), args.malicious_image_folder)
            if image is None:
                continue

            current_weight = args.focus_weight if idx >= focus_start_idx else 1.0
            try:
                act_original = forward_and_capture(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    tracer=tracer,
                    prompt_text=original_prompt,
                    image=image,
                    conv_mode=conv_mode,
                )
                active_mask = {}
                for layer in target_layers:
                    threshold = benign_mu[layer] + args.sigma_multiplier * benign_std[layer]
                    active_mask[layer] = (act_original[layer] > threshold).float()

                diff_matrix = {layer: [] for layer in target_layers}
                for legitimized_prompt in legitimized_prompts:
                    act_legitimized = forward_and_capture(
                        model=model,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        tracer=tracer,
                        prompt_text=legitimized_prompt,
                        image=image,
                        conv_mode=conv_mode,
                    )
                    for layer in target_layers:
                        diff = (act_original[layer] - act_legitimized[layer]) * active_mask[layer]
                        diff_matrix[layer].append(diff)

                ensure_stat_buffers(sum_robust_diff, target_layers, act_original)
                for layer in target_layers:
                    stacked = torch.stack(diff_matrix[layer])
                    sorted_diffs, _ = torch.sort(stacked, dim=0)
                    if sorted_diffs.shape[0] > 2:
                        robust_diff = sorted_diffs[1:-1, :].mean(dim=0)
                    else:
                        robust_diff = sorted_diffs.mean(dim=0)
                    sum_robust_diff[layer] += robust_diff * current_weight
                mal_count += current_weight
            except Exception:
                continue
            finally:
                cleanup()

        if mal_count == 0:
            print(f"[warning] No successful malicious passes for {mal_path}; skipping.")
            continue

        dataset_pool = []
        for layer in target_layers:
            expected_diff = sum_robust_diff[layer] / mal_count
            snr_score = expected_diff / (benign_rms[layer] + 1e-6)
            for neuron_idx, score in enumerate(snr_score.tolist()):
                if score > args.min_score:
                    dataset_pool.append({"layer": layer, "neuron": neuron_idx, "score": float(score)})

        dataset_pool.sort(key=lambda item: item["score"], reverse=True)
        for item in dataset_pool[:top_k]:
            key = (item["layer"], item["neuron"])
            merged_neurons[key] = max(merged_neurons[key], item["score"])

    tracer.remove_hooks()

    merged_pool = [
        {"layer": layer, "neuron": neuron_idx, "score": score}
        for (layer, neuron_idx), score in merged_neurons.items()
    ]
    merged_pool.sort(key=lambda item: item["score"], reverse=True)

    final_sensors = defaultdict(lambda: {"indices": [], "scores": []})
    for item in merged_pool:
        final_sensors[item["layer"]]["indices"].append(item["neuron"])
        final_sensors[item["layer"]]["scores"].append(round(item["score"], 6))

    output = {str(layer): values for layer, values in sorted(final_sensors.items())}
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved {len(merged_pool)} merged MIP neurons to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate MIP neurons with contrastive probing.")
    parser.add_argument("--model-path", required=True, help="Path or HF name for the LLaVA checkpoint.")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--benign-dataset", required=True, help="TextVQA JSON/JSONL reference set.")
    parser.add_argument("--benign-image-folder", default="", help="Base folder for TextVQA images.")
    parser.add_argument("--malicious-datasets", nargs="+", required=True, help="Expanded harmful JSON files.")
    parser.add_argument("--malicious-image-folder", default="", help="Base folder for harmful images.")
    parser.add_argument("--output-file", required=True, help="Output MIP neuron map JSON.")
    parser.add_argument("--target-layers", default="10-17", help="Comma/range layer spec, e.g. 10-17 or 10,11,12.")
    parser.add_argument("--top-k-list", type=int, nargs="+", required=True)
    parser.add_argument("--benign-limit", type=int, default=1000)
    parser.add_argument("--malicious-limit", type=int, default=0)
    parser.add_argument("--sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--min-score", type=float, default=1e-4)
    parser.add_argument("--focus-tail-count", type=int, default=85)
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Compute layer-wise cosine similarity for MIP-layer selection on LLaVA.

This script compares each harmful prompt with its context-legitimized rewrites
and writes one row per sample. Each ``layer_*`` value is the arithmetic mean of the
cosine similarities across the rewrites for that sample. With nine rewrites
per query, averaging each layer column gives the mean across all pairs.

The released workflow can provide the final MIP layers directly. This script is
kept to document how the layer-selection evidence was produced.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from io import BytesIO
from statistics import fmean
from pathlib import Path
from typing import Any

DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_END_TOKEN = None
DEFAULT_IM_START_TOKEN = None
F = None
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


class ActivationCapturer:
    """Capture last-token hidden states from every transformer layer."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.hooks = []
        self.activations: dict[int, torch.Tensor] = {}

    def register_hooks(self) -> None:
        self.hooks = []
        self.activations = {}
        layers = get_layers(self.model)

        def get_hook(layer_idx: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden_states = output[0] if isinstance(output, tuple) else output
                self.activations[layer_idx] = hidden_states[:, -1, :].detach().cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(get_hook(layer_idx)))

    def close(self) -> None:
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


def load_runtime_dependencies() -> None:
    """Import heavy LLaVA dependencies only when computation is requested."""
    global DEFAULT_IMAGE_TOKEN
    global DEFAULT_IM_END_TOKEN
    global DEFAULT_IM_START_TOKEN
    global F
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
    import torch.nn.functional as functional_module
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
    F = functional_module
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


def load_image(image_path: str) -> Any | None:
    if image_path.startswith(("http://", "https://")):
        try:
            response = requests.get(image_path, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            print(f"[warning] Failed to download image {image_path}: {exc}")
            return None

    try:
        return Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[warning] Image not found: {image_path}")
        return None


def cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def infer_conv_mode(model_name: str) -> str:
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    if "v1" in model_name.lower():
        return "llava_v1"
    return "llava_v0"


def build_prompt(model: Any, tokenizer: Any, conv_mode: str, prompt_text: str) -> torch.Tensor:
    conv = conv_templates[conv_mode].copy()
    if model.config.mm_use_im_start_end:
        user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)


def get_activations_for_prompt(
    *,
    capturer: ActivationCapturer,
    model: Any,
    tokenizer: Any,
    conv_mode: str,
    prompt_text: str,
    image_tensor: Any,
) -> dict[int, torch.Tensor]:
    input_ids = build_prompt(model, tokenizer, conv_mode, prompt_text).to(model.device)
    capturer.register_hooks()
    with torch.inference_mode():
        model(input_ids, images=image_tensor)
    activations = dict(capturer.activations)
    capturer.close()
    return activations


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    disable_llava_torch_init()

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
    capturer = ActivationCapturer(model)

    with open(args.input_file, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    data_list = list(dataset.values())
    if args.num_samples > 0:
        data_list = data_list[: args.num_samples]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    num_layers = len(get_layers(model))
    layer_headers = [f"layer_{layer_idx}" for layer_idx in range(num_layers)]

    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "category", *layer_headers])
        writer.writeheader()

        for idx, item in enumerate(tqdm(data_list)):
            image = load_image(str(item.get("image", "")))
            unsafe_prompt = item.get("Original_Question", "")
            safe_prompts = item.get("Legitimized_Prompts_List", [])
            if image is None or not unsafe_prompt or not safe_prompts:
                continue

            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            acts_unsafe = get_activations_for_prompt(
                capturer=capturer,
                model=model,
                tokenizer=tokenizer,
                conv_mode=conv_mode,
                prompt_text=unsafe_prompt,
                image_tensor=image_tensor,
            )
            layer_sims = {layer_idx: [] for layer_idx in range(num_layers)}

            for safe_prompt in safe_prompts:
                acts_safe = get_activations_for_prompt(
                    capturer=capturer,
                    model=model,
                    tokenizer=tokenizer,
                    conv_mode=conv_mode,
                    prompt_text=safe_prompt,
                    image_tensor=image_tensor,
                )
                for layer_idx in range(num_layers):
                    sim = F.cosine_similarity(
                        acts_unsafe[layer_idx].to(torch.float32),
                        acts_safe[layer_idx].to(torch.float32),
                        dim=-1,
                    ).item()
                    layer_sims[layer_idx].append(sim)

            row = {"image_id": idx, "category": item.get("category", "unknown")}
            for layer_idx in range(num_layers):
                row[f"layer_{layer_idx}"] = fmean(layer_sims[layer_idx])
            writer.writerow(row)
            csv_file.flush()
            cleanup()

    print(f"Saved layer-wise mean cosine similarities to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute layer-wise harmful/legitimized similarity.")
    parser.add_argument("--model-path", required=True, help="Path or HF name for the LLaVA checkpoint.")
    parser.add_argument("--model-base", default=None, help="Optional base model path.")
    parser.add_argument("--input-file", required=True, help="Expanded contrastive-pair JSON file.")
    parser.add_argument("--output-file", required=True, help="Output CSV path.")
    parser.add_argument("--num-samples", type=int, default=-1, help="Limit samples for a quick run.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--conv-mode", default=None, help="Override LLaVA conversation template.")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

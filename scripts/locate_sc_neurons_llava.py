"""Locate safety-critical neurons with Integrated Gradients on LLaVA.

This script reproduces SC-Neuron localization:

1. Load harmful prompts from VLSafe-style JSONL using the ``query`` field.
2. Load benign prompts from Dolly-style JSONL using the ``question`` field.
3. Attribute the refusal logit margin ``z(Sorry) - z(Sure)`` to MLP
   intermediate neurons with Captum LayerConductance.
4. Score neurons as ``ReLU(harmful_attr) - lambda * ReLU(benign_attr)``.

The resulting SC candidates can be classified with the logit-lens script.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

LayerConductance = None
conv_templates = None
get_model_name_from_path = None
load_pretrained_model = None
torch = None
tqdm = None


def load_runtime_dependencies() -> None:
    """Import heavy dependencies only when localization is requested."""
    global LayerConductance
    global conv_templates
    global get_model_name_from_path
    global load_pretrained_model
    global torch
    global tqdm

    import torch as torch_module
    from captum.attr import LayerConductance as captum_layer_conductance
    from tqdm import tqdm as tqdm_module

    from llava.conversation import conv_templates as llava_conv_templates
    from llava.mm_utils import get_model_name_from_path as llava_get_model_name_from_path
    from llava.model.builder import load_pretrained_model as llava_load_pretrained_model

    LayerConductance = captum_layer_conductance
    conv_templates = llava_conv_templates
    get_model_name_from_path = llava_get_model_name_from_path
    load_pretrained_model = llava_load_pretrained_model
    torch = torch_module
    tqdm = tqdm_module


def disable_llava_torch_init() -> None:
    from llava.utils import disable_torch_init

    disable_torch_init()


def load_prompts_from_jsonl(file_path: str, text_key: str, limit: int = 0) -> list[str]:
    prompts = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get(text_key, "")
            if prompt:
                prompts.append(str(prompt))
            if limit > 0 and len(prompts) >= limit:
                break
    return prompts


def infer_conv_mode(model_name_or_path: str) -> str:
    lower_name = model_name_or_path.lower()
    if "llama-2" in lower_name:
        return "llava_llama_2"
    if "v1" in lower_name:
        return "llava_v1"
    if "mpt" in lower_name:
        return "mpt"
    return "llava_v0"


def apply_llava_template(prompt: str, conv_mode: str) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def get_target_token_ids(tokenizer: Any, positive_anchor: str, negative_anchor: str) -> tuple[int, int]:
    positive_id = tokenizer.encode(positive_anchor, add_special_tokens=False)[0]
    negative_id = tokenizer.encode(negative_anchor, add_special_tokens=False)[0]
    print(
        "Target margin: "
        f"logit('{positive_anchor}'={positive_id}) - logit('{negative_anchor}'={negative_id})"
    )
    return positive_id, negative_id


def get_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model"):
        return model.model.model.layers
    return model.layers


def parse_layers(layer_spec: str) -> list[int]:
    layers = []
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


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_layer_ig(
    *,
    model: Any,
    tokenizer: Any,
    layer_module: Any,
    prompts: list[str],
    conv_mode: str,
    positive_id: int,
    negative_id: int,
    ig_steps: int,
    device: str,
) -> Any | None:
    def forward_func(inputs_embeds: Any) -> Any:
        outputs = model(inputs_embeds=inputs_embeds)
        last_token_logits = outputs.logits[:, -1, :]
        return last_token_logits[:, positive_id] - last_token_logits[:, negative_id]

    conductance = LayerConductance(forward_func, layer_module)
    total_attribution = None
    count = 0

    for prompt in tqdm(prompts, desc="Running IG", leave=False):
        try:
            final_prompt = apply_llava_template(prompt, conv_mode)
            inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
            input_embeds = model.get_input_embeddings()(inputs.input_ids)

            attribution = conductance.attribute(
                inputs=input_embeds,
                n_steps=ig_steps,
                attribute_to_layer_input=True,
                internal_batch_size=1,
            )
            if isinstance(attribution, tuple):
                attribution = attribution[0]

            token_attr = attribution[0, -1, :].detach().cpu()
            if total_attribution is None:
                total_attribution = torch.zeros_like(token_attr)
            total_attribution += token_attr
            count += 1

            del attribution, inputs, input_embeds, token_attr
        except Exception as exc:
            print(f"[warning] Skipping prompt due to error: {exc}")
            continue

    del conductance
    cleanup()

    if count == 0:
        return None
    return total_attribution / count


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    disable_llava_torch_init()

    harmful_prompts = load_prompts_from_jsonl(args.harmful_jsonl, args.harmful_key, args.harmful_limit)
    benign_prompts = load_prompts_from_jsonl(args.benign_jsonl, args.benign_key, args.benign_limit)
    if not harmful_prompts or not benign_prompts:
        raise ValueError("Harmful or benign prompt set is empty.")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _image_processor, _context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map=args.device_map,
        device=args.device,
        use_flash_attn=False,
    )
    model.eval()

    conv_mode = args.conv_mode or infer_conv_mode(model_name)
    positive_id, negative_id = get_target_token_ids(tokenizer, args.positive_anchor, args.negative_anchor)
    layers = get_layers(model)

    sc_neurons = {}
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else None
    if checkpoint_file:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    for layer_idx in parse_layers(args.layers):
        print(f"Analyzing layer {layer_idx}")
        target_layer = layers[layer_idx].mlp.down_proj

        harmful_attr = compute_layer_ig(
            model=model,
            tokenizer=tokenizer,
            layer_module=target_layer,
            prompts=harmful_prompts,
            conv_mode=conv_mode,
            positive_id=positive_id,
            negative_id=negative_id,
            ig_steps=args.ig_steps,
            device=args.device,
        )
        benign_attr = compute_layer_ig(
            model=model,
            tokenizer=tokenizer,
            layer_module=target_layer,
            prompts=benign_prompts,
            conv_mode=conv_mode,
            positive_id=positive_id,
            negative_id=negative_id,
            ig_steps=args.ig_steps,
            device=args.device,
        )
        if harmful_attr is None or benign_attr is None:
            print(f"[warning] Skipping layer {layer_idx}: no attribution result.")
            continue

        score = torch.relu(harmful_attr.float()) - args.lambda_value * torch.relu(benign_attr.float())
        top_values, top_indices = torch.topk(score, k=args.top_k)
        sc_neurons[str(layer_idx)] = {
            "indices": top_indices.cpu().numpy().tolist(),
            "scores": top_values.detach().cpu().numpy().tolist(),
        }

        if checkpoint_file:
            with open(checkpoint_file, "w", encoding="utf-8") as handle:
                json.dump(sc_neurons, handle, indent=2)
        cleanup()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(sc_neurons, handle, indent=2)
    print(f"Saved SC neuron candidates to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate SC neurons with Integrated Gradients.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--harmful-jsonl", required=True, help="VLSafe-style JSONL with harmful prompts.")
    parser.add_argument("--benign-jsonl", required=True, help="Dolly-style JSONL with benign prompts.")
    parser.add_argument("--harmful-key", default="query")
    parser.add_argument("--benign-key", default="question")
    parser.add_argument("--harmful-limit", type=int, default=240)
    parser.add_argument("--benign-limit", type=int, default=240)
    parser.add_argument("--layers", default="10-27")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--ig-steps", type=int, default=20)
    parser.add_argument("--positive-anchor", default="Sorry")
    parser.add_argument("--negative-anchor", default="Sure")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--conv-mode", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

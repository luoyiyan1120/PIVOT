"""Split SC-Neuron candidates into T and R neurons with logits lens.

Given SC candidates from Integrated Gradients, this script projects each MLP
neuron's down-projection vector through the final norm and language-model head.
Candidates whose top token is a discourse pivot are saved as T-Neurons, while
candidates whose top token is a refusal token are saved as R-Neurons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

get_model_name_from_path = None
load_pretrained_model = None
torch = None
tqdm = None


DEFAULT_T_KEYWORDS = [
    "but",
    "otherwise",
    "instead",
    "unfortunately",
    "however",
    "although",
    "But",
    "Otherwise",
    "Instead",
    "Unfortunately",
    "However",
    "Although",
]


DEFAULT_R_KEYWORDS = [
    "sorry",
    "Sorry",
    "SORRY",
    "apologize",
    "Apologize",
    "apologies",
    "Apologies",
    "cannot",
    "Cannot",
    "unable",
    "Unable",
    "decline",
    "Decline",
    "refuse",
    "Refuse",
    "never",
    "Never",
    "prevent",
    "Prevent",
]


def load_runtime_dependencies() -> None:
    global get_model_name_from_path
    global load_pretrained_model
    global torch
    global tqdm

    import torch as torch_module
    from tqdm import tqdm as tqdm_module

    from llava.mm_utils import get_model_name_from_path as llava_get_model_name_from_path
    from llava.model.builder import load_pretrained_model as llava_load_pretrained_model

    get_model_name_from_path = llava_get_model_name_from_path
    load_pretrained_model = llava_load_pretrained_model
    torch = torch_module
    tqdm = tqdm_module


def disable_llava_torch_init() -> None:
    from llava.utils import disable_torch_init

    disable_torch_init()


def get_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model"):
        return model.model.model.layers
    return model.layers


def load_keywords(path: str | None, defaults: list[str]) -> list[str]:
    if not path:
        return defaults
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_token(token: str) -> str:
    return token.strip()


def split_candidates(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    load_runtime_dependencies()
    disable_llava_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _image_processor, _context_len = load_pretrained_model(
        args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model.eval()

    t_keywords = set(load_keywords(args.t_keywords_file, DEFAULT_T_KEYWORDS))
    r_keywords = set(load_keywords(args.r_keywords_file, DEFAULT_R_KEYWORDS))

    with open(args.input_neurons, "r", encoding="utf-8") as handle:
        sc_candidates = json.load(handle)

    layers = get_layers(model)
    lm_head = model.lm_head
    final_norm = model.model.norm

    t_neurons: dict[str, Any] = {}
    r_neurons: dict[str, Any] = {}

    for layer_idx_str, data in sc_candidates.items():
        layer_idx = int(layer_idx_str)
        indices = data["indices"][: args.max_candidates_per_layer]
        scores = data["scores"][: args.max_candidates_per_layer]
        down_proj_weight = layers[layer_idx].mlp.down_proj.weight.data

        layer_t = {"indices": [], "scores": [], "top_tokens": []}
        layer_r = {"indices": [], "scores": [], "top_tokens": []}

        for item_idx, neuron_idx in enumerate(tqdm(indices, desc=f"Layer {layer_idx}")):
            neuron_vector = down_proj_weight[:, neuron_idx].unsqueeze(0)

            with torch.no_grad():
                hidden_states = final_norm(neuron_vector.to(torch.float16))
                logits = lm_head(hidden_states)
                probs = torch.softmax(logits, dim=-1)
                top_k = torch.topk(probs, k=args.top_tokens)

            top_token_ids = top_k.indices[0].tolist()
            top_tokens = [clean_token(tokenizer.decode([token_id])) for token_id in top_token_ids]
            match_tokens = top_tokens[: args.match_top_tokens]

            if any(token in t_keywords for token in match_tokens):
                layer_t["indices"].append(neuron_idx)
                layer_t["scores"].append(scores[item_idx])
                layer_t["top_tokens"].append(top_tokens)

            if any(token in r_keywords for token in match_tokens):
                layer_r["indices"].append(neuron_idx)
                layer_r["scores"].append(scores[item_idx])
                layer_r["top_tokens"].append(top_tokens)

        if layer_t["indices"]:
            t_neurons[layer_idx_str] = layer_t
        if layer_r["indices"]:
            r_neurons[layer_idx_str] = layer_r

    return t_neurons, r_neurons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split SC candidates into T and R neurons.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--input-neurons", required=True, help="SC candidate JSON from IG localization.")
    parser.add_argument("--output-t-neurons", required=True)
    parser.add_argument("--output-r-neurons", required=True)
    parser.add_argument("--max-candidates-per-layer", type=int, default=500)
    parser.add_argument("--top-tokens", type=int, default=10)
    parser.add_argument("--match-top-tokens", type=int, default=1)
    parser.add_argument("--t-keywords-file", default="")
    parser.add_argument("--r-keywords-file", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_neurons, r_neurons = split_candidates(args)

    output_t = Path(args.output_t_neurons)
    output_r = Path(args.output_r_neurons)
    output_t.parent.mkdir(parents=True, exist_ok=True)
    output_r.parent.mkdir(parents=True, exist_ok=True)

    with open(output_t, "w", encoding="utf-8") as handle:
        json.dump(t_neurons, handle, indent=2)
    with open(output_r, "w", encoding="utf-8") as handle:
        json.dump(r_neurons, handle, indent=2)

    t_count = sum(len(value["indices"]) for value in t_neurons.values())
    r_count = sum(len(value["indices"]) for value in r_neurons.values())
    print(f"Saved {t_count} T-Neurons to {output_t}")
    print(f"Saved {r_count} R-Neurons to {output_r}")


if __name__ == "__main__":
    main()

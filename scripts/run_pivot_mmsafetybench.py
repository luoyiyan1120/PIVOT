"""Generate LLaVA-1.5 responses to original MM-SafetyBench image-query pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pivot.llava_pivot import PIVOTController, generate_plain, generate_with_pivot


TARGET_CATEGORIES = {
    "Illegal Activity": ("01-Illegal_Activity.json", "01-Illegal_Activity"),
    "Hate Speech": ("02-HateSpeech.json", "02-HateSpeech"),
    "Malware Generation": ("03-Malware_Generation.json", "03-Malware_Generation"),
    "Physical Harm": ("04-Physical_Harm.json", "04-Physical_Harm"),
    "Fraud": ("06-Fraud.json", "06-Fraud"),
    "Pornography": ("07-Sex.json", "07-Sex"),
}

LEGACY_CATEGORY_ALIASES = {
    "01-Illegal_Activity": "01-Illegal_Activitiy"
}


def load_runtime_dependencies() -> dict[str, Any]:
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    return {
        "torch": torch,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "get_model_name_from_path": get_model_name_from_path,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


def load_image(image_path: str) -> Image.Image | None:
    if not image_path:
        return None
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as image:
            return image.convert("RGB")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_mmsafetybench_from_official_json(args: argparse.Namespace) -> list[dict[str, str]]:
    dataset = []
    for category, (json_name, img_dir_name) in TARGET_CATEGORIES.items():
        json_path = Path(args.json_base_dir) / json_name
        if not json_path.exists() and img_dir_name in LEGACY_CATEGORY_ALIASES:
            json_path = Path(args.json_base_dir) / f"{LEGACY_CATEGORY_ALIASES[img_dir_name]}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        for q_id, item_data in data.items():
            for image_type in args.image_types:
                query_key = "Rephrased Question(SD)" if image_type == "SD" else "Rephrased Question"
                query = item_data.get(query_key, "")
                if not query:
                    continue
                image_dir = Path(args.img_base_dir) / img_dir_name / image_type
                if not image_dir.exists() and img_dir_name in LEGACY_CATEGORY_ALIASES:
                    image_dir = Path(args.img_base_dir) / LEGACY_CATEGORY_ALIASES[img_dir_name] / image_type
                if not image_dir.exists():
                    image_dir = Path(args.img_base_dir) / img_dir_name
                if not image_dir.exists() and img_dir_name in LEGACY_CATEGORY_ALIASES:
                    image_dir = Path(args.img_base_dir) / LEGACY_CATEGORY_ALIASES[img_dir_name]
                image_path = image_dir / f"{q_id}.jpg"
                if not image_path.exists():
                    png_path = image_path.with_suffix(".png")
                    image_path = png_path if png_path.exists() else image_path
                if not image_path.exists() and not args.keep_missing_images:
                    continue
                dataset.append(
                    {
                        "sample_id": f"{img_dir_name}_{q_id}_{image_type}",
                        "category": category,
                        "image_type": image_type,
                        "image_path": str(image_path),
                        "query": query,
                    }
                )
                if args.num_samples > 0 and len(dataset) >= args.num_samples:
                    return dataset
    return dataset


def load_mmsafetybench_from_jsonl(args: argparse.Namespace) -> list[dict[str, str]]:
    dataset = []
    with open(args.input_jsonl, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            query = item.get("query") or item.get("question")
            image_path = item.get("image") or item.get("image_path")
            if not all(isinstance(value, str) and value.strip() for value in (query, image_path)):
                raise ValueError(f"{args.input_jsonl}:{idx + 1}: expected a query and image path.")
            dataset.append(
                {
                    "sample_id": str(item.get("sample_id", idx)),
                    "category": str(item.get("category", "")),
                    "image_type": str(item.get("image_type", "")),
                    "image_path": image_path,
                    "query": query,
                }
            )
            if args.num_samples > 0 and len(dataset) >= args.num_samples:
                break
    return dataset


def prepare_llava_inputs(runtime: dict[str, Any], model: Any, tokenizer: Any, image_processor: Any, query: str, image: Image.Image):
    # Use the same original benchmark query and LLaVA chat template in every mode.
    conv = runtime["conv_templates"]["llava_v1"].copy()
    if model.config.mm_use_im_start_end:
        prompt = (
            runtime["DEFAULT_IM_START_TOKEN"]
            + runtime["DEFAULT_IMAGE_TOKEN"]
            + runtime["DEFAULT_IM_END_TOKEN"]
            + "\n"
            + query
        )
    else:
        prompt = runtime["DEFAULT_IMAGE_TOKEN"] + "\n" + query
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    input_ids = runtime["tokenizer_image_token"](
        conv.get_prompt(),
        tokenizer,
        runtime["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    image_tensor = runtime["process_images"]([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=model.dtype) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=model.dtype)
    return input_ids, image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIVOT on MM-SafetyBench with LLaVA-1.5.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--neuron-map", help="Local combined MIP/T map or MIP map supplied with --t-neuron-map.")
    parser.add_argument("--t-neuron-map", default=None, help="Separate T map when --neuron-map is MIP-only.")
    parser.add_argument("--thresholds", help="Local threshold JSON from calibration.")
    parser.add_argument("--model-key", default="llava-v1.5-7b")
    parser.add_argument("--input-jsonl", default="", help="Optional simple JSONL with image/query fields.")
    parser.add_argument("--json-base-dir", default="data/MM-SafetyBench/processed_questions")
    parser.add_argument("--img-base-dir", default="data/MM-SafetyBench/imgs")
    parser.add_argument("--image-types", nargs="+", default=["SD_TYPO"])
    parser.add_argument("--output-csv", default="results/pivot_mmsafetybench.csv")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates all loaded samples.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-missing-images", action="store_true")
    parser.add_argument("--mode", choices=["baseline", "sense", "pivot"], default="pivot",
                        help="Generate normally, measure unreinforced saliency, or run the paper's PIVOT.")
    parser.add_argument("--decode-mode", choices=["full", "slice"], default="full")
    args = parser.parse_args()
    if args.num_samples < 0 or args.max_new_tokens <= 0:
        parser.error("--num-samples must be nonnegative and --max-new-tokens must be positive.")
    if args.mode != "baseline":
        if not args.neuron_map or not args.thresholds:
            parser.error(f"{args.mode} mode requires local --neuron-map and --thresholds files.")
        for flag, path in (("--neuron-map", args.neuron_map), ("--thresholds", args.thresholds),
                           ("--t-neuron-map", args.t_neuron_map)):
            if path and not Path(path).is_file():
                parser.error(f"{flag} file not found: {path}")
    return args


def main() -> None:
    args = parse_args()
    dataset = load_mmsafetybench_from_jsonl(args) if args.input_jsonl else load_mmsafetybench_from_official_json(args)
    if not dataset:
        raise ValueError("No samples loaded. Check dataset paths.")
    print(f"Loaded {len(dataset)} samples for {args.mode}.")

    runtime = load_runtime_dependencies()
    runtime["torch"].manual_seed(args.seed)
    runtime["disable_torch_init"]()

    model_name = runtime["get_model_name_from_path"](args.model_path)
    tokenizer, model, image_processor, _context_len = runtime["load_pretrained_model"](
        args.model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device=args.device,
        use_flash_attn=False,
    )
    model.eval()

    pivot = None
    if args.mode != "baseline":
        pivot = PIVOTController.from_files(
            model,
            neuron_map_file=args.neuron_map,
            threshold_file=args.thresholds,
            model_key=args.model_key,
            t_neuron_map_file=args.t_neuron_map,
            enable_mip_feedback=args.mode == "pivot",
            enable_intervention=args.mode == "pivot",
        )

    output_file = Path(args.output_csv)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "category", "image_type", "image_path", "query", "method", "decode_mode", "phi_harm", "response"]

    with open(output_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in tqdm(dataset, desc=f"Running {args.mode}"):
            if not item["query"].strip():
                raise ValueError(f'Missing query for sample {item["sample_id"]}.')
            image = load_image(item["image_path"])
            if image is None:
                raise ValueError(f'Missing image for sample {item["sample_id"]}.')
            with image:
                image_sizes = [image.size]
                input_ids, image_tensor = prepare_llava_inputs(
                    runtime, model, tokenizer, image_processor, item["query"], image,
                )
            if pivot is None:
                response = generate_plain(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    max_new_tokens=args.max_new_tokens,
                    decode_mode=args.decode_mode,
                )
                phi_harm = ""
            else:
                response, phi_harm = generate_with_pivot(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    pivot=pivot,
                    max_new_tokens=args.max_new_tokens,
                    decode_mode=args.decode_mode,
                )
            writer.writerow(
                {
                    **item,
                    "method": args.mode,
                    "decode_mode": args.decode_mode,
                    "phi_harm": phi_harm,
                    "response": response,
                }
            )
    print(f"Saved {len(dataset)} responses to {output_file}")


if __name__ == "__main__":
    main()

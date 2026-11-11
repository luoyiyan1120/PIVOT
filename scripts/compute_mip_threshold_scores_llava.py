"""Calibrate PIVOT's relative-saliency threshold for a LLaVA checkpoint.

One prefill per image-query pair measures saliency before local reinforcement
at each MIP layer, using the same probe as PIVOT inference.
The threshold is selected at a density crossing that maximizes TPR minus FPR.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pivot.llava_pivot import MIPSaliencyProbe, load_mip_neurons, mip_map_sha256

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


def load_image(image_path: str, base_folder: str = "") -> Any:
    if not image_path:
        raise ValueError("Calibration requires the sample's original image.")
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as image:
            return image.convert("RGB")

    path = Path(image_path).expanduser()
    if base_folder and not path.is_absolute():
        path = Path(base_folder).expanduser() / path
    with Image.open(path) as image:
        return image.convert("RGB")


def infer_conv_mode(model_name_or_path: str) -> str:
    lower_name = model_name_or_path.lower()
    if "llama-2" in lower_name:
        return "llava_llama_2"
    if "v1" in lower_name:
        return "llava_v1"
    return "llava_v0"


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_benign_items(path: str) -> list[dict[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            query = item.get("question", "") or item.get("instruction", "")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{path}: missing benign query on line {idx + 1}.")
            items.append(
                {
                    "id": str(item.get("question_id", item.get("id", f"benign_{idx}"))),
                    "category": item.get("category_label", "utility"),
                    "query": query,
                    "image_path": item.get("image") or item.get("image_path", ""),
                }
            )
    return items


def load_malicious_items(paths: list[str]) -> list[dict[str, str]]:
    items = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = list(data.values())
        for item in data:
            query = item.get("Original_Question", "")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{path}: missing Original_Question in item {len(items)}.")
            items.append(
                {
                    "id": str(item.get("question_id", item.get("id", f"malicious_{len(items)}"))),
                    "category": item.get("category", Path(path).name),
                    "query": query,
                    "image_path": item.get("image") or item.get("image_path", ""),
                }
            )
    return items


def process_and_write(
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    probe: MIPSaliencyProbe,
    items: list[dict[str, str]],
    output_path: str,
    conv_mode: str,
    base_folder: str,
) -> list[float]:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    layer_keys = sorted(probe.mip_neurons)
    fieldnames = ["id", "category", "query", *[f"rho_{layer}" for layer in layer_keys], "rho_total"]
    totals = []

    with open(output_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(tqdm(items, desc=f"Processing {output_file.name}")):
            with load_image(item["image_path"], base_folder) as image:
                image_size = image.size
                image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=model.dtype) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=model.dtype)

            conv = conv_templates[conv_mode].copy()
            if model.config.mm_use_im_start_end:
                user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + item["query"]
            else:
                user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + item["query"]
            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).to(model.device)

            try:
                probe.attach()
                with torch.inference_mode():
                    kwargs = dict(input_ids=input_ids, images=image_tensor, use_cache=False)
                    try:
                        model(**kwargs, image_sizes=[image_size])
                    except TypeError as exc:
                        if "unexpected keyword argument 'image_sizes'" not in str(exc):
                            raise
                        model(**kwargs)
                probe.finish()
            finally:
                probe.close()

            row = {
                "id": item.get("id", str(idx)),
                "category": item.get("category", "unknown"),
                "query": item["query"],
                "rho_total": probe.phi_harm,
            }
            for layer in layer_keys:
                row[f"rho_{layer}"] = probe.layer_rhos[layer]
            writer.writerow(row)
            totals.append(probe.phi_harm)
            cleanup()
    return totals


def fit_threshold(benign_scores: list[float], harmful_scores: list[float]):
    """Choose the KDE crossing with the largest estimated TPR minus FPR."""
    import numpy as np
    from scipy.optimize import brentq
    from scipy.stats import gaussian_kde

    benign = np.asarray(benign_scores, dtype=float)
    harmful = np.asarray(harmful_scores, dtype=float)
    for label, scores in (("benign", benign), ("harmful", harmful)):
        if scores.size < 3 or not np.isfinite(scores).all() or np.ptp(scores) == 0:
            raise ValueError(f"KDE needs at least three finite, nonconstant {label} scores.")

    benign_kde = gaussian_kde(benign, bw_method="scott")
    harmful_kde = gaussian_kde(harmful, bw_method="scott")
    padding = 4 * max(benign.std(), harmful.std())
    grid = np.linspace(min(benign.min(), harmful.min()) - padding,
                       max(benign.max(), harmful.max()) + padding, 8193)
    benign_density, harmful_density = benign_kde(grid), harmful_kde(grid)
    if np.allclose(benign_density, harmful_density, rtol=1e-6, atol=1e-10):
        raise ValueError("The two KDEs are indistinguishable.")

    def density_difference(value):
        return float(benign_kde(value)[0] - harmful_kde(value)[0])

    difference = benign_density - harmful_density
    changes = (difference[:-1] < 0) != (difference[1:] < 0)
    crossings = []
    for index in np.flatnonzero(changes):
        root = float(brentq(density_difference, grid[index], grid[index + 1]))
        if not crossings or abs(root - crossings[-1]) > 1e-10:
            crossings.append(root)
    candidates = []
    for threshold in crossings:
        fpr = float(benign_kde.integrate_box_1d(threshold, np.inf))
        tpr = float(harmful_kde.integrate_box_1d(threshold, np.inf))
        candidates.append({"threshold": threshold, "kde_tpr": tpr, "kde_fpr": fpr,
                           "youden_j": tpr - fpr})
    if not candidates:
        raise ValueError("No crossing found between the calibration KDEs.")
    selected = max(candidates, key=lambda item: (item["youden_j"], -item["threshold"]))
    if selected["youden_j"] <= 0:
        raise ValueError("No KDE crossing gives a positive TPR minus FPR for the high-score trigger.")
    threshold = selected["threshold"]
    report = {
        "threshold": threshold,
        "trigger": "rho_total > threshold",
        "selection": "KDE crossing maximizing TPR - FPR (ties select the lower crossing)",
        "kde_bandwidth": "scott",
        "counts": {"benign": len(benign), "harmful": len(harmful)},
        "empirical_rates": {"tpr": float((harmful > threshold).mean()),
                            "fpr": float((benign > threshold).mean())},
        "selected_crossing": selected,
        "crossings": candidates,
    }
    return report, (grid, benign_density, harmful_density)


def save_plot(path: Path, threshold: float, curves) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, benign_density, harmful_density = curves
    fig, ax = plt.subplots(figsize=(6.3, 3.9), layout="constrained")
    ax.plot(grid, benign_density, color="#B33A9B", label="Benign calibration")
    ax.plot(grid, harmful_density, color="#2161CE", label="Harmful calibration")
    ax.axvline(threshold, color="#343434", ls="--", label=f"Threshold = {threshold:.6f}")
    ax.set(xlabel=r"$\Phi_{\mathrm{harm}}$", ylabel="Probability density")
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MIP saliency scores for threshold calibration.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--neuron-map", required=True, help="Local MIP neuron map JSON.")
    parser.add_argument("--benign-dataset", required=True)
    parser.add_argument("--benign-image-folder", default="")
    parser.add_argument("--benign-output", required=True)
    parser.add_argument("--malicious-datasets", nargs="+", required=True)
    parser.add_argument("--malicious-image-folder", default="")
    parser.add_argument("--malicious-output", required=True)
    parser.add_argument("--threshold-output", type=Path, help="Threshold JSON. Defaults beside the harmful CSV.")
    parser.add_argument("--plot-output", type=Path, help="Optional KDE plot path (.pdf or .png).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conv-mode", default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative.")
    return args


def main() -> None:
    args = parse_args()
    benign_items = load_benign_items(args.benign_dataset)
    malicious_items = load_malicious_items(args.malicious_datasets)
    if args.limit > 0:
        benign_items = benign_items[: args.limit]
        malicious_items = malicious_items[: args.limit]
    if min(len(benign_items), len(malicious_items)) < 3:
        raise ValueError("KDE requires at least three samples in each calibration set.")
    print(f"Calibration samples: {len(benign_items)} benign, {len(malicious_items)} harmful")

    harmful_csv = Path(args.malicious_output)
    threshold_output = args.threshold_output or harmful_csv.with_name(f"{harmful_csv.stem}_threshold.json")
    output_paths = [Path(args.benign_output), harmful_csv, threshold_output]
    if args.plot_output:
        output_paths.append(args.plot_output)
    outputs = {path.resolve() for path in output_paths}
    inputs = {Path(path).resolve() for path in (args.neuron_map, args.benign_dataset, *args.malicious_datasets)}
    if len(outputs) != len(output_paths) or outputs & inputs:
        raise ValueError("Output files must have distinct paths and cannot overwrite input files.")

    load_runtime_dependencies()
    disable_llava_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        False,
        False,
        device=args.device,
        use_flash_attn=False,
    )
    model.eval()
    conv_mode = args.conv_mode or infer_conv_mode(model_name)
    mip_neurons = load_mip_neurons(args.neuron_map)
    probe = MIPSaliencyProbe(model, mip_neurons)

    benign_scores = process_and_write(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        probe=probe,
        items=benign_items,
        output_path=args.benign_output,
        conv_mode=conv_mode,
        base_folder=args.benign_image_folder,
    )
    harmful_scores = process_and_write(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        probe=probe,
        items=malicious_items,
        output_path=args.malicious_output,
        conv_mode=conv_mode,
        base_folder=args.malicious_image_folder,
    )
    report, curves = fit_threshold(benign_scores, harmful_scores)
    report.update(
        model_path=args.model_path,
        conv_mode=conv_mode,
        model_dtype=str(model.dtype),
        neuron_map=args.neuron_map,
        mip_map_sha256=mip_map_sha256(mip_neurons),
        sensing_protocol="relative saliency before local reinforcement, using the signed preceding-layer peak and layerwise L1-normalized MIP scores, prefill only",
        calibration_inputs={"benign": args.benign_dataset, "harmful": args.malicious_datasets},
        score_csvs={"benign": args.benign_output, "harmful": args.malicious_output},
    )
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    with threshold_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.plot_output:
        save_plot(args.plot_output, report["threshold"], curves)
    print(f'Threshold = {report["threshold"]:.8f} ({threshold_output})')


if __name__ == "__main__":
    main()

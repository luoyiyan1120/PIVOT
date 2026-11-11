# PIVOT

PIVOT is an inference-time safety intervention for multimodal language models.
It measures relative intent saliency from localized malicious-intent perception
(MIP) neurons and uses that signal to control an intervention on turning (T)
neurons. This repository contains the LLaVA-1.5 implementation and analysis
scripts.

## Release scope

This is a source-code release. The paper's localized neuron indices and scores,
calibrated thresholds, dataset splits, processed benchmark data, model responses,
and experimental results are currently withheld. The repository does not include
a ready-to-run PIVOT configuration.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/pivot/llava_pivot.py` | MIP saliency measurement and the MIP/T inference hooks |
| `scripts/build_contrastive_pairs.py` | Prepare context-contrastive probing pairs from local inputs |
| `scripts/compute_layer_similarity_llava.py` | Measure layer-wise similarity for MIP layer analysis |
| `scripts/plot_layer_similarity.py` | Plot layer-wise similarity from a local CSV |
| `scripts/locate_mip_neurons_llava.py` | Locate MIP neurons using contrastive probing |
| `scripts/locate_sc_neurons_llava.py` | Locate safety-critical candidates using layer conductance |
| `scripts/split_sc_neurons_logits_lens_llava.py` | Classify turning and refusal candidates using a logit lens |
| `scripts/compute_mip_threshold_scores_llava.py` | Measure saliency and calibrate a threshold on local data |
| `scripts/run_pivot_mmsafetybench.py` | Generate responses from supplied MM-SafetyBench inputs |

The PIVOT controller uses MIP and T neurons. The logit-lens analysis also
produces refusal candidates for separate analyses.

## Prerequisites

Model-facing scripts require the native LLaVA Python package, a compatible
PyTorch environment, and a separately obtained LLaVA-1.5 checkpoint. See the
[official LLaVA repository](https://github.com/haotian-liu/LLaVA) for its runtime
and checkpoint setup. `requirements.txt` lists the additional Python
dependencies used by the scripts.

PIVOT inference requires a local MIP/T neuron map and a threshold calibrated
with the same checkpoint, MIP map, and sensing implementation. The inference
script accepts a combined MIP/T map or a separate T map via
`--t-neuron-map`. Benchmark files and images are supplied locally. Each script
documents its input and output arguments via `--help`.

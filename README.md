<div align="center">

# RVL-DiffGrid

### Robust Vision-Language Diffusion for Counting with Hierarchical Semantic Grids

<p>
<a href="assets/RVL_DiffGrid.pdf"><img src="https://img.shields.io/badge/Venue-IEEE%20ICME%202026-1f6feb.svg"></a>
<a href="#results"><img src="https://img.shields.io/badge/Task-Crowd%20Counting-8a2be2.svg"></a>
<a href="requirements.txt"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
<a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c.svg"></a>
<a href="https://github.com/Masnurul17/RVL_diffgrid/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

Mas Nurul Achmadiah · Muhammad Ridha Agam · Chi-Chia Sun · Mao-Hsiu Hsu · Wen-Kai Kuo · Jun-Wei Hsieh

<sub>National Formosa University · National Taipei University · National Yang Ming Chiao Tung University</sub>

<p>
📄 <a href="assets/RVL_DiffGrid.pdf"><b>Paper (PDF)</b></a> &nbsp;·&nbsp;
💻 <a href="https://github.com/Masnurul17/RVL_diffgrid"><b>Code</b></a>
</p>

Official PyTorch implementation of **RVL-DiffGrid**. If you find this repository useful, please give it a star ⭐.

<img src="assets/architecture.png" width="92%">

</div>

---

## Overview

Crowd counting in highly congested scenes is hard: global semantic cues alone are not enough for
fine-grained density estimation. **RVL-DiffGrid** is a diffusion-enhanced vision–language framework
with a **dual-branch** design:

- A **global (CLIP) branch** that supplies *training-time* semantic supervision. It is **discarded at
  inference**; the CLIP encoders are retained only to condition the U-Net cross-attention.
- A **diffusion-guided spatial branch** that encodes the image into latent space, applies a **single
  forward diffusion step** at a fixed timestep, and extracts multi-scale cross-attention from a
  **frozen, text-conditioned U-Net**, organized into **Hierarchical Semantic Grids (HSG)** with
  structural regularization.

The final count is obtained by **spatially integrating the predicted density map** — no fine-tuning of
the diffusion backbone and no generative sampling. With only **0.30M trainable parameters** and
**13.56 ms / image (~73.7 FPS)** on an RTX 5090, RVL-DiffGrid reaches competitive accuracy across five
benchmarks.

## Contents

- [Architecture](#architecture)
- [Results](#results)
- [Attention Visualization](#attention-visualization)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository Structure & Status](#repository-structure--status)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Architecture

RVL-DiffGrid couples a CLIP-based global branch (used only for training supervision) with a frozen
diffusion branch that produces the density map used at inference.

<div align="center">
<table>
<tr>
<td align="center"><b>Global Branch (training-only)</b></td>
</tr>
<tr>
<td><img src="assets/clip_branch.png" width="640"></td>
</tr>
<tr>
<td align="center"><b>Diffusion Grid Count Branch (inference path)</b></td>
</tr>
<tr>
<td><img src="assets/diffusion_branch.png" width="780"></td>
</tr>
</table>
</div>

**Pipeline (paper §III):**

1. **CLIP encoding & fusion** — image/text embeddings `Fv`, `Et` are fused via a lightweight mapping
   `Ffused = ϕ(Fv, Et)` (Eq. 3). The fused embedding conditions the global counting head **and** is
   projected to the diffusion context (768-dim, `N = 77` tokens), bypassing the diffusion text encoder.
2. **VAE latent encoding** — `z0 = VAEenc(I)`; input resized to `512×512`, yielding a `64×64×4` latent (Eq. 7).
3. **Single forward diffusion step** at a fixed timestep **`t = 500`** (Eq. 8).
4. **Frozen U-Net** extracts multi-scale latent features `F = {F64, F32, F16}` (Eq. 10) and
   cross-attention maps `As`, `s ∈ {64, 32, 16}` (Eq. 11); features are refined by `Φ(·)` into `Fref` (Eq. 12).
5. **Hierarchical Semantic Grids (HSG)** — attention maps are resized to a fixed `16×16` grid,
   per-image min–max normalized, and fused with uniform weights `ws = 1/3` (Eq. 13–14).
6. **Density regression** — `D̂ = Γ(Fref, S)` (Eq. 15); the final count is
   `Ĉ = Σ D̂(x, y)` (Eq. 16).

**Training objective** (Eq. 20):

```
L_total = L_dens + L_global + L_struct
        = L_dens
        + (L_Huber + λ_rel · L_rel + λ_ADR · L_ADR)      # global branch (Eq. 4–6)
        + (λ_LRRC · L_LRRC + λ_grid · L_grid)            # structural reg. (Eq. 17–19)
```

At inference, only the density-regression branch is kept; the global and structural terms act purely
as training-time supervision.

## Results

Performance on five crowd-counting benchmarks (MAE / RMSE; lower is better). Best per column in **bold**.

| Method | SH Part A | SH Part B | UCF-CC-50 | UCF-QNRF | JHU-Crowd++ |
|---|:---:|:---:|:---:|:---:|:---:|
| CSRNet | 68.2 / 115.0 | 10.6 / 16.0 | 266.1 / 397.5 | 120.3 / 208.5 | – |
| DM-Count | 59.7 / 95.7 | 7.4 / 11.8 | 211.0 / 291.5 | 85.6 / 148.3 | – |
| LIMM | 50.8 / 84.2 | 6.5 / 10.2 | – | 76.4 / 125.3 | 53.0 / **207.9** |
| DSGC-Net | **48.9** / **77.8** | 5.9 / **9.3** | – | 79.3 / 133.9 | – |
| CLIP-EBC | 52.5 / 85.9 | 6.6 / 10.5 | – | 80.3 / 139.3 | – |
| **RVL-DiffGrid (ours)** | 51.39 / 94.36 | **5.79** / 9.12 | 217.88 / **268.11** | **75.19** / **121.78** | 56.21 / 215.30 |

RVL-DiffGrid obtains the best density-regression results on **SH Part B**, **UCF-QNRF**, and the lowest
**RMSE on UCF-CC-50**, while staying competitive on Part A and JHU-Crowd++.

<details>
<summary><b>Ablation — components on ShanghaiTech Part A (Table II)</b></summary>

| DGC | HSG | LRRC | MAE | RMSE |
|:---:|:---:|:---:|:---:|:---:|
| – | – | – | 87.83 | 141.01 |
| ✓ | – | – | 65.80 | 121.70 |
| – | ✓ | ✓ | 67.26 | 100.85 |
| ✓ | ✓ | – | 62.68 | 97.06 |
| ✓ | ✓ | ✓ | **51.39** | **94.36** |

</details>

<details>
<summary><b>Parameter sensitivity (Table III) — default t=500, g=16, uniform weights</b></summary>

| Timestep t | MAE / RMSE | Grid g | MAE / RMSE |
|:---:|:---:|:---:|:---:|
| 250 | 57.43 / 103.21 | 8×8 | 54.76 / 98.53 |
| **500** | **51.39 / 94.36** | **16×16** | **51.39 / 94.36** |
| 750 | 55.87 / 100.64 | 32×32 | 53.21 / 97.18 |

</details>

<details>
<summary><b>Efficiency (RTX 5090, batch size 1)</b></summary>

| Total params | Trainable | GFLOPs | Latency | Throughput |
|:---:|:---:|:---:|:---:|:---:|
| 428.24M | **0.30M** | 123.21 | 13.56 ms | ~73.7 FPS |

</details>

## Attention Visualization

Cross-attention under three prompt conditions on SH Part A (numbers = full-dataset MAE). The standard
prompt consistently activates crowd-dense regions; random or removed prompts yield noisier, less
structured attention — confirming language as a semantic anchor (Table IV).

<div align="center">
<img src="assets/attn_vis.png" width="80%">
</div>

| Prompt | MAE / RMSE | Prompt | MAE / RMSE |
|---|:---:|---|:---:|
| Standard (`"a crowd of people"`) | **51.39 / 94.36** | No prompt (removed) | 58.21 / 104.92 |
| Random prompt | 56.83 / 101.47 | Visual-only token | 54.67 / 98.13 |

## Installation

```bash
git clone https://github.com/Masnurul17/RVL_diffgrid.git
cd RVL_diffgrid

conda create -n rvldiffgrid python=3.10 -y
conda activate rvldiffgrid
pip install -r requirements.txt
```

> The diffusion branch downloads Stable Diffusion v1.5 from the Hugging Face Hub on first run.
> `diffusion_backbone.py` relies on `Attention.get_attention_scores` (valid for `diffusers>=0.21`).

## Dataset Preparation

Supported benchmarks: **ShanghaiTech Part A/B**, **UCF-CC-50**, **UCF-QNRF**, **JHU-Crowd++**.
Each split is described by a JSON file; the loader accepts flexible keys
(`image_path` / `img_path` / `image_id`, and `gt_count` or a `points` list).

```
<dataset_root>/
├── train/images/*.jpg
├── test/images/*.jpg
└── Processed_JSON/
    ├── annotations_train.json
    └── annotations_test.json
```

Ground-truth density maps use adaptive Gaussian kernels (ShanghaiTech / UCF) and fixed kernels
(JHU-Crowd++), as described in the paper (§IV-A).

## Training

```bash
python scripts/train.py \
  --train_json <root>/Processed_JSON/annotations_train.json \
  --test_json  <root>/Processed_JSON/annotations_test.json \
  --ckpt_dir   checkpoints/sha \
  --config     configs/sha.yaml
```

Per-dataset learning rates and loss weights (`λ_ADR`, `λ_rel`, `λ_LRRC`, `λ_grid`) follow Table V and
live in `configs/`.

## Evaluation

The reported count comes from **spatial integration of the density map** (Eq. 16):

```bash
python scripts/eval.py \
  --test_json <root>/Processed_JSON/annotations_test.json \
  --ckpt_path checkpoints/sha/best_mae.pth \
  --batch_size 1
```

## Repository Structure & Status

```
rvl-diffgrid/
├── configs/                         # per-dataset hyperparameters (Table V)
├── rvlc/
│   ├── models/
│   │   ├── clip_backbone.py         # CLIP ViT-L/14-336 encoder (global branch)
│   │   ├── fusion.py                # phi(Fv, Et) -> Ffused (Eq. 3)
│   │   ├── global_count_head.py     # global counting head (training-only)
│   │   ├── clip_to_unet_context.py  # CLIP -> 768-d context, N=77 tokens (Eq. 9)
│   │   ├── diffusion_backbone.py    # VAE + frozen U-Net + cross-attn (Eq. 8-11)
│   │   ├── refinement.py            # Phi(F64,F32,F16) -> Fref (Eq. 12)
│   │   ├── hsg.py                   # Hierarchical Semantic Grids (Eq. 13-14)
│   │   ├── density_regressor.py     # Gamma(Fref, S) -> D_hat (Eq. 15)
│   │   └── rvl_diffgrid.py          # full dual-branch model
│   ├── losses/                      # global / density / structural losses (Eq. 4-19)
│   ├── data/                        # dataset + GT density-map generation
│   └── utils/                       # transforms, metrics
├── scripts/{train.py, eval.py}
└── assets/                          # figures
```

| Component | Status |
|---|:---:|
| CLIP backbone, fusion, global head | ✅ implemented |
| `clip_to_unet_context` (Eq. 9) | ✅ implemented |
| `diffusion_backbone` — cross-attn capture, t=500 (Eq. 8–11) | ✅ implemented |
| `eval.py` — count from density integration (Eq. 16) | ✅ implemented |
| HSG, refinement Φ, density regressor Γ (Eq. 12–15) | ✅ implemented |
| Losses `L_dens / L_global / L_struct` (Eq. 4–20) | ✅ implemented |
| GT density-map generation — adaptive/fixed Gaussian (§IV-A) | ✅ implemented |
| `train.py` — full objective `L_total` (Eq. 20) | ✅ implemented |
| full model wiring `rvl_diffgrid.py` | ✅ implemented |

## Citation

If you use this work, please cite:



## Acknowledgements

This research was supported by the National Science and Technology Council, Taiwan
(Grant No. NSTC113-2221-E-305-018-MY3). The implementation builds on
[OpenCLIP](https://github.com/mlfoundations/open_clip),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) via
[🤗 Diffusers](https://github.com/huggingface/diffusers), and is inspired by
[CountGD](https://github.com/niki-amini-naieni/CountGD) and T2ICount.

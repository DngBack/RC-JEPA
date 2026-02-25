# RC-JEPA — Risk-Controllable Joint-Embedding Predictive Architecture

RC-JEPA endows JEPA world models with **statistical reliability guarantees**.
A lightweight probabilistic predictor is trained on top of a frozen JEPA encoder,
and **Conformal Risk Control (CRC)** in latent space lets the model provably
bound expected error on non-abstained predictions below a user-chosen risk
level α.

## Project structure

```
RC-JEPA/
├── configs/
│   ├── train_rc_jepa.yaml        # Phase 1 training hyper-parameters
│   └── eval_conformal.yaml       # Phase 2-3 calibration & evaluation
├── data/
│   ├── datasets.py               # ImageNet-100, CIFAR-100, ImageFolder
│   ├── masking.py                # Multi-block mask generator
│   └── corruptions.py           # ImageNet-C loader
├── models/
│   ├── frozen_encoder.py         # Self-contained ViT + frozen wrapper
│   ├── prob_predictor.py         # Lightweight Gaussian predictor (μ, σ²)
│   └── losses.py                 # NLL + Latent Ranking Loss
├── conformal/
│   ├── calibrator.py             # Hoeffding-based CRC → τ̂
│   └── selector.py               # Inference-time accept/abstain
├── scripts/
│   ├── train_predictor.py        # Phase 1: train probabilistic predictor
│   ├── calibrate.py              # Phase 2: find risk threshold τ̂
│   └── evaluate_shift.py         # Phase 3: test on clean + OOD data
├── utils.py
├── requirements.txt
└── docs/overview_idea.md
```

## Getting started

### 1. Create conda environment

From the project root:

```bash
cd /home/admin1/Desktop/RC-JEPA

# Create environment with Python 3.10
conda create -n rcjepa python=3.10 -y

# Activate it (required before every run)
conda activate rcjepa

# Install dependencies
pip install -r requirements.txt
```

You only need to create the env once. Use `conda activate rcjepa` whenever you open a new terminal to run the scripts.

### 2. Run the full pipeline (CIFAR-100)

The config uses **CIFAR-100** by default. Data is downloaded automatically into `./data_store`. Run these three steps in order:

```bash
cd /home/admin1/Desktop/RC-JEPA
conda activate rcjepa
pip install -r requirements.txt

# Phase 1 — Train the probabilistic predictor (encoder frozen)
python3 scripts/train_predictor.py --config configs/train_rc_jepa.yaml --data_dir ./data_store

# Phase 2 — Calibrate risk threshold on held-out data
python3 scripts/calibrate.py \
  --ckpt_path outputs/best_predictor.pth \
  --calib_dir ./data_store \
  --target_alpha 0.1 \
  --save_threshold outputs/tau_alpha0.1.json

# Phase 3 — Evaluate on clean data and save plots
python3 scripts/evaluate_shift.py \
  --ckpt_path outputs/best_predictor.pth \
  --threshold_file outputs/tau_alpha0.1.json \
  --test_clean ./data_store \
  --output_dir results/plots
```

- **Phase 1**: Checkpoints go to `outputs/`. Watch `latent_spearman` in the logs; aim for **> 0.80**.
- **Phase 2**: Writes `outputs/tau_alpha0.1.json` (threshold for risk level α = 0.1).
- **Phase 3**: Writes `results/plots/risk_coverage_curve.png` and `results/plots/safe_degradation.png`.

### 3. Using ImageNet-100 or ImageNet-C

- **ImageNet-100**: Put train/val in e.g. `data_store/imagenet100/train` and `.../val`. Use a separate **calibration** split (e.g. 20% of val → `val_calib/`) and set `--calib_dir` and `--data_dir` accordingly.
- **ImageNet-C**: Set `--test_corrupt /path/to/imagenet_c` in the evaluate script to test under distribution shift.

## Key idea (one paragraph)

Standard JEPAs predict latent targets deterministically and cannot flag
unreliable predictions. RC-JEPA replaces the predictor with a Gaussian head
trained via **NLL + a Latent Ranking Loss** (ensuring uncertainty scores
monotonically track true errors). At calibration time, CRC finds a threshold
τ̂ on the uncertainty score such that the **expected latent error on accepted
samples ≤ α** with high probability. By Theorem 1, bounding latent error also
bounds downstream task error (up to the Lipschitz constant of the task head).

## Citation

If you use this code, please cite the accompanying paper (forthcoming).

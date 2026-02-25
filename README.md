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

## Quick start

### 1. Environment

```bash
conda create -n rcjepa python=3.10 -y && conda activate rcjepa
pip install -r requirements.txt
```

### 2. Phase 1 — Train the probabilistic predictor

The encoder is frozen; only the predictor receives gradients.

```bash
python scripts/train_predictor.py \
    --config configs/train_rc_jepa.yaml \
    --data_dir /path/to/imagenet100/train
```

Track `metrics/latent_spearman` — it should exceed **0.80** for the conformal
selector to work well. Adjust `--lambda_rank` / margin `gamma` if needed.

### 3. Phase 2 — Calibrate the risk threshold

Use a **strictly held-out** calibration split (≥ 20 % of validation).

```bash
python scripts/calibrate.py \
    --ckpt_path outputs/best_predictor.pth \
    --calib_dir /path/to/imagenet100/val_calib \
    --target_alpha 0.1 \
    --delta 0.1
```

### 4. Phase 3 — Evaluate on clean + shifted data

```bash
python scripts/evaluate_shift.py \
    --ckpt_path outputs/best_predictor.pth \
    --threshold_file outputs/thresholds/tau_alpha0.1.json \
    --test_clean /path/to/imagenet100/val_test \
    --test_corrupt /path/to/imagenet_c
```

Produces `risk_coverage_curve.png` and `safe_degradation.png` in the output
directory.

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

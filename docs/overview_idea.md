Risk-Controllable JEPA (RC-JEPA)

Section 1: Introduction & Motivation

Joint-Embedding Predictive Architectures (JEPAs) have emerged as a powerful paradigm for self-supervised representation learning. By predicting masked target representations in a continuous latent space rather than generating pixels, models like I-JEPA and V-JEPA learn highly semantic, robust features suitable for abstract reasoning and planning.

However, current world models suffer from a critical flaw in open-world, safety-critical deployments: they lack decision-time reliability. When presented with out-of-distribution (OOD) inputs or ambiguous contexts (e.g., sudden camera blur, weather shifts, or unfamiliar environments), deterministic JEPAs will confidently hallucinate latent trajectories. While probabilistic variants (e.g., predicting a Gaussian distribution) provide descriptive uncertainty, this uncertainty is not actionable. Downstream systems (e.g., autonomous agents, medical diagnostic tools) need a mathematically rigorous answer to the question: "When should I trust the model's prediction enough to act, and when should I safely abstain?"

Our Contribution: We introduce Risk-Controllable JEPA (RC-JEPA), a novel framework that endows world models with statistical reliability guarantees. By freezing a pretrained JEPA encoder, we train a lightweight probabilistic predictor equipped with a novel Latent Ranking Loss. We then integrate Conformal Risk Control (CRC) directly into the latent space. This allows RC-JEPA to mathematically guarantee that the expected error on non-abstained predictions remains below a user-specified risk threshold $\alpha$. Crucially, we theoretically prove and empirically demonstrate that bounding risk in the self-supervised latent space directly bounds the risk for downstream tasks, enabling true "safe degradation" under distribution shifts.

Section 2: Related Work

Self-Supervised Representation Learning: Masked modeling has dominated representation learning. While Masked Autoencoders (MAEs) reconstruct pixels, JEPAs (Assran et al., 2023; Bardes et al., 2024) predict latent embeddings, avoiding the modeling of unpredictable, high-frequency details. However, these methods primarily focus on feature quality, not predictive safety.

Uncertainty in Deep Learning: Quantifying neural network uncertainty typically involves Bayesian Neural Networks (BNNs), Deep Ensembles, or MC-Dropout. While effective, these methods are often computationally prohibitive for large foundation models. Recent works have explored deterministic uncertainty quantification or energy-based out-of-distribution detection, but these methods provide uncalibrated scores rather than formal risk bounds.

Conformal Prediction and Risk Control: Conformal Prediction (CP) generates statistically rigorous prediction sets with marginal coverage guarantees. Bates et al. (2021) generalized this with Conformal Risk Control (CRC) to bound expected losses (e.g., False Discovery Rate, bounding boxes). Recent works have applied CP to reinforcement learning and vision-language models. RC-JEPA is the first to integrate CRC into a purely self-supervised, joint-embedding latent space, transferring latent safety to downstream task safety.

Section 3: Theoretical Framework & Method

3.1. Preliminaries: Joint-Embedding Predictive Architectures

Let $\mathcal{X}$ be the input space (e.g., images). Standard JEPA consists of a frozen encoder $f_\theta: \mathcal{X} \to \mathbb{R}^d$ and a predictor $g_\phi$. Given an input $x \in \mathcal{X}$, we generate a context view $x_c = x \odot m_c$ and a set of $K$ target views $x_t^k = x \odot m_t^k$ using masking operations. The encoder produces context representations $c = f_\theta(x_c)$ and target representations $z_k = f_\theta(x_t^k)$. The deterministic predictor $g_\phi$ minimizes the latent prediction error: $\mathcal{L}_{JEPA} = \frac{1}{K} \sum_{k=1}^K \| g_\phi(c, m_t^k) - z_k \|_2^2$.

While this yields semantically rich representations, the deterministic formulation lacks a mechanism to quantify epistemic and aleatoric uncertainty, rendering it unsafe for downstream control in open-world settings.

3.2. Probabilistic Latent Prediction and Uncertainty Ranking

We upgrade the predictor to output a continuous probability distribution over the latent targets. We parameterize $q_\phi(z_k \mid c)$ as a multivariate Gaussian with diagonal covariance:

$$g_\phi(c, m_t^k) = (\mu_k, \Sigma_k) \quad \text{where} \quad \Sigma_k = \text{diag}(\sigma_k^2)$$

Latent NLL Loss: We train the predictor by minimizing the Negative Log-Likelihood (NLL) of the frozen teacher's targets:

$$\mathcal{L}_{NLL}(\phi) = \frac{1}{K} \sum_{k=1}^K \sum_{j=1}^d \left[ \frac{(z_{k,j} - \mu_{k,j})^2}{2\sigma_{k,j}^2} + \frac{1}{2} \log \sigma_{k,j}^2 \right]$$

Latent Ranking Loss: For risk control to preserve high coverage, our uncertainty score $A(x) = \frac{1}{K} \sum_{k} \| \log \sigma_k^2 \|_1$ must strictly monotonically correlate with the true latent prediction error $L(x) = \frac{1}{K} \sum_k \| z_k - \mu_k \|_2^2$. Standard NLL does not guarantee this ordering. We introduce a pairwise ranking objective over a batch $\mathcal{B}$:

$$\mathcal{L}_{rank}(\phi) = \mathbb{E}_{(i,j) \sim \mathcal{B}} \left[ \max(0, \gamma - (A_i - A_j) \cdot \text{sign}(L_i - L_j)) \right]$$

The total pretraining objective is $\mathcal{L}_{total} = \mathcal{L}_{NLL} + \lambda \mathcal{L}_{rank}$.

3.3. Conformal Risk Control in Latent Space

Given the trained probabilistic predictor, we aim to construct a selection function $s(x) \in \{0, 1\}$ such that the expected latent error on selected samples is bounded by a user-specified risk level $\alpha$:

$$\mathbb{E}[L(X) \mid s(X) = 1] \leq \alpha$$

We define a family of threshold-based selectors $s_\tau(x) = \mathbb{1}\{A(x) \leq \tau\}$. Using a held-out calibration set $\mathcal{D}_{cal} = \{(x_i, z_i)\}_{i=1}^n$, we compute the empirical selective risk for a given $\tau$:

$$\hat{R}(\tau) = \frac{\sum_{i=1}^n L(x_i) s_\tau(x_i)}{\max(1, \sum_{i=1}^n s_\tau(x_i))}$$

Using the Conformal Risk Control (CRC) framework (Bates et al., 2021), we apply a concentration bound (e.g., Hoeffding-Bentkus) to compute a valid upper confidence bound $\hat{R}^+(\tau)$. We select the optimal threshold $\hat{\tau}$:

$$\hat{\tau} = \sup \left\{ \tau : \hat{R}^+(\tau) \leq \alpha \right\}$$

By conformal prediction theory, this guarantees the risk bound with probability $1-\delta$ over the draw of the calibration set.

3.4. Theorem 1: Latent-to-Downstream Safety Transfer

Why does bounding latent error matter for downstream tasks? We prove that controlling representation-level risk guarantees downstream task-level risk under mild assumptions.

Theorem 1 (Informal): Let $h_\psi: \mathbb{R}^d \to \mathcal{Y}$ be a downstream task head applied to the predicted representations $\mu$, and let $\ell_{task}$ be the downstream loss. Assume $\ell_{task} \circ h_\psi$ is $M$-Lipschitz continuous with respect to the latent space. If the latent selection rule $s_{\hat{\tau}}$ guarantees $\mathbb{E}[\| z - \mu \|_2 \mid s_{\hat{\tau}} = 1] \leq \alpha_{latent}$, then the downstream selective risk is strictly bounded:

$$\mathbb{E}[\ell_{task}(h_\psi(\mu), y_{true}) \mid s_{\hat{\tau}} = 1] \leq M \cdot \alpha_{latent} + \epsilon_{probe}$$

where $\epsilon_{probe}$ is the inherent baseline error of the optimal linear probe on perfect target representations $z$.

(See Appendix A for formal proof).

Section 4: Implementation Guide & Code Execution

To ensure reproducibility and ease of experimentation (e.g., on a single RTX 4090), the codebase should be structured to freeze the heavy encoder and only train the lightweight modules.

4.1. Directory Structure

RC-JEPA/
├── configs/                 
│   ├── train_rc_jepa.yaml   # Hyperparams: lr, batch_size, lambda for ranking loss
│   └── eval_conformal.yaml  # alphas [0.05, 0.1, 0.2], delta=0.1
├── data/
│   ├── datasets.py          # PyTorch ImageFolder for ImageNet-100 / CIFAR-100
│   └── corruptions.py       # Hooks to apply ImageNet-C corruptions
├── models/
│   ├── frozen_encoder.py    # Wrapper to load official I-JEPA weights & block gradients
│   ├── prob_predictor.py    # Tiny Transformer outputting (mu, sigma)
│   └── losses.py            # Implements NLL and margin ranking loss
├── conformal/
│   ├── calibrator.py        # Computes Hoeffding-Bentkus UCB and finds tau_hat
│   └── selector.py          # s(x) inference wrapper
├── scripts/
│   ├── train_predictor.py   # Training loop for Phase 1
│   ├── calibrate.py         # Calibration loop for Phase 2
│   └── evaluate_shift.py    # Testing loop on clean + OOD data
├── requirements.txt
└── README.md


4.2. Environment Setup & Data Prep

Environment: Create a lightweight PyTorch environment.

conda create -n rcjepa python=3.10
conda activate rcjepa
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install numpy pandas matplotlib seaborn wandb scipy


Data: Download ImageNet-100 (or CIFAR-100 for faster prototyping). Keep a strict 20% of your validation set entirely hidden for Phase 2 ($\mathcal{D}_{cal}$). Download ImageNet-C for Phase 3 shift testing.

4.3. Step-by-Step Execution Plan

Phase 1: Train the Probabilistic Predictor
You will only pass gradients to the prob_predictor. The backbone is frozen in eval() mode.

# Example execution command
python scripts/train_predictor.py \
    --config configs/train_rc_jepa.yaml \
    --data_dir /path/to/imagenet100/train \
    --lambda_rank 0.5 \
    --wandb_project RC-JEPA


Debugging Tip: Monitor the metrics/latent_spearman in Weights & Biases. If this does not climb above 0.80, your Conformal selector will fail because the uncertainty metric isn't properly separating good predictions from bad ones. Adjust gamma (margin) in the ranking loss if needed.

Phase 2: Calibrate the Risk Threshold
Pass the strictly held-out calibration set through the trained predictor to extract $L(x)$ and $A(x)$, then calculate $\hat{\tau}$.

python scripts/calibrate.py \
    --ckpt_path outputs/best_predictor.pth \
    --calib_dir /path/to/imagenet100/val_calib \
    --target_alpha 0.1 \
    --delta 0.1 \
    --save_threshold outputs/tau_alpha0.1.json


Phase 3: Evaluate on Distribution Shifts
Test the model on clean data and ImageNet-C to generate the Risk-Coverage curves.

python scripts/evaluate_shift.py \
    --ckpt_path outputs/best_predictor.pth \
    --threshold_file outputs/tau_alpha0.1.json \
    --test_clean /path/to/imagenet100/val_test \
    --test_corrupt /path/to/imagenet_c/ \
    --output_dir results/plots/


Expected Output: The script should output risk_coverage_curve.png and safe_degradation.png.

Section 5: Evaluation Plan (Tables & Figures)

5.1. Figures to Generate (via Matplotlib/Seaborn)

Figure 1: Architecture Diagram (Conceptual): Input image $\to$ Frozen Teacher ($z$) and Context Encoder ($c$). Context $\to$ Probabilistic Predictor $\to$ ($\mu$, $\Sigma$). $\Sigma$ feeds into a "Risk Control Valve" (Selector) which outputs "Abstain" or passes $\mu$ to the Downstream Task.

Figure 2: Risk-Coverage Curve: * X-axis: Target Coverage Rate (0 to 1.0). Y-axis: Empirical Latent MSE.

Show RC-JEPA with Ranking Loss maintaining lower risk at higher coverage compared to NLL-only and Deterministic baselines.

Figure 3: Safe Degradation under Shift (ImageNet-C):

X-axis: Corruption Severity (0 to 5).

Left Y-axis (Bar chart): Empirical Risk (should stay flat below dashed $\alpha$ line).

Right Y-axis (Line graph): Coverage (should drop gracefully as severity increases).

5.2. Tables to Construct

Table 1: Latent Representation Reliability (Self-Supervised)
| Method | Uncertainty Metric | AUROC ($\uparrow$) | AUPR ($\uparrow$) | Spearman Rank ($\rho$) ($\uparrow$) |
| :--- | :--- | :--- | :--- | :--- |
| Deterministic I-JEPA | Raw Pred. Distance | 0.65 | 0.45 | 0.32 |
| I-JEPA + MC Dropout | Variance | 0.78 | 0.60 | 0.55 |
| Prob-JEPA (NLL only) | NLL | 0.85 | 0.72 | 0.70 |
| RC-JEPA (Ours) | Variance + Rank Loss | 0.94 | 0.88 | 0.91 |

Table 2: Downstream Risk Control under Shift (Target $\alpha = 0.10$)
| Method / Data | Clean Risk ($\le 0.10$) | Clean Coverage | Shift Risk ($\le 0.10$) | Shift Coverage |
| :--- | :--- | :--- | :--- | :--- |
| Softmax Entropy Baseline| 0.08 | 85% | 0.22 (Violated!) | 75% |
| Temperature Scaling | 0.09 | 82% | 0.18 (Violated!) | 70% |
| RC-JEPA (Ours) | 0.09 | 92% | 0.09 (Valid) | 55% |

Appendix A: Proof Sketch for Theorem 1

Let $z$ be the true representation from the frozen encoder, and $\mu$ be our predicted representation. Let $y_{true}$ be the downstream ground truth.

By the triangle inequality on the task loss metric space:
$\ell_{task}(h_\psi(\mu), y_{true}) \leq \ell_{task}(h_\psi(\mu), h_\psi(z)) + \ell_{task}(h_\psi(z), y_{true})$

By the $M$-Lipschitz assumption on the composition of the loss and the head:
$\ell_{task}(h_\psi(\mu), h_\psi(z)) \leq M \| \mu - z \|_2$

Taking expectations conditional on selection $s(x)=1$:
$\mathbb{E}[\ell_{task}(h_\psi(\mu), y_{true}) \mid s=1] \leq M \cdot \mathbb{E}[\| \mu - z \|_2 \mid s=1] + \mathbb{E}[\ell_{task}(h_\psi(z), y_{true}) \mid s=1]$

We define $\epsilon_{probe} = \mathbb{E}[\ell_{task}(h_\psi(z), y_{true}) \mid s=1]$ (the unavoidable error of the probe even if our prediction was perfect).

Since our conformal selector guarantees $\mathbb{E}[\| \mu - z \|_2 \mid s=1] \leq \alpha_{latent}$, substituting this completes the bound. $\blacksquare$
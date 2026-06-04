# Universal Convergence of TD(0) — Numerical Experiments

Numerical experiments validating theoretical convergence results for **TD(0)
with linear function approximation**.  The code studies how the convergence
rate of the Polyak–Ruppert averaged iterates depends on the discount factor γ,
the eigenvalue spectrum of the feature second-moment matrix Σ, and the
structure of the feature matrix Φ.

## Installation

**With conda (recommended):**

```bash
git clone <repo-url>
cd numerics
conda create -n td0_numerics python=3.11 numpy matplotlib scipy -y
conda activate td0_numerics
```

**With pip:**

```bash
git clone <repo-url>
cd numerics
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Python ≥ 3.10 is required (the code uses the `X | Y` union-type syntax).

## Running experiments

Scripts can be run from the project root or from inside `src/`.  Each
simulation script saves a `.npz` archive in `experiments/data/`; the matching
`draw_*.py` script reads that archive and writes a `.png` to
`experiments/figures/`.

```bash
# --- Controlled eigenvalue spectrum (d = 3) ---
python src/simulations/Controlled_EV_d3.py
python src/draw_from_simulations/draw_controlled_EV_d3_clean.py

# --- Gamma-dependence experiment ---
python src/simulations/gamma_dependence.py
python src/draw_from_simulations/draw_gamma_dependence.py       # C(γ) vs (1−γ) log-log plot
python src/draw_from_simulations/draw_SC_gamma_dependence.py    # sanity-check curves

# --- Learning-rate schedule comparison (uniform vs split) ---
python src/simulations/compare_td0_with_different_LR.py
python src/draw_from_simulations/draw_compare_td0_with_different_LR.py

# --- PCTD0 vs TD0 comparison ---
python src/simulations/compare_pctd0_and_td0.py
python src/draw_from_simulations/draw_compare_pctd0_and_td0.py

# --- PCTD0 across multiple gamma values ---
python src/simulations/pctd0_simulations.py
python src/draw_from_simulations/draw_pctd0_simulations.py
```

Global parameters (number of states N, number of features D, discount γ,
batch size, etc.) are defined in `src/simulations/parameters_simulations.py`.

## Repository structure

```
numerics/
├── src/
│   ├── markov_reward_process.py       # MarkovRewardProcess class
│   ├── parametrisation.py             # LinearParametrisation and subclasses
│   │                                  #   DiagonalParametrisation
│   │                                  #   RandomLinearParametrisation
│   ├── td0.py                         # TD(0) learner — training loop & error metrics
│   │                                  #   StandardTD0LinearFA
│   │                                  #   PCTD0LinearFA
│   ├── matrix_generators.py           # generate_stochastic, generate_bistochastic
│   │
│   ├── simulations/                   # experiment generation scripts
│   │   ├── parameters_simulations.py  # shared configuration (N, D, γ, batch size …)
│   │   ├── mse_helpers.py             # high-level experiment helpers
│   │   ├── Controlled_EV_d3.py        # controlled Σ spectrum, d = 3
│   │   ├── gamma_dependence.py        # C(γ) vs (1−γ) sweep
│   │   ├── compare_td0_with_different_LR.py   # uniform vs split LR schedule
│   │   ├── compare_pctd0_and_td0.py           # PCTD0 vs TD0
│   │   ├── td0_simulations.py                 # standard TD0 baseline
│   │   └── pctd0_simulations.py               # PCTD0 across γ values
│   │
│   └── draw_from_simulations/         # visualisation scripts (one per experiment)
│       ├── draw_controlled_EV_d3_clean.py
│       ├── draw_gamma_dependence.py
│       ├── draw_SC_gamma_dependence.py
│       ├── draw_compare_td0_with_different_LR.py
│       ├── draw_compare_pctd0_and_td0.py
│       └── draw_pctd0_simulations.py
│
├── experiments/
│   ├── data/      # generated .npz archives (git-ignored, created by simulations/)
│   └── figures/   # generated .png figures  (git-ignored, created by draw_from_simulations/)
│
├── requirements.txt
└── README.md
```

## Core modules

### `src/markov_reward_process.py` — `MarkovRewardProcess`

Finite MRP with a stochastic transition matrix P, a deterministic reward
vector r(s), and i.i.d. Gaussian reward noise N(0, σ²).  Computes the
stationary distribution ρ and the spectral gap at construction time.
Provides vectorised inverse-CDF samplers for transitions and rewards.

### `src/parametrisation.py` — linear function approximation

| Class | Feature matrix Φ |
|---|---|
| `LinearParametrisation` | Base class; caller sets `feature_matrix` directly |
| `DiagonalParametrisation` | Prescribed singular-value spectrum via random SVD |
| `RandomLinearParametrisation` | Entry-wise Uniform(−1,1), row-normalised so ‖φ(x)‖₂ ≤ 1 |

All classes support:
- `nb_parallel_copies`: run several independent instances in one batched call
- `nb_copies_to_average`: average multiple trajectories per step to reduce variance
- `do_average`: maintain a Polyak–Ruppert running average θ̄

### `src/td0.py` — `TD0LinearFA`

TD(0) learner.  Key design choices:
- **Batched tensor arithmetic**: all γ values and all copies are updated
  simultaneously with a single NumPy call per iteration.
- **Coupled experiments**: passing `nb_parallel_copies=2` with two rows of
  learning rates compares two settings under identical random samples.
- `train()` logs three MSE metrics at logarithmically-spaced checkpoints:
  `MSE_values`, `MSE_S`, `MSE_advantages`.

Two concrete classes:
- `StandardTD0LinearFA` — standard TD(0) update.
- `PCTD0LinearFA` — paired-comparison variant with mean-centred geometry.

### `src/simulations/mse_helpers.py` — experiment helpers

Thin wrappers that wire together an MRP, a parametrisation, and a TD(0) agent,
then return the averaged MSE array.  Experiment scripts call these instead of
managing the boilerplate themselves.

## Notation

| Symbol | Meaning |
|---|---|
| d | number of states |
| k | number of features |
| Φ | (d, k) feature matrix, Φ[s] = φ(s)ᵀ |
| P | (d, d) stochastic transition matrix |
| ρ | (d,) stationary distribution |
| D | diag(ρ) |
| γ | discount factor |
| θ | (k,) parameter vector, V_θ(s) = φ(s)ᵀ θ |
| θ* | TD(0) fixed point, solution of H θ* = b |
| H | Σ − γ Σ₁ (TD matrix) |
| S | (H + Hᵀ)/2 (symmetric part of H) |
| Σ | ΦᵀDΦ (feature second-moment matrix) |
| Σ₁ | ΦᵀDPΦ (one-step cross-covariance) |
| b | ΦᵀDr (projected rewards) |

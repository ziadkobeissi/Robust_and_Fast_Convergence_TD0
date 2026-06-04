"""
Gamma-dependence experiment.

Runs TD(0) simultaneously for a logarithmic grid of γ values, using
α(γ) = (1−γ)/4 as the learning rate.  The same MRP and feature matrix
are shared across all γ values via the num_coupled_trajectories mechanism.

Three variants are produced in a single run:
  - total error:   θ₀ ≠ 0  (bias),  σ > 0  (noise)
  - bias only:     θ₀ ≠ 0,           σ = 0
  - variance only: θ₀ = 0,           σ > 0

Each variant is saved as a separate .npz archive in experiments/data/.
Results are later visualised by:
  draw_gamma_dependence.py   — C(γ) vs (1−γ) log-log plot
  draw_SC_gamma_dependence.py — sanity-check MSE curves with C(γ)/t overlaid
"""

import time
from pathlib import Path

import sys
from pathlib import Path

import numpy as np

# Ensure core modules in src/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mse_helpers import compute_mean_mse_gamma_sweep


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED               = 3
D                  = 20            # number of states = number of features
DIRICHLET_PARAM    = 1.0 / D       # concentration parameter for random P
BATCH_SIZE         = 1             # mini-batch size (1 = standard TD(0))
NB_ITER            = int(1e7)      # TD(0) steps per variant
NB_COPIES_TO_AVG   = 100           # parallel copies averaged for variance reduction

NB_GAMMAS          = 20
# (1−γ) spans two decades from 0.01 to ~0.5, so γ ∈ [0.5, 0.99]
ONE_MINUS_GAMMA    = np.logspace(-2, np.log10(0.5), num=NB_GAMMAS)
GAMMA_VECT         = 1.0 - ONE_MINUS_GAMMA
ALPHA_VECT         = (1.0 - GAMMA_VECT) / 4.0   # one rate per γ

# Identity feature matrix: Φ = I_d (uniform singular-value spectrum)
DIAG_SIGMA_HALF    = np.ones(D)

# Logarithmically-spaced iteration indices at which MSE is recorded
START_PLOT          = 10
NB_MSE_CHECKPOINTS  = 200
_ratio              = (NB_ITER / START_PLOT) ** (1.0 / NB_MSE_CHECKPOINTS)
MSE_CHECKPOINTS     = np.unique(
    np.floor(START_PLOT * _ratio ** np.arange(NB_MSE_CHECKPOINTS + 1)).astype(int)
)

_ROOT    = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = _ROOT / "experiments" / "data"

# Three (with_bias, std_dev_noise) settings
VARIANTS = [
    (True,  1.0),   # total error  (bias + variance)
    (True,  0.0),   # bias only
    (False, 1.0),   # variance only
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path(d: int, with_bias: bool, std_dev_noise: float,
              batch_size: int = 1) -> Path:
    """Canonical path for a single variant's .npz archive."""
    tag = ("_WB" if with_bias else "_WoutB") + ("_WV" if std_dev_noise > 0 else "_WoutV")
    return SAVE_DIR / f"gamma_dependence_TD0_d{d}{tag}_B{batch_size}.npz"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0_total = time.perf_counter()
    np.random.seed(SEED)

    print(
        f"Gamma-dependence experiment  |  d={D}, {NB_GAMMAS} γ values, "
        f"γ ∈ [{GAMMA_VECT.min():.3f}, {GAMMA_VECT.max():.3f}], "
        f"batch={BATCH_SIZE}, iters={NB_ITER:,}, copies={NB_COPIES_TO_AVG}"
    )

    for with_bias, std_dev_noise in VARIANTS:
        t0 = time.perf_counter()

        initial_theta = (
            np.random.uniform(-1.0, 1.0, size=D) if with_bias else np.zeros(D)
        )

        mse = compute_mean_mse_gamma_sweep(
            num_states=D, gamma_vect=GAMMA_VECT, alpha_vect=ALPHA_VECT,
            std_dev_noise=std_dev_noise, initial_theta=initial_theta,
            dirichlet_param=DIRICHLET_PARAM, diag_Sigma_half=DIAG_SIGMA_HALF,
            nb_iter=NB_ITER, num_independent_copies=NB_COPIES_TO_AVG,
            mse_checkpoints=MSE_CHECKPOINTS, batch_size=BATCH_SIZE,
        )
        # mse shape: (len(checkpoints), 3, len(gamma_vect))

        dt = time.perf_counter() - t0
        print(f"  bias={with_bias}, σ={std_dev_noise}  →  {dt:.1f} s")

        results = {
            "gamma_vect":              GAMMA_VECT,
            "alpha_vect":              ALPHA_VECT,
            "batch_size":              BATCH_SIZE,
            "dimension":               D,
            "nb_iter":                 NB_ITER,
            "num_independent_copies":    NB_COPIES_TO_AVG,
            "with_bias":               with_bias,
            "std_dev_noise":           std_dev_noise,
            "dirichlet_param":         DIRICHLET_PARAM,
            "mse_checkpoints":         MSE_CHECKPOINTS,
            "MSE_values":              mse[:, 0, :],
            "MSE_constant":            mse[:, 1, :],
            "MSE_non_constant":        mse[:, 2, :],
            "start_plot":              START_PLOT,
            "nb_mse_checkpoints":      NB_MSE_CHECKPOINTS,
            "seed":                    SEED,
            "computation_time_seconds": dt,
        }

        savepath = data_path(D, with_bias, std_dev_noise, batch_size=BATCH_SIZE)
        np.savez(savepath, **results)
        print(f"  Saved → {savepath}")

    print(f"Total time: {time.perf_counter() - t0_total:.1f} s")


if __name__ == "__main__":
    main()

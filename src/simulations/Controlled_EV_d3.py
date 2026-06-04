"""
Experiment: controlled eigenvalue spectrum in dimension d=3.

Runs TD(0) for 5 values of k ∈ {0,1,2,3,4}.  For each k the feature matrix Φ
is built so that its singular values are (1, 10^{-k/2}, 10^{-k/2}), making the
smallest eigenvalue of the feature second-moment matrix Σ = Φᵀ D Φ equal to

    ω  =  10^{-k} / d

(where D = (1/d) I because P is bistochastic).  Varying k thus spans four
decades of eigenvalue spread, letting us study how ill-conditioning of Σ
affects the convergence rate of TD(0).

Results are saved as a .npz archive in experiments/data/ and are later
visualised by draw_controlled_EV_d3.py.
"""

import time
from pathlib import Path

import sys
from pathlib import Path

import numpy as np

# Ensure core modules in src/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mse_helpers import compute_mean_mse_controlled_ev


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED           = 2
D              = 3                      # number of states / features
GAMMA          = 0.9                  # discount factor
ALPHA          = (1.0 - GAMMA) / 4.0   # constant learning rate
BATCH_SIZE     = 1                      # mini-batch size (1 = standard TD(0))
NB_ITER        = int(1e5)              # TD(0) steps per trial
NB_TRIALS      = 10                  # independent copies averaged for MSE
DIRICHLET_PARAM = 1.0                  # Dirichlet parameter for the transition matrix

WITH_BIAS      = True   # whether θ₀ ≠ 0
STD_DEV_NOISE  = 0.0     # reward noise σ  (0.0 = deterministic rewards)

# Logarithmically-spaced iteration indices at which MSE is recorded
START_PLOT          = 10
NB_MSE_CHECKPOINTS  = 200
_ratio              = (NB_ITER / START_PLOT) ** (1.0 / NB_MSE_CHECKPOINTS)
MSE_CHECKPOINTS     = np.unique(
    np.floor(START_PLOT * _ratio ** np.arange(NB_MSE_CHECKPOINTS + 1)).astype(int)
)

_ROOT    = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = _ROOT / "experiments" / "data"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()
    np.random.seed(SEED)

    initial_theta = np.random.uniform(-1.0, 1.0, size=D) if WITH_BIAS else np.zeros(D)

    print(
        f"TD(0) controlled-EV experiment  |  d={D}, γ={GAMMA}, α={ALPHA:.4f}, "
        f"batch={BATCH_SIZE}, σ={STD_DEV_NOISE}, bias={WITH_BIAS}, "
        f"trials={NB_TRIALS}, iters={NB_ITER:,}"
    )

    # Each value of k sets the singular-value spread of Φ.
    # ω is the smallest eigenvalue of Σ and serves as the x-axis label in plots.
    k_values = [0, 1, 2, 3, 4]
    omegas, mse_per_k = [], []

    for k in k_values:
        # Singular values of Φ: [1, 10^{-k/2}, 10^{-k/2}]
        diag_Sigma_half = np.array([1.0, 10.0 ** (-0.5 * k), 10.0 ** (-0.5 * k)])

        # Smallest eigenvalue of Σ = Φᵀ D Φ  (D = I/d for bistochastic P)
        omega = 10.0 ** (-k) / D
        omegas.append(omega)

        mean_mse = compute_mean_mse_controlled_ev(
            num_states=D, gamma=GAMMA, alpha=ALPHA, std_dev_noise=STD_DEV_NOISE,
            initial_theta=initial_theta, dirichlet_param=DIRICHLET_PARAM,
            diag_Sigma_half=diag_Sigma_half, nb_iter=NB_ITER,
            nb_trials=NB_TRIALS, mse_checkpoints=MSE_CHECKPOINTS,
            batch_size=BATCH_SIZE,
        )
        mse_per_k.append(mean_mse)
        print(f"  k={k}  (ω={omega:.2e})  done")

    dt = time.perf_counter() - t0
    print(f"Total computation time: {dt:.1f} s")

    # Stack to shape (len(k_values), len(MSE_CHECKPOINTS), 3)
    mse_array = np.array(mse_per_k)

    results = {
        "gamma":                 GAMMA,
        "alpha":                 ALPHA,
        "batch_size":            BATCH_SIZE,
        "dimension":             D,
        "nb_iter":               NB_ITER,
        "nb_trials":             NB_TRIALS,
        "with_bias":             WITH_BIAS,
        "std_dev_noise":         STD_DEV_NOISE,
        "dirichlet_param":       DIRICHLET_PARAM,
        "mse_checkpoints":       MSE_CHECKPOINTS,
        "MSE_values":            mse_array[:, :, 0],
        "MSE_constant":          mse_array[:, :, 1],
        "MSE_non_constant":      mse_array[:, :, 2],
        "omegas":                omegas,
        "start_plot":            START_PLOT,
        "nb_mse_computations":   NB_MSE_CHECKPOINTS,
        "seed":                  SEED,
        "computation_time_seconds": dt,
    }

    savepath = (
        SAVE_DIR
        / f"CEV_TD0_d{D}_gamma_{GAMMA}_alpha_{ALPHA}"
          f"{'_WB' if WITH_BIAS else '_WoutB'}"
          f"{'_WV' if STD_DEV_NOISE > 0 else '_WoutV'}"
          f"_B{BATCH_SIZE}"
          ".npz"
    )
    np.savez(savepath, **results)
    print(f"Saved → {savepath}")


if __name__ == "__main__":
    main()

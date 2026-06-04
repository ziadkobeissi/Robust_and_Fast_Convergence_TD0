"""
TD(0) simulation with linear function approximation.

Reads all parameters from parameters_simulations.py.
Results are saved as a .npz archive in experiments/td0/.

Metrics saved:
    MSE_values        — E_ρ[(V_θ(s) − V_θ*(s))²]
    MSE_constant      — (μᵀ(θ − θ*))²
    MSE_non_constant  — MSE_values − MSE_constant
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parametrisation       import RandomLinearParametrisation
from markov_reward_process import MarkovRewardProcess
from td0                   import StandardTD0LinearFA
from matrix_generators     import generate_stochastic

import parameters_simulations as P


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT    = Path(__file__).resolve().parent.parent.parent
_TD0_DIR = _ROOT / "experiments" / "td0"

_SUBDIR = {
    (True,  True):  "total_MSE",
    (True,  False): "MSE_bias_term_only",
    (False, True):  "MSE_variance_term_only",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate flags ---
    assert P.WITH_BIAS_TERM or P.WITH_VARIANCE_TERM, \
        "WITH_BIAS_TERM and WITH_VARIANCE_TERM cannot both be False."
    #assert P.BATCH_SIZE % 2 == 0, "BATCH_SIZE must be even."

    subdir = _SUBDIR[(P.WITH_BIAS_TERM, P.WITH_VARIANCE_TERM)]
    save_dir = _TD0_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Effective simulation parameters from flags ---
    expected_reward = np.ones(P.N) if P.WITH_BIAS_TERM else np.zeros(P.N)
    initial_theta   = np.random.uniform(-1.0, 1.0, P.D) if P.WITH_BIAS_TERM else np.zeros(P.D)

    if P.WITH_VARIANCE_TERM:
        std_dev_noise = P.STD_DEV_NOISE
    else:
        if P.STD_DEV_NOISE != 0.0:
            print(f"[td0] WITH_VARIANCE_TERM=False: STD_DEV_NOISE={P.STD_DEV_NOISE} is ignored, using 0.")
        std_dev_noise = 0.0

    t0 = time.perf_counter()
    np.random.seed(P.SEED)

    # --- MRP ---
    trans_mat = generate_stochastic(P.N, dirichlet_param=P.DIRICHLET_PARAM)
    MRP = MarkovRewardProcess(trans_mat, expected_reward, std_dev_noise)

    # --- Learning rate ---
    if P.USE_SPECTRAL_GAP_ALPHA:
        effective_gamma = P.GAMMA * (1.0 - MRP.spectral_gap)
    else:
        effective_gamma = P.GAMMA

    alpha_0 = (2.0 * P.BATCH_SIZE * (1.0 - effective_gamma) / (1.0 + effective_gamma) 
               /(1.0 + effective_gamma + (1.0 - effective_gamma) * (P.BATCH_SIZE - 1.0)))
    alpha_1 = 2.0 / (1.0 + effective_gamma)

    alpha_scalar = alpha_0 / 2.
    
    # Shape (1, D): one coupled trajectory, uniform rate across features
    alpha = np.full((1, P.D), alpha_scalar)

    print(
        f"TD(0) [{subdir}]  |  N={P.N}, D={P.D}, γ={P.GAMMA}, "
        f"α={alpha_scalar:.4f}, batch={P.BATCH_SIZE} transitions/step, "
        f"σ={std_dev_noise}, trials={P.NB_TRIALS}, iters={P.NB_ITER:,}"
    )

    # --- Parametrisation ---
    param = RandomLinearParametrisation(
        num_states=P.N,
        num_features=P.D,
        initial_theta=initial_theta,
        num_coupled_trajectories=1,
        num_independent_copies=P.NB_TRIALS,
        do_average=True,
        first_constant_feature=P.FIRST_CONSTANT_FEATURE,
    )

    # --- Run ---
    agent = StandardTD0LinearFA(MRP, param,
                                gamma_vect=np.array([P.GAMMA]),
                                batch_size=P.BATCH_SIZE)
    _, mse_avg = agent.train(
        P.NB_ITER, alpha,
        compute_mse_averaged=True,
        mse_checkpoints=P.MSE_CHECKPOINTS,
    )
    # mse_avg shape: (len(checkpoints), 3, 1, NB_TRIALS)
    # Average over trials → (len(checkpoints), 3, 1) → squeeze to (len(checkpoints), 3)
    mse = np.mean(mse_avg, axis=-1)[:, :, 0]

    dt = time.perf_counter() - t0
    print(f"Done  ({dt:.1f} s)")

    # --- Save ---
    results = {
        "gamma":              P.GAMMA,
        "alpha":              alpha_scalar,
        "batch_size":         P.BATCH_SIZE,
        "num_states":         P.N,
        "num_features":       P.D,
        "nb_iter":            P.NB_ITER,
        "nb_trials":          P.NB_TRIALS,
        "std_dev_noise":      std_dev_noise,
        "dirichlet_param":    P.DIRICHLET_PARAM,
        "mse_checkpoints":    P.MSE_CHECKPOINTS,
        "seed":               P.SEED,
        "first_constant_feature": P.FIRST_CONSTANT_FEATURE,
        "with_bias_term":     P.WITH_BIAS_TERM,
        "with_variance_term": P.WITH_VARIANCE_TERM,
        # Axis-1: 0 → mse_values, 1 → mse_constant, 2 → mse_non_constant
        "MSE_values":         mse[:, 0],
        "MSE_constant":       mse[:, 1],
        "MSE_non_constant":   mse[:, 2],
        "computation_time_seconds": dt,
    }

    savepath = save_dir / f"td0_N{P.N}_D{P.D}_gamma_{P.GAMMA}_B{P.BATCH_SIZE}.npz"
    np.savez(savepath, **results)
    print(f"Saved → {savepath}")


if __name__ == "__main__":
    main()

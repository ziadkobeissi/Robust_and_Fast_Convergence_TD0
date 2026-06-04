"""
PCTD0 (Pairwise-Centered TD(0)) simulation with four coupled trajectories
at different discount factors.

All parameters are read from parameters_simulations.py.

The four γ values are:
    γ₁ = 0.9
    γ₂ = 0.99
    γ₃ = 1.0
    γ₄ = (2 − 2g) / (2 − g)    where g = spectral gap of the MRP

All trajectories share the same MRP, feature matrix, and sampled transitions
at every step (genuine coupling via num_coupled_trajectories=4).
Each trajectory gets its own learning rate derived from the formula in
compare_pctd0_and_td0.py; spectral-gap correction is forced for γ=1 to
avoid division by zero.

Note: for γ=1, MSE_constant and MSE_with_constant_part involve (1−γ)⁻¹
and are therefore numerically infinite/NaN — only MSE_values is meaningful.

Metrics saved (each of shape (len(checkpoints), 4)):
    MSE_values             — centred value MSE, column i = trajectory i
    MSE_constant           — ((learned_r̄ − r̄) / (1−γ))²
    MSE_with_constant_part — full error including the constant component
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parametrisation       import RandomLinearParametrisation
from markov_reward_process import MarkovRewardProcess
from td0                   import PCTD0LinearFA
from matrix_generators     import generate_stochastic

import parameters_simulations as P


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT      = Path(__file__).resolve().parent.parent.parent
_PCTD0_DIR = _ROOT / "experiments" / "pctd0"

_SUBDIR = {
    (True,  True):  "total_MSE",
    (True,  False): "MSE_bias_term_only",
    (False, True):  "MSE_variance_term_only",
}

NUM_GAMMAS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_alpha(gamma: float, spectral_gap: float) -> float:
    """Learning rate for one gamma using the compare_pctd0_and_td0.py formula.

    Spectral-gap correction is forced when gamma >= 1 to keep effective_gamma
    strictly below 1 and avoid division by zero.
    """
    if gamma >= 1.0 or P.USE_SPECTRAL_GAP_ALPHA:
        eg = gamma * (1.0 - spectral_gap)
    else:
        eg = gamma
    alpha_0 = 2.0/(1.0 + eg) / (1.0 + 2.0*eg/(1.0 - eg)/P.BATCH_SIZE)
    return alpha_0 / 2.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate ---
    assert P.WITH_BIAS_TERM or P.WITH_VARIANCE_TERM, \
        "WITH_BIAS_TERM and WITH_VARIANCE_TERM cannot both be False."
    assert P.BATCH_SIZE % 2 == 0, "BATCH_SIZE must be even."

    subdir   = _SUBDIR[(P.WITH_BIAS_TERM, P.WITH_VARIANCE_TERM)]
    save_dir = _PCTD0_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    pctd0_batch = P.BATCH_SIZE // 2

    # --- Effective parameters from flags ---
    expected_reward = np.ones(P.N) if P.WITH_BIAS_TERM else np.zeros(P.N)
    initial_theta   = np.random.uniform(-1.0, 1.0, P.D) if P.WITH_BIAS_TERM else np.zeros(P.D)

    if P.WITH_VARIANCE_TERM:
        std_dev_noise = P.STD_DEV_NOISE
    else:
        if P.STD_DEV_NOISE != 0.0:
            print(f"[pctd0] WITH_VARIANCE_TERM=False: STD_DEV_NOISE={P.STD_DEV_NOISE} is ignored, using 0.")
        std_dev_noise = 0.0

    t0 = time.perf_counter()
    np.random.seed(P.SEED)

    # --- MRP ---
    trans_mat = generate_stochastic(P.N, dirichlet_param=P.DIRICHLET_PARAM)
    MRP = MarkovRewardProcess(trans_mat, expected_reward, std_dev_noise)

    # --- Parametrisation (4 coupled trajectories) ---
    param = RandomLinearParametrisation(
        num_states=P.N,
        num_features=P.D,
        initial_theta=initial_theta,
        num_coupled_trajectories=NUM_GAMMAS,
        num_independent_copies=P.NB_TRIALS,
        do_average=True,
        first_constant_feature=P.FIRST_CONSTANT_FEATURE,
    )

    # --- Agent ---
    gamma4 = 1.15
    gammas       = np.array([0.9, 0.99, 1.0, gamma4])
    agent = PCTD0LinearFA(MRP, param,
                          gamma_vect=gammas,
                          batch_size=pctd0_batch)

    spectral_gap = MRP.spectral_gap
    projected_spectral_gap = agent.projected_spectral_gap if P.FIRST_CONSTANT_FEATURE else spectral_gap
    #if gamma4 >= 1.0/(1.0 - projected_spectral_gap):
    #    raise ValueError(f"Gamma4={gamma4:.4f} is too large for the projected spectral gap {projected_spectral_gap:.4f}; it may cause numerical issues.")

    # --- Learning rates: one per trajectory ---
    alphas = np.array([compute_alpha(g, spectral_gap) for g in gammas])
    #alphas = 0.5 * np.ones_like(gammas) 
    alpha  = np.outer(alphas, np.ones(P.D))  # (4, D)

    print(
        f"PCTD0 multi-γ [{subdir}]  |  N={P.N}, D={P.D}, "
        f"spectral_gap={spectral_gap:.4f}, γ₄={gamma4:.4f}, "
        f"{pctd0_batch} pairs ({P.BATCH_SIZE} transitions)/step, "
        f"σ={std_dev_noise}, trials={P.NB_TRIALS}, iters={P.NB_ITER:,}"
    )
    for g, a in zip(gammas, alphas):
        print(f"  γ={g:.6f}  α={a:.6f}")

    # --- Run ---
    _, mse_avg = agent.train(
        P.NB_ITER, alpha,
        compute_mse_averaged=True,
        mse_checkpoints=P.MSE_CHECKPOINTS,
    )
    # mse_avg: (len(checkpoints), 3, NUM_GAMMAS, NB_TRIALS)
    # Average over trials → (len, 3, NUM_GAMMAS)
    mse = np.mean(mse_avg, axis=-1)

    dt = time.perf_counter() - t0
    print(f"Done  ({dt:.1f} s)")

    # --- Save ---
    results = {
        "gammas":             gammas,          # (4,)
        "alphas":             alphas,          # (4,) one per gamma
        "spectral_gap":       spectral_gap,
        "batch_size":         P.BATCH_SIZE,
        "batch_size_pairs":   pctd0_batch,
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
        # Each array has shape (len(checkpoints), NUM_GAMMAS); column i = trajectory i
        "MSE_values":             mse[:, 0, :],
        "MSE_constant":           mse[:, 1, :],
        "MSE_with_constant_part": mse[:, 2, :],
        "computation_time_seconds": dt,
    }

    savepath = save_dir / f"pctd0_multi_gamma_N{P.N}_D{P.D}_B{P.BATCH_SIZE}.npz"
    np.savez(savepath, **results)
    print(f"Saved → {savepath}")


if __name__ == "__main__":
    main()

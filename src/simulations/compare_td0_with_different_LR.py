"""
Experiment: compare two TD(0) learning-rate schedules.

All parameters are read from parameters_simulations.py.

Both settings use the same Markov chain, reward, and feature matrix
(RandomLinearParametrisation with a constant first feature φ₁(s) = 1).
They start from the same θ₀ and are coupled step-by-step through the
num_coupled_trajectories mechanism: at every iteration both copies receive
the same sampled transitions and rewards, so their difference is due
solely to the learning-rate schedule.

Setting 1 — uniform:   α_t = α · 1_d  (one rate for all d feature dimensions)
Setting 2 — split:     α_t = [α_first, α_rest, …, α_rest]
                              (dedicated rate for the constant feature,
                               a common rate for the remaining features)

α_first_coord = 1 / (4 (1 − γ)) — fixed regardless of batch size.
α_other_coords (= α_uniform) is given by the same formula as in
compare_pctd0_and_td0.py when BATCH_SIZE > 1; falls back to
(1 − γ_eff) / 4 when BATCH_SIZE = 1.

Results are saved as a .npz archive in experiments/data/ and visualised
by draw_compare_td0_with_different_LR.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

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
SAVE_DIR = _ROOT / "experiments" / "data"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate ---
    assert P.WITH_BIAS_TERM or P.WITH_VARIANCE_TERM, \
        "WITH_BIAS_TERM and WITH_VARIANCE_TERM cannot both be False."

    # --- Effective parameters from flags ---
    expected_reward = np.ones(P.N) if P.WITH_BIAS_TERM else np.zeros(P.N)
    initial_theta   = np.random.uniform(-1., 1., P.D) if P.WITH_BIAS_TERM else np.zeros(P.D)

    if P.WITH_VARIANCE_TERM:
        std_dev_noise = P.STD_DEV_NOISE
    else:
        if P.STD_DEV_NOISE != 0.0:
            print(f"[compare_LR] WITH_VARIANCE_TERM=False: STD_DEV_NOISE={P.STD_DEV_NOISE} ignored, using 0.")
        std_dev_noise = 0.0

    t0 = time.perf_counter()
    np.random.seed(P.SEED)

    # --- MRP ---
    trans_mat = generate_stochastic(P.N, dirichlet_param=P.DIRICHLET_PARAM)
    MRP = MarkovRewardProcess(trans_mat, expected_reward, std_dev_noise)

    # --- Learning rates ---
    if P.USE_SPECTRAL_GAP_ALPHA:
        effective_gamma = P.GAMMA * (1.0 - MRP.spectral_gap)
    else:
        effective_gamma = P.GAMMA

    # α for the constant feature: unchanged regardless of batch size
    alpha_first_coord = 1.0 / (4.0 * (1.0 - P.GAMMA))

    # α for all other features (= α_uniform): same formula as compare_pctd0_and_td0.py
    if P.BATCH_SIZE > 1:
        alpha_0 = 2.0/(1.0 + effective_gamma) / (1.0 + 2.0*effective_gamma/(1.0 - effective_gamma)/(P.BATCH_SIZE - 1.0))
        alpha_other_coords = alpha_0 / 2.0
    else:
        alpha_other_coords = (1.0 - effective_gamma) / 4.0
    alpha_uniform = alpha_other_coords

    print(
        f"Compare LR schedules  |  N={P.N}, D={P.D}, γ={P.GAMMA}, "
        f"α_uniform={alpha_uniform:.4f}, "
        f"α_first={alpha_first_coord:.4f}, α_rest={alpha_other_coords:.4f}, "
        f"batch={P.BATCH_SIZE}, bias={P.WITH_BIAS_TERM}, variance={P.WITH_VARIANCE_TERM}, "
        f"trials={P.NB_TRIALS}, iters={P.NB_ITER:,}"
    )

    # Learning-rate matrix: shape (2, D)
    #   row 0 → setting 1 (uniform)
    #   row 1 → setting 2 (split)
    alpha = np.array([
        [alpha_uniform]     * P.D,
        [alpha_first_coord] + [alpha_other_coords] * (P.D - 1),
    ])

    # One parametrisation with 2 coupled copies — they share the same
    # random transitions at every step (genuine coupling)
    param = RandomLinearParametrisation(
        num_states=P.N,
        num_features=P.D,
        initial_theta=initial_theta,
        num_coupled_trajectories=2,
        num_independent_copies=P.NB_TRIALS,
        do_average=True,
        first_constant_feature=P.FIRST_CONSTANT_FEATURE,
    )

    agent = StandardTD0LinearFA(MRP, param,
                                gamma_vect=np.array([P.GAMMA]),
                                batch_size=P.BATCH_SIZE)
    _, mse_avg = agent.train(
        P.NB_ITER, alpha,
        compute_mse_averaged=True,
        mse_checkpoints=P.MSE_CHECKPOINTS,
    )
    # mse_avg shape: (len(checkpoints), 3, 2, NB_TRIALS)
    # Average over trials (last axis) → (len(checkpoints), 3, 2)
    mse = np.mean(mse_avg, axis=-1)

    dt = time.perf_counter() - t0
    print(f"Total computation time: {dt:.1f} s")

    results = {
        "gamma":               P.GAMMA,
        "batch_size":          P.BATCH_SIZE,
        "alpha_uniform":       alpha_uniform,
        "alpha_first_coord":   alpha_first_coord,
        "alpha_other_coords":  alpha_other_coords,
        "num_states":          P.N,
        "num_features":        P.D,
        "nb_iter":             P.NB_ITER,
        "nb_trials":           P.NB_TRIALS,
        "std_dev_noise":       std_dev_noise,
        "dirichlet_param":     P.DIRICHLET_PARAM,
        "mse_checkpoints":     P.MSE_CHECKPOINTS,
        "seed":                P.SEED,
        "first_constant_feature": P.FIRST_CONSTANT_FEATURE,
        "with_bias_term":      P.WITH_BIAS_TERM,
        "with_variance_term":  P.WITH_VARIANCE_TERM,
        # Axis-2 index 0 → setting 1 (uniform), index 1 → setting 2 (split)
        # Axis-1: 0 → mse_values, 1 → mse_constant, 2 → mse_non_constant
        "MSE_values_uniform":       mse[:, 0, 0],
        "MSE_constant_uniform":     mse[:, 1, 0],
        "MSE_non_constant_uniform": mse[:, 2, 0],
        "MSE_values_split":         mse[:, 0, 1],
        "MSE_constant_split":       mse[:, 1, 1],
        "MSE_non_constant_split":   mse[:, 2, 1],
        "computation_time_seconds": dt,
    }

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    savepath = SAVE_DIR / f"compare_LR_N{P.N}_D{P.D}_gamma_{P.GAMMA}_B{P.BATCH_SIZE}.npz"
    np.savez(savepath, **results)
    print(f"Saved → {savepath}")


if __name__ == "__main__":
    main()

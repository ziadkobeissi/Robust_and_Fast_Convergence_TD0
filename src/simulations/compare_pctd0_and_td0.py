"""
Experiment: compare Standard TD(0) and Pairwise-Centered TD(0) (PCTD0).

All parameters are read from parameters_simulations.py.

Both algorithms operate on the same MRP and feature matrix.
They start from the same θ₀ and use the same learning rate α.
The two agents run sequentially and cannot share a param object because
their update rules differ.

Optional third curve (RUN_TD0_SPLIT=True): a second TD(0) instance that
uses the split learning-rate schedule from compare_td0_with_different_LR.py,
i.e. α_first_coord for the constant feature and α_scalar for all other features.
Both TD(0) variants share the same parametrisation object (num_coupled_trajectories=2)
so they receive identical transitions at every step.

Metrics saved:
    TD0 uniform: MSE_values, MSE_constant, MSE_non_constant
    TD0 split  : MSE_values, MSE_constant, MSE_non_constant  (if RUN_TD0_SPLIT)
    PCTD0      : MSE_values, MSE_constant, MSE_with_constant_part

Results are saved as a .npz archive in experiments/data/ and visualised
by draw_compare_pctd0_and_td0.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parametrisation       import RandomLinearParametrisation
from markov_reward_process import MarkovRewardProcess
from td0                   import StandardTD0LinearFA, PCTD0LinearFA
from matrix_generators     import generate_stochastic

import parameters_simulations as P


# ---------------------------------------------------------------------------
# Script-local flags  (not in parameters_simulations — specific to this comparison)
# ---------------------------------------------------------------------------

# If True, TD0 runs first (so its random seed is identical to a standalone td0_simulations.py run).
# If False, only PCTD0 is run.
RUN_TD0_FIRST = True

# If True (requires RUN_TD0_FIRST), run a second TD(0) coupled trajectory
# with the split learning rate: α_first_coord for φ₁, α_scalar for all others.
RUN_TD0_SPLIT = True

_ROOT    = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = _ROOT / "experiments" / "data"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate ---
    assert P.WITH_BIAS_TERM or P.WITH_VARIANCE_TERM, \
        "WITH_BIAS_TERM and WITH_VARIANCE_TERM cannot both be False."
    assert P.BATCH_SIZE % 2 == 0, "BATCH_SIZE must be even."
    assert not (RUN_TD0_SPLIT and not RUN_TD0_FIRST), \
        "RUN_TD0_SPLIT requires RUN_TD0_FIRST=True."
    pctd0_batch = P.BATCH_SIZE // 2

    # --- Effective parameters from flags ---
    expected_reward = np.ones(P.N) if P.WITH_BIAS_TERM else np.zeros(P.N)
    initial_theta   = np.random.uniform(-1., 1., P.D) if P.WITH_BIAS_TERM else np.zeros(P.D)

    if P.WITH_VARIANCE_TERM:
        std_dev_noise = P.STD_DEV_NOISE
    else:
        if P.STD_DEV_NOISE != 0.0:
            print(f"[compare] WITH_VARIANCE_TERM=False: STD_DEV_NOISE={P.STD_DEV_NOISE} ignored, using 0.")
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

    alpha_0 = 2.0/(1.0 + effective_gamma) / (1.0 + 2.0*effective_gamma/(1.0 - effective_gamma)/P.BATCH_SIZE)
    alpha_scalar = alpha_0 / 2.

    # Split-LR: dedicated rate for the constant first feature
    alpha_first_coord = 1.0 / (4.0 * (1.0 - P.GAMMA))

    # TD0 alpha: (1, D) for uniform only, (2, D) for uniform + split
    num_coupled_td0 = 2 if (RUN_TD0_FIRST and RUN_TD0_SPLIT) else 1
    if RUN_TD0_FIRST and RUN_TD0_SPLIT:
        alpha_td0 = np.array([
            [alpha_scalar]     * P.D,
            [alpha_first_coord] + [alpha_scalar] * (P.D - 1),
        ])  # (2, D)
    else:
        alpha_td0 = np.full((1, P.D), alpha_scalar)  # (1, D)

    # PCTD0 always uses the uniform rate
    alpha_pctd0 = np.full((1, P.D), alpha_scalar)

    print(
        f"Compare TD0 vs PCTD0  |  N={P.N}, D={P.D}, γ={P.GAMMA}, "
        f"α={alpha_scalar:.4f}, α_first={alpha_first_coord:.4f}, "
        f"{P.BATCH_SIZE} transitions/step, "
        f"bias={P.WITH_BIAS_TERM}, variance={P.WITH_VARIANCE_TERM}, "
        f"td0_split={RUN_TD0_SPLIT}, trials={P.NB_TRIALS}, iters={P.NB_ITER:,}"
    )

    # --- TD0 parametrisation ---
    param_td0 = RandomLinearParametrisation(
        num_states=P.N,
        num_features=P.D,
        initial_theta=initial_theta,
        num_coupled_trajectories=num_coupled_td0,
        num_independent_copies=P.NB_TRIALS,
        do_average=True,
        first_constant_feature=P.FIRST_CONSTANT_FEATURE,
    )

    # --- PCTD0 parametrisation: separate θ, same feature matrix ---
    param_pctd0 = RandomLinearParametrisation(
        num_states=P.N,
        num_features=P.D,
        initial_theta=initial_theta,
        num_coupled_trajectories=1,
        num_independent_copies=P.NB_TRIALS,
        do_average=True,
        first_constant_feature=P.FIRST_CONSTANT_FEATURE,
    )
    param_pctd0.feature_matrix = param_td0.feature_matrix.copy()

    # --- Run TD0 ---
    if RUN_TD0_FIRST:
        t1 = time.perf_counter()
        agent_td0 = StandardTD0LinearFA(MRP, param_td0,
                                        gamma_vect=np.array([P.GAMMA]),
                                        batch_size=P.BATCH_SIZE)
        _, mse_avg_td0 = agent_td0.train(
            P.NB_ITER, alpha_td0,
            compute_mse_averaged=True,
            mse_checkpoints=P.MSE_CHECKPOINTS,
        )
        print(f"  TD0  done  ({time.perf_counter() - t1:.1f} s)")
        # mse_avg_td0: (len(checkpoints), 3, num_coupled_td0, NB_TRIALS) → mean over trials
        mse_td0_all = np.mean(mse_avg_td0, axis=-1)   # (len, 3, num_coupled_td0)
        mse_td0     = mse_td0_all[:, :, 0]             # uniform: (len, 3)
        if RUN_TD0_SPLIT:
            mse_td0_split = mse_td0_all[:, :, 1]       # split:   (len, 3)

    # --- Run PCTD0 ---
    t2 = time.perf_counter()
    agent_pctd0 = PCTD0LinearFA(MRP, param_pctd0,
                                gamma_vect=np.array([P.GAMMA]),
                                batch_size=pctd0_batch)
    _, mse_avg_pctd0 = agent_pctd0.train(
        P.NB_ITER, alpha_pctd0,
        compute_mse_averaged=True,
        mse_checkpoints=P.MSE_CHECKPOINTS,
    )
    print(f"  PCTD0 done  ({time.perf_counter() - t2:.1f} s)")
    mse_pctd0 = np.mean(mse_avg_pctd0, axis=-1)[:, :, 0]

    dt = time.perf_counter() - t0
    print(f"Total computation time: {dt:.1f} s")

    # --- Save ---
    results = {
        "gamma":              P.GAMMA,
        "alpha":              alpha_scalar,
        "alpha_first_coord":  alpha_first_coord,
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
        "run_td0_first":      RUN_TD0_FIRST,
        "run_td0_split":      RUN_TD0_SPLIT,
        "MSE_values_pctd0":             mse_pctd0[:, 0],
        "MSE_constant_pctd0":           mse_pctd0[:, 1],
        "MSE_with_constant_part_pctd0": mse_pctd0[:, 2],
        "computation_time_seconds": dt,
    }
    if RUN_TD0_FIRST:
        results.update({
            "MSE_values_td0":       mse_td0[:, 0],
            "MSE_constant_td0":     mse_td0[:, 1],
            "MSE_non_constant_td0": mse_td0[:, 2],
        })
        if RUN_TD0_SPLIT:
            results.update({
                "MSE_values_td0_split":       mse_td0_split[:, 0],
                "MSE_constant_td0_split":     mse_td0_split[:, 1],
                "MSE_non_constant_td0_split": mse_td0_split[:, 2],
            })

    savepath = SAVE_DIR / f"compare_pctd0_td0_N{P.N}_D{P.D}_gamma_{P.GAMMA}_B{P.BATCH_SIZE}.npz"
    np.savez(savepath, **results)
    print(f"Saved → {savepath}")


if __name__ == "__main__":
    main()

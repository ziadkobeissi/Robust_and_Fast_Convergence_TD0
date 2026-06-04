"""
High-level helpers that assemble a full TD(0) experiment and return mean MSE.

Each function builds an MRP, a parametrisation, a StandardTD0LinearFA agent, runs
training, and returns the averaged MSE array — hiding the boilerplate from
experiment scripts.
"""

import sys
from pathlib import Path

import numpy as np

# Core modules live one level up in src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markov_reward_process import MarkovRewardProcess
from parametrisation       import DiagonalParametrisation, LinearParametrisation
from td0                   import StandardTD0LinearFA
from matrix_generators     import generate_bistochastic, generate_stochastic


def compute_mean_mse_controlled_ev(
    num_states: int, gamma: float, alpha: float, std_dev_noise: float,
    initial_theta: np.ndarray, dirichlet_param: float,
    diag_Sigma_half: np.ndarray, nb_iter: int, nb_trials: int,
    mse_checkpoints: list[int], batch_size: int = 1,
) -> np.ndarray:
    """Run a single TD(0) experiment with a controlled feature singular-value spectrum.

    Uses a bistochastic P (uniform stationary distribution ρ = 1/num_states · 1) so
    that the singular-value spectrum of Φ directly controls the convergence
    behaviour via the feature second-moment matrix Σ = Φᵀ D Φ = (1/num_states) ΦᵀΦ.

    The nb_trials parallel copies are averaged to estimate the mean MSE.

    Parameters
    ----------
    batch_size : mini-batch size for the TD(0) update (default 1 = standard TD(0)).

    Returns
    -------
    mse_averaged_mean : (len(mse_checkpoints), 3) ndarray
        Mean over nb_trials of (mse_values, mse_constant, mse_non_constant).
    """
    np.random.seed(1)
    U   = generate_bistochastic(num_states, dirichlet_param=dirichlet_param)
    MRP = MarkovRewardProcess(U, np.zeros(num_states), std_dev_noise)

    param            = DiagonalParametrisation(diag_Sigma_half, initial_theta,
                                               num_coupled_trajectories=1,
                                               num_independent_copies=nb_trials)
    param.do_average = True

    agent = StandardTD0LinearFA(MRP, param, gamma_vect=np.array([gamma]),
                                batch_size=batch_size)
    _, mse_avg = agent.train(nb_iter, np.array([alpha]),
                             compute_mse_averaged=True,
                             mse_checkpoints=mse_checkpoints)

    # mse_avg shape: (len(checkpoints), 3, 1, nb_trials) → average over trials
    return np.mean(mse_avg, axis=-1).reshape(-1, 3)


def compute_mean_mse_gamma_sweep(
    num_states: int, gamma_vect: np.ndarray, alpha_vect: np.ndarray, std_dev_noise: float,
    initial_theta: np.ndarray, dirichlet_param: float,
    diag_Sigma_half: np.ndarray, nb_iter: int, num_independent_copies: int,
    mse_checkpoints: list[int], batch_size: int = 1,
) -> np.ndarray:
    """Run TD(0) simultaneously for multiple γ values and average MSE over copies.

    One coupled trajectory is used per γ value (num_coupled_trajectories = len(gamma_vect)),
    so all γ values share the same random transitions at each step.

    Parameters
    ----------
    batch_size : mini-batch size for the TD(0) update (default 1 = standard TD(0)).

    Returns
    -------
    mse : (len(mse_checkpoints), 3, len(gamma_vect)) ndarray
    """
    np.random.seed(1)
    U   = generate_stochastic(num_states, dirichlet_param=dirichlet_param)
    MRP = MarkovRewardProcess(U, np.zeros(num_states), std_dev_noise)

    param            = DiagonalParametrisation(diag_Sigma_half, initial_theta,
                                               num_coupled_trajectories=gamma_vect.shape[0],
                                               num_independent_copies=num_independent_copies)
    param.do_average = True

    agent = StandardTD0LinearFA(MRP, param, gamma_vect=gamma_vect,
                                batch_size=batch_size)
    _, mse_avg = agent.train(nb_iter, alpha_vect,
                             compute_mse_averaged=True,
                             mse_checkpoints=mse_checkpoints)

    # mse_avg: (len(checkpoints), 3, num_coupled_trajectories, num_independent_copies) → average over copies
    return np.mean(mse_avg, axis=-1)


def compute_mean_mse_random_instances(
    num_states: int, num_features: int,
    gamma_vect: np.ndarray, alpha_vect: np.ndarray,
    expected_reward: np.ndarray, std_dev_noise: float,
    initial_theta: np.ndarray, dirichlet_param: float,
    nb_iter: int, num_independent_copies: int,
    mse_checkpoints: list[int], nb_trials: int,
    batch_size: int = 1,
) -> np.ndarray:
    """Average TD(0) MSE over multiple independent random (P, Φ) pairs.

    Each trial draws a fresh stochastic transition matrix and a fresh
    Dirichlet-distributed feature matrix, enabling analysis of typical-case
    convergence across problem instances.

    Returns
    -------
    mse_array : (nb_trials, len(mse_checkpoints), 3, len(gamma_vect), num_independent_copies)
    """
    mse_trials = []
    for trial in range(nb_trials):
        np.random.seed(trial)
        U   = generate_stochastic(num_states, dirichlet_param=dirichlet_param)
        MRP = MarkovRewardProcess(U, expected_reward, std_dev_noise)

        param = LinearParametrisation(
            num_states=num_states, num_features=num_features,
            initial_theta=initial_theta,
            num_coupled_trajectories=gamma_vect.shape[0],
            num_independent_copies=num_independent_copies,
            do_average=True,
        )
        # Each column is a feature function on states; rows are Dirichlet-distributed
        param.feature_matrix = np.random.dirichlet(
            dirichlet_param * np.ones(num_states), size=num_features).T
        print(f"[Trial {trial}] feature matrix shape: {param.feature_matrix.shape}")

        agent = StandardTD0LinearFA(MRP, param, gamma_vect=gamma_vect,
                                    batch_size=batch_size)
        print(f"[Trial {trial}] largest eigenvalue of Σ: "
              f"{np.linalg.eig(agent.Sigma)[0].max():.4f}")

        _, mse_avg = agent.train(nb_iter, alpha_vect,
                                 compute_mse_averaged=True,
                                 mse_checkpoints=mse_checkpoints)
        mse_trials.append(mse_avg)

    mse_array = np.array(mse_trials)
    print(f"mse_array shape: {mse_array.shape}")
    return mse_array

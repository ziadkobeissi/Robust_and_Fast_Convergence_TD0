"""
Markov Reward Process.

Notation
--------
    num_states : number of states
    P   : (num_states, num_states) stochastic transition matrix
    ρ   : (num_states,)   stationary distribution  (left eigenvector for λ=1)
    D   : diag(ρ)
    r   : (num_states,)   deterministic expected-reward vector
    σ   : standard deviation of i.i.d. Gaussian reward noise
"""

import numpy as np


class MarkovRewardProcess:
    """
    Finite Markov reward process defined by a stochastic transition matrix,
    a deterministic expected-reward vector r(s), and i.i.d. Gaussian reward noise.

    Attributes
    ----------
    stat_dist    : (num_states,) stationary distribution ρ
    spectral_gap : 1 − σ₂  where σ₂ is the second singular value of D^{1/2} P D^{-1/2}
    """

    def __init__(self, transition_matrix: np.ndarray, expected_reward: np.ndarray,
                 std_dev_noise: float = 0.1):
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("transition_matrix rows must sum to 1 (not a stochastic matrix).")

        self.transition_matrix = transition_matrix
        # Pre-compute row CDFs for fast vectorised inverse-CDF sampling
        self.cdf_transition_matrix = np.cumsum(transition_matrix, axis=1)
        self.num_states    = transition_matrix.shape[0]
        self.expected_reward = expected_reward
        self.std_dev_noise = std_dev_noise

        # Stationary distribution: normalised left eigenvector for eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
        stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1.0)][:, 0])
        self.stat_dist            = stat_dist / stat_dist.sum()
        self.cumulative_stat_dist = np.cumsum(self.stat_dist)

        self.spectral_gap = self._compute_spectral_gap()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_transition(self, current_states: np.ndarray) -> np.ndarray:
        """Vectorised inverse-CDF sampling: draw the next state for each copy.

        Parameters
        ----------
        current_states : (n,) int array

        Returns
        -------
        next_states : (n,) int array
        """
        return (
            self.cdf_transition_matrix[current_states]
            >= np.random.rand(current_states.shape[0], 1)
        ).argmax(axis=1).astype(np.int64)

    def sample_reward(self, states: np.ndarray) -> np.ndarray:
        """Sample noisy scalar rewards  R(s) ~ N(r(s), σ²).

        Parameters
        ----------
        states : (n,) int array

        Returns
        -------
        rewards : (n,) float array
        """
        return (
            self.expected_reward[states]
            + self.std_dev_noise * np.random.randn(states.shape[0])
        )

    def sample_stationary(self, n: int) -> np.ndarray:
        """Draw n i.i.d. states from the stationary distribution ρ.

        Returns
        -------
        states : (n,) int array
        """
        return (
            self.cumulative_stat_dist[None, :]
            >= np.random.rand(n, 1)
        ).argmax(axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def transition_operator(self, f: np.ndarray) -> np.ndarray:
        """Apply the transition operator P to a batch of functions.

        (Pf)(s) = Σ_{s'} P(s, s') f(s')

        Parameters
        ----------
        f  : (..., num_states) ndarray — one function per batch element

        Returns
        -------
        Pf : (..., num_states) ndarray
        """
        return f @ self.transition_matrix.T

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def _compute_spectral_gap(self) -> float:
        """Spectral gap via the symmetrised matrix A = D^{1/2} P D^{-1/2}.

        Computed as  1 − σ₂,  where σ₂ is the second largest singular value
        of A (equivalently, the square root of the second largest eigenvalue
        of AᵀA).
        """
        sqrt_rho = np.sqrt(self.stat_dist)
        A = np.diag(sqrt_rho) @ self.transition_matrix @ np.diag(1.0 / sqrt_rho)
        # Squared singular values = eigenvalues of AᵀA, sorted ascending
        sq_sing_vals = np.sort(np.abs(np.linalg.eigvals(A.T @ A)))
        gap = 1.0 - sq_sing_vals[-2]
        print(f"Spectral gap: {gap:.2e}")
        return gap

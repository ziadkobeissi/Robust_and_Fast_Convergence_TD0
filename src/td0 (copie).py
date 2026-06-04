"""
TD(0) with linear function approximation.

Notation
--------
    Φ   : (d, k) feature matrix
    θ   : (k,)   parameter vector,  V_θ(s) = φ(s)ᵀ θ
    θ*  : TD(0) fixed point, solution of  H θ* = b
    H   : Σ − γ Σ₁    (TD matrix)
    S   : (H + Hᵀ)/2   (symmetric part; controls convergence rate)
    Σ   : Φᵀ D Φ       (feature second-moment matrix under ρ)
    Σ₁  : Φᵀ D P Φ     (one-step cross-covariance)
    b   : Φᵀ D r        (projected rewards)
    δ   : r + γ V(s') − V(s)  (TD error)
"""

import numpy as np
import scipy as sp

from markov_reward_process import MarkovRewardProcess
from parametrisation import LinearParametrisation


class TD0LinearFA:
    """
    TD(0) learner with linear function approximation  V_θ(s) = φ(s)ᵀ θ.

    Multiple γ values and multiple parallel copies are handled simultaneously
    via batched tensor arithmetic.  The convention is that nb_parallel_copies
    equals the number of γ values (one independent run per γ), though it can
    also be used to compare two settings under identical random samples
    (see compare_td0_with_different_LR.py).

    Parameters
    ----------
    do_pctd : bool
        If True, use the paired-comparison TD(0) update (PCTD0) instead of
        standard TD(0).
    """

    def __init__(self, mrp: MarkovRewardProcess, param: LinearParametrisation,
                 gamma_vect: np.ndarray = np.array([0.99]), do_pctd: bool = False):
        self.mrp        = mrp
        self.param      = param
        # Shape (n_gamma, 1, 1) so it broadcasts over (copies, avg_copies, states)
        self.gamma_vect = gamma_vect.reshape(-1, 1, 1)
        self.num_states = mrp.num_states
        self.do_pctd    = do_pctd

        self._compute_matrices()
        self._compute_solution()

    # ------------------------------------------------------------------
    # Pre-computation of key matrices
    # ------------------------------------------------------------------

    def _compute_matrices(self):
        """Build Σ, H, S and their mean-centred counterparts Σ̂, Ĥ, Ŝ."""
        Phi = self.param.feature_matrix     # (d, k)
        rho = self.mrp.stat_dist            # (d,)
        D   = np.diag(rho)                  # (d, d)
        P   = self.mrp.transition_matrix    # (d, d)

        # Feature second-moment matrix under ρ:  Σ = Φᵀ D Φ
        self.Sigma      = Phi.T @ D @ Phi
        print(f"Minimum eigenvalue of Σ: {np.linalg.eigvalsh(self.Sigma).min():.4e}")
        self.Sigma_half = sp.linalg.sqrtm(self.Sigma)

        # Mean feature vector under ρ:  μ = Φᵀ ρ
        self.mu = Phi.T @ rho

        # One-step cross-covariance:  Σ₁ = Φᵀ D P Φ
        Sigma_1 = Phi.T @ D @ P @ Phi

        # TD matrix H = Σ − γ Σ₁  and its symmetric part S = (H + Hᵀ)/2
        # Shape: (n_gamma, k, k) via broadcasting with gamma_vect (n_gamma, 1, 1)
        self.H = self.Sigma[None] - self.gamma_vect * Sigma_1[None]
        self.S = (self.H + self.H.transpose(0, 2, 1)) / 2.0

        # Mean-centred variants (relevant when Φ contains a constant column)
        mu_muT             = np.outer(self.mu, self.mu)
        self.Sigmahat      = self.Sigma  - mu_muT
        self.Sigmahat_half = sp.linalg.sqrtm(self.Sigmahat)
        Sigmahat_1         = Sigma_1 - mu_muT
        self.Hhat = self.Sigmahat[None] - self.gamma_vect * Sigmahat_1[None]
        self.Shat = (self.Hhat + self.Hhat.transpose(0, 2, 1)) / 2.0

    def _compute_solution(self):
        """Solve H θ* = b for each γ to obtain the TD(0) fixed point.

        b = Φᵀ D r  is the projection of the reward vector onto the feature space.
        """
        b = (self.mrp.expected_reward * self.mrp.stat_dist) @ self.param.feature_matrix  # (k,)

        n_gamma         = self.gamma_vect.shape[0]
        self.theta_star = np.zeros((n_gamma, 1, self.param.num_features))
        for i in range(n_gamma):
            self.theta_star[i, 0] = np.linalg.solve(self.H[i], b)

        # Pre-compute V_θ*(s) = Φ θ*  for all states; used in MSE evaluations
        self.values_at_solution = self.theta_star @ self.param.feature_matrix.T
        # shape: (n_gamma, 1, d)

        self.centered_feature_matrix = self.param.feature_matrix - self.mu.reshape(1, -1)
        constants_are_admissible = self.param.has_constant_function()
        if constants_are_admissible:
            self.thetahat_star = self.theta_star
        else:
            bhat = (self.mrp.expected_reward * self.mrp.stat_dist) @ self.centered_feature_matrix  # (k,)
            self.thetahat_star = np.zeros((n_gamma, 1, self.param.num_features))
            for i in range(n_gamma):
                self.thetahat_star[i, 0] = np.linalg.solve(self.Hhat[i], bhat) 

        self.values_at_solution_centered = self.thetahat_star @ self.centered_feature_matrix.T
    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_iid(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw n i.i.d. (s, s', r) transitions from the stationary distribution."""
        s      = self.mrp.sample_stationary(n)
        s_next = self.mrp.sample_transition(s)
        r      = self.mrp.sample_reward(s)
        return s, s_next, r

    # ------------------------------------------------------------------
    # TD error
    # ------------------------------------------------------------------

    def _td_error(self, s: np.ndarray, s_next: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Compute the TD(0) error  δ = r + γ V(s') − V(s).

        Returns
        -------
        δ : (nb_parallel_copies, nb_copies_to_average) ndarray
        """
        gamma_2d = self.gamma_vect.reshape(-1, 1)   # (n_gamma, 1)
        return r + gamma_2d * self.param.evaluate(s_next) - self.param.evaluate(s)

    # ------------------------------------------------------------------
    # Parameter update rules
    # ------------------------------------------------------------------

    def _update_td0(self, alpha: np.ndarray):
        """Standard TD(0) step:  θ ← θ + α δ φ(s)."""
        s, s_next, r = self._sample_iid(self.param.nb_copies_to_average)

        delta = self._td_error(s, s_next, r).reshape(
            self.param.nb_parallel_copies, self.param.nb_copies_to_average, 1)
        phi_s = self.param.feature(s).reshape(1, s.shape[0], -1)

        self.param.update_theta(
            alpha.reshape(self.param.nb_parallel_copies, 1, -1) * delta * phi_s)

    def _update_pctd0(self, alpha: np.ndarray):
        """Paired-comparison TD(0) step using two independent (s, s', r) samples.

        The anti-symmetric estimator  Δθ = α · (δ₁−δ₂)/2 · (φ(s₁)−φ(s₂))
        cancels the bias from correlated state visits in certain settings.
        """
        s1, s1_next, r1 = self._sample_iid(self.param.nb_copies_to_average)
        s2, s2_next, r2 = self._sample_iid(self.param.nb_copies_to_average)

        pctd = 0.5 * (self._td_error(s1, s1_next, r1) - self._td_error(s2, s2_next, r2))
        pctd = pctd.reshape(self.param.nb_parallel_copies, self.param.nb_copies_to_average, 1)

        phi_diff = (self.param.feature(s1) - self.param.feature(s2)).reshape(1, s1.shape[0], -1)
        self.param.update_theta(
            alpha.reshape(self.param.nb_parallel_copies, 1, -1) * pctd * phi_diff)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, num_iterations: int, alpha: np.ndarray,
              compute_mse: bool = False, compute_mse_averaged: bool = False,
              mse_checkpoints: list[int] = []) -> tuple[np.ndarray, np.ndarray]:
        """Run TD(0) for ``num_iterations`` steps, logging MSE at checkpoints.

        Parameters
        ----------
        alpha                : (nb_parallel_copies, k) learning-rate array
        compute_mse          : log MSE of current θ at each checkpoint
        compute_mse_averaged : log MSE of Polyak-average θ̄ at each checkpoint
        mse_checkpoints      : iteration indices at which MSE is evaluated

        Returns
        -------
        mse_list, mse_list_averaged :
            arrays of shape (len(mse_checkpoints), 3, nb_parallel_copies, nb_copies_to_average)
            — the 3 metrics are (mse_S, mse_values, mse_advantages) in that order.
        """
        update         = self._update_pctd0 if self.do_pctd else self._update_td0
        compute_errors = self.compute_pctd_errors if self.do_pctd else self.compute_td_errors
        checkpoint_set = set(mse_checkpoints)

        mse_list, mse_list_averaged = [], []
        for t in range(1, num_iterations + 1):
            update(alpha)
            if t in checkpoint_set:
                if compute_mse:
                    mse_list.append(compute_errors(with_average=False))
                if compute_mse_averaged and self.param.do_average:
                    mse_list_averaged.append(compute_errors(with_average=True))

        return np.array(mse_list), np.array(mse_list_averaged)

    # ------------------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------------------

    def compute_td_errors(self, with_average: bool = False) -> np.ndarray:
        """Compute the MSE for the current parameter estimate.

        Let Δ(s) = V_θ(s) − V_θ*(s) be the pointwise value-function error.

        MSE     = E_ρ[Δ(s)²]

        Returns
        -------
        mse
            of shape (nb_parallel_copies, nb_copies_to_average)
        """
        delta_V      = self.param.evaluate_all_states(with_average) - self.values_at_solution
        delta_V_centered = (self.param.evaluate_all_states_arbitrary_matrix(with_average, self.centered_feature_matrix) 
                            - self.param.generic_evaluate_all_states(self.theta_star, self.centered_feature_matrix))
        # Both: (nb_parallel_copies, nb_copies_to_average, d)

        rho   = self.mrp.stat_dist.reshape(1, 1, -1)  # broadcast over copies

        mse     = np.sum(rho * delta_V ** 2, axis=-1)
        centered_mse = np.sum(rho * (delta_V_centered) ** 2, axis=-1)

        return mse, centered_mse
    
    def compute_pctd_errors(self, with_average: bool = False) -> np.ndarray:
        """Compute the paired-comparison MSE for the current parameter estimate.

        Let Δ(s) = V_θ(s) − V_θ*(s) be the pointwise value-function error.

        PCTD-MSE = E_ρ[(Δ(s₁)−Δ(s₂))²]  where s₁, s₂ ~ ρ i.i.d.

        Returns
        -------
        pctd_mse
            of shape (nb_parallel_copies, nb_copies_to_average)
        """
        delta_V      = (self.param.evaluate_all_states_arbitrary_matrix(with_average, self.centered_feature_matrix)
                        - self.values_at_solution_centered)
        # shape: (nb_parallel_copies, nb_copies_to_average, d)

        rho   = self.mrp.stat_dist.reshape(1, 1, -1)  # broadcast over copies

        pctd_mse = np.sum(rho * rho * delta_V ** 2, axis=-1) * 2.0
        return pctd_mse

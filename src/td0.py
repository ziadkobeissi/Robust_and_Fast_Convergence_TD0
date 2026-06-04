"""
TD(0) with linear function approximation — three-class hierarchy.

    TD0LinearFABase       : common infrastructure (matrices, solution, sampling, training loop)
    StandardTD0LinearFA   : standard TD(0)
    PCTD0LinearFA         : paired-comparison TD(0) with mean-centred geometry

Design
------
The single method `_compute_matrices` lives only in TD0LinearFABase and uses
`self.Phi` as the effective feature matrix.  Each subclass sets `self.Phi`
via `_setup_Phi` (called once at the start of `__init__`):

    StandardTD0LinearFA : self.Phi = self.param.feature_matrix          (Φ)
    PCTD0LinearFA       : self.Phi = self.param.feature_matrix − 1·μᵀ  (Φ̂, ρ-centred)

With Φ̂ = Φ − 1·μᵀ the standard formulas produce the centred quantities
automatically — Σ̂, Σ̂₁, Ĥ, Ŝ, b̂ — with no code duplication.

Notation
--------
    Φ   : (num_states, num_features) feature matrix
    θ   : (num_features,)   parameter vector,  V_θ(s) = φ(s)ᵀ θ
    θ*  : TD fixed point, solution of  H θ* = b
    H   : Σ − γ Σ₁    (TD matrix built from self.Phi)
    S   : (H + Hᵀ)/2
    Σ   : self.Phi.T D self.Phi
    Σ₁  : self.Phi.T D P self.Phi
    μ   : Φᵀ ρ  (mean feature vector under ρ; stored in self.mu)
    b   : self.Phi.T D r
    δ   : r + γ V(s') − V(s)  (TD error, always uses original Φ)
"""

import numpy as np
import scipy as sp

from markov_reward_process import MarkovRewardProcess
from parametrisation import LinearParametrisation


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TD0LinearFABase:
    """
    Common infrastructure for TD(0) learners with linear function approximation.

    `_setup_Phi` is called first; it sets self.Phi and (optionally) self.mu.
    `_compute_matrices` and `_compute_solution` then build all derived quantities
    from self.Phi.  Subclasses must implement `_update` and `compute_errors`.
    """

    def __init__(self, mrp: MarkovRewardProcess, param: LinearParametrisation,
                 gamma_vect: np.ndarray = np.array([0.99]),
                 batch_size: int = 1):
        self.mrp        = mrp
        self.param      = param
        # Shape (num_coupled_trajectories, 1, 1) so it broadcasts over
        # (num_coupled_trajectories, batch_size, num_independent_copies)
        self.gamma_vect = gamma_vect.reshape(-1, 1, 1)
        self.num_states = mrp.num_states
        self.batch_size = batch_size

        self.mu = self.param.feature_matrix.T @ self.mrp.stat_dist
        self._setup_Phi()         # sets self.Phi (subclass-specific)
        self._compute_matrices()  # uses self.Phi
        self._compute_solution()  # uses self.Phi (for b) and self.param.feature_matrix (for values)
        self.projected_spectral_gap = self._compute_projected_spectral_gap(self.param.first_constant_feature)  # uses self.Phi and self.mu

    # ------------------------------------------------------------------
    # Feature matrix selection
    # ------------------------------------------------------------------

    def _setup_Phi(self):
        """Set self.Phi — the effective feature matrix for the TD equations."""
        self.Phi = self.param.feature_matrix

    # ------------------------------------------------------------------
    # Matrix pre-computation  (single implementation, uses self.Phi)
    # ------------------------------------------------------------------

    def _compute_matrices(self):
        """Build Σ, H, S from self.Phi.

        self.mu = self.Phi.T @ ρ is the mean of the effective features under ρ.
        For StandardTD0 this equals E_ρ[φ(s)]; for PCTD0 it equals 0.
        """
        Phi      = self.Phi                      # (num_states, num_features) — set by _setup_Phi
        rho      = self.mrp.stat_dist            # (num_states,)
        D        = np.diag(rho)                  # (num_states, num_states)
        trans_mat = self.mrp.transition_matrix   # (num_states, num_states)

        self.Sigma      = Phi.T @ D @ Phi
        print(f"Minimum eigenvalue of Σ: {np.linalg.eigvalsh(self.Sigma).min():.4e}")
        self.Sigma_half = sp.linalg.sqrtm(self.Sigma)

        # Mean of effective features under ρ (= 0 for PCTD0, restored afterwards)
        self.mu = Phi.T @ rho

        Sigma_1 = Phi.T @ D @ trans_mat @ Phi
        self.H  = self.Sigma[None] - self.gamma_vect * Sigma_1[None]
        # (num_coupled_trajectories, num_features, num_features)
        self.S  = (self.H + self.H.transpose(0, 2, 1)) / 2.0
    
    def _compute_projected_spectral_gap(self, first_constant_feature: bool) -> float:
        """Compute the projected spectral gap of P with respect to self.Phi.

        The projected spectral gap is defined as 1 − σ₂, where σ₂ is the
        second largest singular value of the matrix A = Σ^{-1/2} Σ₁ Σ^{-1/2}.
        """
        _continue = True
        if first_constant_feature:
            Phi_hat = self.Phi[:,1:] - self.mu[1:]  # (num_states, num_features-1)
        else:
            _continue = not self.param.has_constant_function
            Phi_hat = self.Phi - self.mu  # (num_states, num_features)
        
        D = np.diag(self.mrp.stat_dist)
        trans_mat = self.mrp.transition_matrix  

        if not _continue:
            print("Warning: projected spectral gap not computed : the constant function is admissible but not the first feature")
            return 0.0
        Sigma_hat = Phi_hat.T @ D @ Phi_hat
        Sigma_hat_inv_half = np.linalg.inv(sp.linalg.sqrtm(Sigma_hat))
        Sigma_1_hat = Phi_hat.T @ D @ trans_mat @ Phi_hat
        #aux = Sigma_hat_inv_half @ Sigma_1_hat @ Sigma_hat_inv_half
        #max_sing_val = np.linalg.svd(aux, compute_uv=False)[0]

        Sigma_aux_hat = Phi_hat.T @ trans_mat.T @ D @ trans_mat @ Phi_hat
        max_sing_val = np.sort(np.abs(np.linalg.eigvals(Sigma_hat_inv_half @ Sigma_aux_hat @ Sigma_hat_inv_half)))[-1]
        gap = 1.0 - max_sing_val
        print(f"Spectral gap projected on the set of centered features: {gap:.2e}")
        return gap

    def _compute_solution(self):
        """Solve H θ* = b for each γ.

        b = self.Phi.T D r  uses the effective feature matrix (centred for PCTD0).
        values_at_solution = θ* @ Φᵀ  uses the original feature matrix so that
        compute_errors can compare directly with param.evaluate_all_states().
        """
        b = (self.mrp.expected_reward * self.mrp.stat_dist) @ self.Phi  # (num_features,)

        num_coupled_trajectories = self.gamma_vect.shape[0]
        self.theta_star = np.zeros(
            (num_coupled_trajectories, 1, self.param.num_features))
        for i in range(num_coupled_trajectories):
            self.theta_star[i, 0] = np.linalg.lstsq(self.H[i], b, rcond=1e-8)[0]

        # V_θ*(s) = φ(s)ᵀ θ*  evaluated at every state
        # shape (num_coupled_trajectories, 1, num_states)
        self.values_at_solution = self.theta_star @ self.Phi.T

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
        """Compute δ = r + γ V(s') − V(s)  using the original feature matrix.

        Parameters
        ----------
        s, s_next : (num_independent_copies,) int array
        r         : (num_independent_copies,) float array

        Returns
        -------
        δ : (num_coupled_trajectories, num_independent_copies) ndarray
        """
        gamma_2d = self.gamma_vect.reshape(-1, 1)
        return r + gamma_2d * self.param.evaluate(s_next) - self.param.evaluate(s)

    def _td_error_minibatch(self, s: np.ndarray, s_next: np.ndarray,
                            r: np.ndarray) -> np.ndarray:
        """Compute δ = r + γ V(s') − V(s) for a minibatch of transitions.

        Parameters
        ----------
        s, s_next : (batch_size, num_independent_copies) int array
        r         : (batch_size, num_independent_copies) float array

        Returns
        -------
        δ : (num_coupled_trajectories, batch_size, num_independent_copies) ndarray
        """
        # gamma_vect already has shape (num_coupled_trajectories, 1, 1)
        return (r
                + self.gamma_vect * self.param.evaluate_minibatch(s_next)
                - self.param.evaluate_minibatch(s))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _update(self, alpha: np.ndarray):
        raise NotImplementedError

    def compute_errors(self, with_average: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, num_iterations: int, alpha: np.ndarray,
              compute_mse: bool = False, compute_mse_averaged: bool = False,
              mse_checkpoints: list[int] = []) -> tuple[np.ndarray, np.ndarray]:
        """Run TD(0) for ``num_iterations`` steps, logging MSE at checkpoints.

        Parameters
        ----------
        alpha                : (num_coupled_trajectories, num_features) learning-rate array
        compute_mse          : log MSE of current θ at each checkpoint
        compute_mse_averaged : log MSE of Polyak-average θ̄ at each checkpoint
        mse_checkpoints      : iteration indices at which MSE is evaluated

        Returns
        -------
        mse_list, mse_list_averaged :
            arrays of shape (len(mse_checkpoints), 3, num_coupled_trajectories, num_independent_copies)
            — the 3 metrics are (mse_values, mse_constant, mse_non_constant) for
            StandardTD0LinearFA, and (mse_values, mse_constant, mse_with_constant_part)
            for PCTD0LinearFA.
        """
        checkpoint_set = set(mse_checkpoints)

        mse_list, mse_list_averaged = [], []
        for t in range(1, num_iterations + 1):
            self._update(alpha)
            if t in checkpoint_set:
                if compute_mse:
                    mse_list.append(self.compute_errors(with_average=False))
                if compute_mse_averaged and self.param.do_average:
                    mse_list_averaged.append(self.compute_errors(with_average=True))

        return np.array(mse_list), np.array(mse_list_averaged)


# ---------------------------------------------------------------------------
# Standard TD(0)
# ---------------------------------------------------------------------------

class StandardTD0LinearFA(TD0LinearFABase):
    """
    Standard TD(0) with linear function approximation.

    self.Phi = self.param.feature_matrix  (unchanged)
    Fixed point: θ* solving H θ* = b  (non-centred)
    Metrics: use S = (H + Hᵀ)/2        (non-centred)
    """

    # _setup_Phi inherited from base: self.Phi = self.param.feature_matrix

    def _update(self, alpha: np.ndarray):
        """Mini-batch TD(0) step:  θ ← θ + α · (1/B) Σᵢ δᵢ φ(sᵢ).

        With batch_size=1 this reduces to the standard single-sample update.
        """
        B  = self.batch_size
        n  = self.param.num_independent_copies
        num_coupled_trajectories = self.param.num_coupled_trajectories

        s_flat, s_next_flat, r_flat = self._sample_iid(B * n)
        s      = s_flat.reshape(B, n)
        s_next = s_next_flat.reshape(B, n)
        r      = r_flat.reshape(B, n)

        delta = self._td_error_minibatch(s, s_next, r)
        # (num_coupled_trajectories, B, n)
        phi_s = self.param.feature_matrix[s]
        # (B, n, num_features)

        # Average TD gradient over the batch dimension
        # grad[c, j, f] = (1/B) Σ_b delta[c, b, j] * phi_s[b, j, f]
        grad = np.einsum('cbj,bjf->cjf', delta, phi_s) / B
        # (num_coupled_trajectories, num_independent_copies, num_features)

        self.param.update_theta(
            alpha.reshape(num_coupled_trajectories, 1, -1) * grad)

    def compute_errors(self, with_average: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute three convergence metrics for the current parameter estimate.

        Let Δ(s) = V_θ(s) − V_θ*(s) = (θ−θ*)ᵀ φ(s).

        mse_values       = E_ρ[Δ(s)²]
        mse_constant     = (μᵀ(θ−θ*))²   (squared mean-value error, μ = E_ρ[φ(s)])
        mse_non_constant = mse_values − mse_constant

        The decomposition holds because E_ρ[φ(s)−μ] = 0 makes the cross-term vanish:
            E_ρ[Δ²] = (μᵀ(θ−θ*))² + E_ρ[((θ−θ*)ᵀ(φ(s)−μ))²]

        Returns (mse_values, mse_constant, mse_non_constant),
        each of shape (num_coupled_trajectories, num_independent_copies).
        """
        theta      = self.param.theta_average if with_average else self.param.theta
        diff_theta = theta - self.theta_star
        # (num_coupled_trajectories, num_independent_copies, num_features)

        delta_V    = diff_theta @ self.Phi.T
        # (num_coupled_trajectories, num_independent_copies, num_states)
        rho        = self.mrp.stat_dist.reshape(1, 1, -1)
        mse_values = np.sum(rho * delta_V ** 2, axis=-1)
        # (num_coupled_trajectories, num_independent_copies)

        # (μᵀ(θ−θ*))²  where μ = Φᵀρ, shape (num_features,)
        mse_constant     = np.sum(self.mu * diff_theta, axis=-1) ** 2
        mse_non_constant = mse_values - mse_constant

        return mse_values, mse_constant, mse_non_constant


# ---------------------------------------------------------------------------
# Batch-2 TD(0)  (fair baseline for PCTD0 comparisons)
# ---------------------------------------------------------------------------

class Batch2TD0LinearFA(StandardTD0LinearFA):
    """
    Standard TD(0) with a mini-batch of size 2.

    Convenience subclass of StandardTD0LinearFA with batch_size=2.
    Equivalent to StandardTD0LinearFA(..., batch_size=2).
    """

    def __init__(self, mrp: MarkovRewardProcess, param: LinearParametrisation,
                 gamma_vect: np.ndarray = np.array([0.99])):
        super().__init__(mrp, param, gamma_vect, batch_size=2)


# ---------------------------------------------------------------------------
# Pairwise-centered TD(0)
# ---------------------------------------------------------------------------

class PCTD0LinearFA(TD0LinearFABase):
    """
    Pairwise-centered TD(0) with linear function approximation.

    self.Phi = self.param.feature_matrix − 1·μᵀ  (ρ-centred features)

    Passing the centred matrix through the standard formulas automatically
    produces all centred quantities — Σ̂, Ĥ, Ŝ, b̂ — with no code duplication.

    Fixed point: θ* solving Ĥ θ* = b̂
    Metrics: ρ-centred error Δ_c(s) = Δ(s) − E_ρ[Δ], metrics use Ŝ

    Update:  Δθ = α · (δ₁−δ₂)/2 · (φ(s₁)−φ(s₂))
    The anti-symmetric estimator cancels the mean-feature bias.

    Online estimates maintained during training (one per trajectory):
        learned_mu              : (num_coupled_trajectories, num_independent_copies, num_features)
                                  running average of φ(s)  → converges to μ
        learned_expected_reward : (num_coupled_trajectories, num_independent_copies)
                                  running average of r(s)  → converges to ρᵀr
    """

    def __init__(self, mrp: MarkovRewardProcess, param: LinearParametrisation,
                 gamma_vect: np.ndarray = np.array([0.99]),
                 batch_size: int = 1):
        # super().__init__ calls _setup_Phi (PCTD0's version via MRO) which sets
        # self.mu = μ, then _compute_matrices overwrites self.mu to 0 (centred mean).
        super().__init__(mrp, param, gamma_vect, batch_size=batch_size)

        # Restore self.mu to the true mean feature vector E_ρ[φ(s)]
        # (_compute_matrices set it to 0 because self.Phi is centred)
        self.mu = self.param.feature_matrix.T @ self.mrp.stat_dist

        # Online estimates: one independent running average per (coupled, independent) copy
        self.learned_mu = np.zeros((
            param.num_coupled_trajectories,
            param.num_independent_copies,
            param.num_features,
        ))
        self.learned_expected_reward = np.zeros((
            param.num_coupled_trajectories,
            param.num_independent_copies,
        ))
        self._n_samples = 0

    def _setup_Phi(self):
        """Set self.Phi to the ρ-centred feature matrix Φ̂ = Φ − 1·μᵀ."""
        self.Phi = self.param.feature_matrix - self.mu.reshape(1, -1)
        # (num_states, num_features)

    def _update(self, alpha: np.ndarray):
        """Pairwise-centered TD(0) step with online estimation of μ and E_ρ[r].

        Draws B pairs of num_independent_copies transitions (2B·n samples total)
        where B = self.batch_size.

        Parameter update (averaged over B pairs):
            Δθ = α · (1/B) Σ_b (δ₁ᵦ−δ₂ᵦ)/2 · (φ(s₁ᵦ)−φ(s₂ᵦ))

        Online estimates updated via incremental mean (n_new = 2B per step):
            learned_mu              ← running mean of φ(sᵢ)  (original features)
            learned_expected_reward ← running mean of rᵢ
        """
        B  = self.batch_size
        n  = self.param.num_independent_copies
        num_coupled_trajectories = self.param.num_coupled_trajectories

        s_flat, s_next_flat, r_flat = self._sample_iid(2 * B * n)

        # Split into two halves of B·n samples, then reshape to (B, n)
        s1      = s_flat[:B * n].reshape(B, n)
        s2      = s_flat[B * n:].reshape(B, n)
        s_next1 = s_next_flat[:B * n].reshape(B, n)
        s_next2 = s_next_flat[B * n:].reshape(B, n)
        r1      = r_flat[:B * n].reshape(B, n)
        r2      = r_flat[B * n:].reshape(B, n)

        phi_s1 = self.param.feature_matrix[s1]   # (B, n, num_features)
        phi_s2 = self.param.feature_matrix[s2]

        # --- Online estimation of μ and E_ρ[r] ---
        # Incremental mean: x̄ ← x̄ + (n_new / n_total) · (batch_mean − x̄)
        # n_new = 2B new samples, batch_mean = mean over both halves
        self._n_samples += 2 * B
        step = (2 * B) / self._n_samples

        # batch_phi: (n, num_features) — mean over 2B samples per copy
        # broadcast to (num_coupled_trajectories, n, num_features) via [np.newaxis]
        batch_phi = (phi_s1.sum(axis=0) + phi_s2.sum(axis=0)) / (2 * B)
        self.learned_mu += step * (batch_phi[np.newaxis] - self.learned_mu)

        # batch_r: (n,) — mean over 2B reward samples per copy
        batch_r = (r1.sum(axis=0) + r2.sum(axis=0)) / (2 * B)
        self.learned_expected_reward += step * (
            batch_r[np.newaxis] - self.learned_expected_reward)

        # --- Paired-comparison TD update ---
        td1  = self._td_error_minibatch(s1, s_next1, r1)
        # (num_coupled_trajectories, B, n)
        td2  = self._td_error_minibatch(s2, s_next2, r2)
        pctd = 0.5 * (td1 - td2)
        # (num_coupled_trajectories, B, n)

        phi_diff = phi_s1 - phi_s2
        # (B, n, num_features)

        # grad[c, j, f] = (1/B) Σ_b pctd[c, b, j] * phi_diff[b, j, f]
        grad = np.einsum('cbj,bjf->cjf', pctd, phi_diff) / B
        # (num_coupled_trajectories, n, num_features)

        self.param.update_theta(
            alpha.reshape(num_coupled_trajectories, 1, -1) * grad)

    def compute_errors(self, with_average: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute three ρ-centred convergence metrics.

        Let Δ̂(s) = (θ−θ*)ᵀ φ̂(s)  where φ̂(s) = φ(s) − μ  (centred features, self.Phi).
        Let r̄ = ρᵀr  (true expected reward).

        mse_values            = E_ρ[Δ̂(s)²]
        mse_constant          = ((learned_r̄ − r̄) / (1−γ))²
        mse_with_constant_part = E_ρ[((θ−θ*)ᵀ(φ(s)−learned_μ) + learned_r̄/(1−γ))²]

        Returns (mse_values, mse_constant, mse_with_constant_part),
        each of shape (num_coupled_trajectories, num_independent_copies).
        """
        theta      = self.param.theta_average if with_average else self.param.theta
        diff_theta = theta - self.theta_star
        # (num_coupled_trajectories, num_independent_copies, num_features)

        # mse_values: centred value MSE using Φ̂ = self.Phi
        delta_V_c  = diff_theta @ self.Phi.T
        # (num_coupled_trajectories, num_independent_copies, num_states)
        rho        = self.mrp.stat_dist.reshape(1, 1, -1)
        mse_values = np.sum(rho * delta_V_c ** 2, axis=-1)
        # (num_coupled_trajectories, num_independent_copies)

        # mse_constant: error in the constant (mean-reward) component
        real_r   = float(np.dot(self.mrp.expected_reward, self.mrp.stat_dist))
        gamma_2d = self.gamma_vect.reshape(-1, 1)
        # (num_coupled_trajectories, 1)
        mse_constant = ((self.learned_expected_reward - real_r) / (1 - gamma_2d)) ** 2
        # (num_coupled_trajectories, num_independent_copies)

        # mse_with_constant_part:
        # E_ρ[((θ−θ*)ᵀ(φ(s)−learned_μ) + learned_r̄/(1−γ))²]
        #   = E_ρ[(diff_θᵀφ(s) − diff_θᵀlearned_μ + learned_r̄/(1−γ))²]
        eval_diff = diff_theta @ self.param.feature_matrix.T
        # (num_coupled_trajectories, num_independent_copies, num_states)
        mu_proj   = np.sum(diff_theta * self.learned_mu, axis=-1, keepdims=True)
        # (num_coupled_trajectories, num_independent_copies, 1)
        r_term    = (self.learned_expected_reward / (1 - gamma_2d))[:, :, np.newaxis]
        # (num_coupled_trajectories, num_independent_copies, 1)
        term      = eval_diff - mu_proj + r_term
        # (num_coupled_trajectories, num_independent_copies, num_states)
        mse_with_constant_part = np.sum(rho * term ** 2, axis=-1)
        # (num_coupled_trajectories, num_independent_copies)

        return mse_values, mse_constant, mse_with_constant_part

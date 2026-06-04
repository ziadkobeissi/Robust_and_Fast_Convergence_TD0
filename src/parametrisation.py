"""
Linear function approximation parametrisations.

Notation
--------
    num_states    : number of states
    num_features  : number of features
    Φ   : (num_states, num_features) feature matrix,  Φ[s] = φ(s)ᵀ
    θ   : (num_features,)   parameter vector,  V_θ(s) = φ(s)ᵀ θ
    θ̄   : Polyak–Ruppert running average of θ
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LinearParametrisation:
    """
    Linear function approximation with batched parameter tensors.

    The parameter tensor θ has shape (num_coupled_trajectories, num_independent_copies, num_features):
      - num_coupled_trajectories  : independent runs (e.g. one per γ value)
      - num_independent_copies    : simultaneous trajectories whose updates are averaged
    When do_average=True, a Polyak–Ruppert running average θ̄ is maintained.
    """

    def __init__(self, num_states: int, num_features: int,
                 initial_theta: np.ndarray | None,
                 num_coupled_trajectories: int, num_independent_copies: int,
                 do_average: bool):
        self.num_states           = num_states
        self.num_features         = num_features
        self.num_coupled_trajectories   = num_coupled_trajectories
        self.num_independent_copies = num_independent_copies
        self.do_average           = do_average

        if initial_theta is None:
            seed = np.random.uniform(-1.0, 1.0, size=(1, 1, num_features))
        else:
            seed = initial_theta.reshape(1, 1, num_features)
        self.theta = np.tile(seed, (num_coupled_trajectories, num_independent_copies, 1))

        self.theta_average = np.zeros_like(self.theta)
        self.num_averages  = 0.0

        self.feature_matrix: np.ndarray | None = None  # (num_states, num_features); set by subclass or caller

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def feature(self, states: np.ndarray) -> np.ndarray:
        """Look up feature vectors φ(s) for a batch of states.

        Parameters
        ----------
        states : (num_independent_copies,) int array

        Returns
        -------
        Φ[states] : (num_independent_copies, num_features) ndarray
        """
        return self.feature_matrix[states]

    def evaluate(self, states: np.ndarray, with_average: bool = False) -> np.ndarray:
        """Compute V_θ(s) for a batch of states.

        Parameters
        ----------
        states       : (num_independent_copies,) int array
        with_average : use θ̄ instead of θ

        Returns
        -------
        V : (num_coupled_trajectories, num_independent_copies) ndarray
        """
        theta = self.theta_average if with_average else self.theta
        phi   = self.feature(states).reshape(1, states.shape[0], -1)  # (1, num_independent_copies, num_features)
        return np.sum(phi * theta, axis=-1)

    def evaluate_minibatch(self, states: np.ndarray, with_average: bool = False) -> np.ndarray:
        """Compute V_θ(s) for a minibatch of states.

        Parameters
        ----------
        states       : (batch_size, num_independent_copies) int array
        with_average : use θ̄ instead of θ

        Returns
        -------
        V : (num_coupled_trajectories, batch_size, num_independent_copies) ndarray
        """
        theta = self.theta_average if with_average else self.theta
        phi   = self.feature_matrix[states]       # (batch_size, num_independent_copies, num_features)
        return np.einsum('cjf,bjf->cbj', theta, phi)

    def evaluate_all_states(self, with_average: bool = False) -> np.ndarray:
        """Compute V_θ(s) for every state via a matrix product.

        Returns
        -------
        V : (num_coupled_trajectories, num_independent_copies, num_states) ndarray
        """
        return self.evaluate_all_states_arbitrary_matrix(with_average, self.feature_matrix)
    
    def evaluate_all_states_arbitrary_matrix(self, with_average, matrix) -> np.ndarray:
        """Compute V_θ(s) for every state via a matrix product.

        Returns
        -------
        V : (num_coupled_trajectories, num_independent_copies, num_states) ndarray
        """
        theta = self.theta_average if with_average else self.theta
        return self.generic_evaluate_all_states(theta, matrix)
    
    def generic_evaluate_all_states(self, theta, matrix) -> np.ndarray:
        """Compute V_θ(s) for every state via a matrix product.

        Returns
        -------
        V : (num_coupled_trajectories, num_independent_copies, num_states) ndarray
        """
        return theta @ matrix.T

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def has_constant_function(self, tol: float = 1e-10) -> tuple[bool, float]:
        """Test whether the constant function 1 lies in the column span of Φ.

        The constant function V(s) = 1 is representable as V_θ(s) = φ(s)ᵀ θ
        for some θ if and only if the all-ones vector 1 ∈ ℝᵈ is in the column
        space of Φ.  Equivalently, the linear system Φ θ = 1 must be consistent.

        This is tested by computing the least-squares solution θ̂ and measuring
        the residual ‖1 − Φ θ̂‖₂:  it is zero (up to numerical tolerance) if
        and only if 1 is in the column space of Φ.

        Parameters
        ----------
        tol : residual threshold below which the system is considered consistent.

        Returns
        -------
        in_span        : bool   — True if 1 ∈ col(Φ) within tolerance.
        residual_norm  : float  — ‖1 − Φ θ̂‖₂ for inspection.
        """
        ones        = np.ones(self.num_states)
        theta_ls, *_ = np.linalg.lstsq(self.feature_matrix, ones, rcond=None)
        residual_norm = float(np.linalg.norm(ones - self.feature_matrix @ theta_ls))
        return residual_norm < tol

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_theta(self, delta: np.ndarray):
        """Apply a parameter increment and refresh the Polyak–Ruppert average.

        Parameters
        ----------
        delta : (num_coupled_trajectories, num_independent_copies, num_features) ndarray
        """
        self.theta += delta
        if self.do_average:
            self.num_averages += 1.0
            # Online average:  θ̄ₙ = θ̄ₙ₋₁ + (θₙ − θ̄ₙ₋₁) / n
            self.theta_average += (self.theta - self.theta_average) / self.num_averages


# ---------------------------------------------------------------------------
# Diagonal Parametrisation  (controlled singular-value experiments)
# ---------------------------------------------------------------------------

class DiagonalParametrisation(LinearParametrisation):
    """
    LinearParametrisation whose feature matrix has a prescribed singular-value spectrum.

    Default (first_constant_feature=False):
        Φ = P · diag(eigenvalues) · Qᵀ  where P, Q are drawn from a random SVD,
        giving full control over the singular values while keeping the basis generic.

    With first_constant_feature=True:
        The first column of Φ is fixed to the all-ones vector φ₁(s) = 1.
        The remaining num_states−1 columns are built randomly with singular values eigenvalues[1:].
        eigenvalues[0] is ignored in this case.
    """

    def __init__(self, eigenvalues: np.ndarray, initial_theta: np.ndarray | None = None,
                 num_coupled_trajectories: int = 1, num_independent_copies: int = 1,
                 do_average: bool = False, first_constant_feature: bool = False):
        num_states = eigenvalues.shape[0]
        super().__init__(num_states=num_states, num_features=num_states,
                         initial_theta=initial_theta,
                         num_coupled_trajectories=num_coupled_trajectories,
                         num_independent_copies=num_independent_copies,
                         do_average=do_average)
        self.eigenvalues = eigenvalues

        if first_constant_feature:
            # First feature: φ₁(s) = 1 for all states (constant function)
            first_col = np.ones((num_states, 1))
            # Remaining num_states−1 features: random with singular values eigenvalues[1:]
            P_sub, _, Q_sub = np.linalg.svd(
                np.random.uniform(-1.0, 1.0, size=(num_states, num_states - 1)),
                full_matrices=False,
            )
            remaining_cols = P_sub @ np.diag(eigenvalues[1:]) @ Q_sub.T
            self.feature_matrix = np.hstack([first_col, remaining_cols])
        else:
            # Random orthogonal P_svd, Q_svd from SVD of a Gaussian matrix
            P_svd, _, Q_svd = np.linalg.svd(
                np.random.uniform(-1.0, 1.0, size=(num_states, num_states)))
            self.feature_matrix = P_svd @ np.diag(eigenvalues) @ Q_svd.T


# ---------------------------------------------------------------------------
# Random Linear Parametrisation
# ---------------------------------------------------------------------------

class RandomLinearParametrisation(LinearParametrisation):
    """
    LinearParametrisation whose feature matrix is drawn entry-wise from
    Uniform(−1, 1) and then row-normalised.

    Normalisation guarantee
    -----------------------
    Without first_constant_feature:
        ||φ(x)||₂ ≤ 1  for every state x  (the maximum-norm row equals exactly 1).

    With first_constant_feature=True:
        The first column of Φ is the all-ones vector φ₁(s) = 1.
        The remaining columns satisfy  ||φ_{(-1)}(x)||₂ ≤ 1  for every state x,
        where φ_{(-1)}(x) is the subvector of non-constant features.
    """

    def __init__(self, num_states: int, num_features: int,
                 initial_theta: np.ndarray | None = None,
                 num_coupled_trajectories: int = 1, num_independent_copies: int = 1,
                 do_average: bool = False, first_constant_feature: bool = False):
        super().__init__(num_states=num_states, num_features=num_features,
                         initial_theta=initial_theta,
                         num_coupled_trajectories=num_coupled_trajectories,
                         num_independent_copies=num_independent_copies,
                         do_average=do_average)
        
        self.first_constant_feature = first_constant_feature
        if first_constant_feature:
            # Random non-constant features: (num_states, num_features − 1)
            random_part = (np.random.uniform(-1.0, 1.0, size=(num_states, num_features - 1))
                           + np.random.uniform(-1.0, 1.0, size=(1, num_features - 1))) 
            # Normalise so that the non-constant subvector has norm ≤ 1 at every state
            random_part /= np.max(np.linalg.norm(random_part, axis=1))
            self.feature_matrix = np.hstack([np.ones((num_states, 1)), random_part])
        else:
            F = (np.random.uniform(-1.0, 1.0, size=(num_states, num_features))
                 + np.random.uniform(-1.0, 1.0, size=(1, num_features)))  # add a random offset to each column
            # Normalise so that the full feature vector has norm ≤ 1 at every state
            self.feature_matrix = F / np.max(np.linalg.norm(F, axis=1))

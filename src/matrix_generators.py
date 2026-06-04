"""
Random stochastic matrix generators.

Both functions rely on Dirichlet distributions to produce matrices that are
random yet controllably dense or sparse via the ``dirichlet_param`` argument:
a small value (≪ 1) gives sparser, more peaked rows; a value near 1 gives
approximately uniform rows.
"""

import numpy as np


def generate_stochastic(num_states: int, dirichlet_param: float = 0.1) -> np.ndarray:
    """Generate a random num_states×num_states stochastic matrix with i.i.d. Dirichlet rows."""
    return np.random.dirichlet(dirichlet_param * np.ones(num_states), size=num_states)


def generate_bistochastic(num_states: int, m: int | None = None,
                           dirichlet_param: float = 0.1) -> np.ndarray:
    """Generate a random num_states×num_states doubly-stochastic (bistochastic) matrix.

    Built as a convex combination of m random permutation matrices with
    Dirichlet-distributed weights.  By Birkhoff's theorem this is guaranteed
    to be bistochastic.  Symmetric results are rejected and retried so that the
    returned matrix is genuinely non-symmetric (and thus has a non-trivial
    spectral structure).

    Parameters
    ----------
    num_states     : matrix dimension (must be ≥ 3)
    m              : number of permutations to mix  (defaults to num_states²)
    dirichlet_param: Dirichlet concentration parameter for the mixing weights
    """
    if num_states == 2:
        raise ValueError("2×2 bistochastic matrices are always symmetric; use num_states ≥ 3.")
    if m is None:
        m = num_states * num_states

    row_idx = np.arange(num_states)
    while True:
        weights = np.random.dirichlet(dirichlet_param * np.ones(m))
        U = np.zeros((num_states, num_states))
        for w in weights:
            perm     = np.random.permutation(num_states)
            perm_mat = np.zeros((num_states, num_states))
            perm_mat[row_idx, perm] = 1.0   # permutation matrix: perm_mat[i, perm[i]] = 1
            U += w * perm_mat
        if not np.allclose(U, U.T, atol=1e-12, rtol=0):
            return U

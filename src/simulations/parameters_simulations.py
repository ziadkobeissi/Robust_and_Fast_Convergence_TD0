"""
Shared simulation parameters.

Edit this file to configure TD(0) and PCTD0 experiments.
Both td0_simulations.py and pctd0_simulations.py read from here.

Mini-batch convention
---------------------
BATCH_SIZE is the total number of transitions consumed per update step.
It must be even.

    TD0   uses batch_size = BATCH_SIZE          (BATCH_SIZE independent transitions)
    PCTD0 uses batch_size = BATCH_SIZE // 2     (BATCH_SIZE//2 pairs = BATCH_SIZE transitions)

Both algorithms therefore process the same number of samples per step,
making their runs directly comparable.
"""

# ---------------------------------------------------------------------------
# Problem dimensions
# ---------------------------------------------------------------------------

SEED            = 2
N               = 1000     # num_states
D               = 100      # num_features
GAMMA           = 0.99      # discount factor
DIRICHLET_PARAM = 2e-3     # Dirichlet concentration for the random transition matrix
STD_DEV_NOISE   = 1.0      # std dev of reward noise  (0.0 = deterministic)
BATCH_SIZE      = 1       # total transitions per update step (even integer for PCTD0)

# Whether the first feature is fixed to the constant function φ₁(s) = 1
FIRST_CONSTANT_FEATURE = True

# ---------------------------------------------------------------------------
# MSE decomposition control
# ---------------------------------------------------------------------------

# WITH_BIAS_TERM = True  : non-zero expected reward r(s) = 1 and random θ₀
# WITH_BIAS_TERM = False : r(s) = 0 and θ₀ = 0  (bias term is suppressed)
#
# WITH_VARIANCE_TERM = True  : reward noise σ = STD_DEV_NOISE
# WITH_VARIANCE_TERM = False : σ = 0  (variance term is suppressed;
#                               STD_DEV_NOISE is ignored and a warning is printed)
#
# At least one must be True.
WITH_BIAS_TERM     = True
WITH_VARIANCE_TERM = True

# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------

# If True, α = spectral_gap(P) / ALPHA_SCALE  (computed at runtime from the MRP).
# If False, α = (1 − γ) / ALPHA_SCALE.
USE_SPECTRAL_GAP_ALPHA = False
#ALPHA_SCALE            = 4.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

NB_ITER  = int(1e7)    # number of TD steps
NB_TRIALS = 100       # num_independent_copies — parallel trajectories averaged for MSE

# ---------------------------------------------------------------------------
# MSE checkpoint schedule  (logarithmically spaced)
# ---------------------------------------------------------------------------

import numpy as np

START_PLOT         = 10
NB_MSE_CHECKPOINTS = 200
_ratio             = (NB_ITER / START_PLOT) ** (1.0 / NB_MSE_CHECKPOINTS)
MSE_CHECKPOINTS    = np.unique(
    np.floor(START_PLOT * _ratio ** np.arange(NB_MSE_CHECKPOINTS + 1)).astype(int)
)

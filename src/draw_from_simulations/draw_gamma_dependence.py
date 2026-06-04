"""
Visualisation: C(γ) vs (1−γ) for the gamma-dependence experiment.

For each γ and each error variant, computes the tightest constant C(γ) such
that  MSE_γ(t) ≤ C(γ) / t  for all logged iterations t:

    C(γ) = max_t  t · MSE_γ(t)

Then plots C(γ) against (1−γ) on a log-log scale for the three variants
(total, bias, variance).  Theory predicts C(γ) ~ (1−γ)^{−2}, so the
expected slope in the log-log plot is −2.

A reference line with slope −2 is drawn for visual comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"   # Computer Modern (LaTeX style)


# ---------------------------------------------------------------------------
# Configuration  (must match gamma_dependence.py)
# ---------------------------------------------------------------------------

D          = 20
BATCH_SIZE = 1      # must match gamma_dependence.py
MSE_KEY    = "MSE_values"   # one of "MSE_values", "MSE_constant", "MSE_non_constant"

_ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = _ROOT / "experiments" / "data"
FIGURE_DIR = _ROOT / "experiments" / "figures"

# Variants to overlay on the same plot
VARIANTS = [
    (True,  1.0, "Total error"),
    (True,  0.0, "Bias only"),
    (False, 1.0, "Variance only"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path(d: int, with_bias: bool, std_dev_noise: float,
              batch_size: int = 1) -> Path:
    tag = ("_WB" if with_bias else "_WoutB") + ("_WV" if std_dev_noise > 0 else "_WoutV")
    return DATA_DIR / f"gamma_dependence_TD0_d{d}{tag}_B{batch_size}.npz"


def style_ax(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1 - \gamma$", fontsize=14)
    ax.set_ylabel(
        r"$C(\gamma) = \max_t \; t \cdot \mathrm{MSE}_\gamma(t)$", fontsize=13
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.legend(loc="lower right", fontsize=12)
    ax.invert_xaxis()   # left axis = small (1−γ), i.e. γ close to 1
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    one_minus_gamma_ref = None
    C_ref               = None

    for with_bias, std_dev_noise, label in VARIANTS:
        path = data_path(D, with_bias, std_dev_noise, BATCH_SIZE)
        data = np.load(path)
        print(f"Loaded {path.name}  ({data['computation_time_seconds']:.1f} s)")

        mse         = data[MSE_KEY]            # (n_checkpoints, n_gammas)
        checkpoints = data["mse_checkpoints"]  # (n_checkpoints,)
        gamma_vect  = data["gamma_vect"]       # (n_gammas,)

        # C(γ) = max_t ( t · MSE_γ(t) ),  shape (n_gammas,)
        # mse.T has shape (n_gammas, n_checkpoints)
        C = np.max(checkpoints * mse.T, axis=1)

        one_minus_gamma = 1.0 - gamma_vect
        ax.plot(one_minus_gamma, C, label=label)

        # Keep one curve to anchor the reference line
        if with_bias and std_dev_noise > 0:
            one_minus_gamma_ref = one_minus_gamma
            C_ref               = C

    # Reference line C ~ (1−γ)^{−2}, anchored at the midpoint of the total-error curve
    mid         = len(C_ref) // 2
    anchor      = C_ref[mid] * one_minus_gamma_ref[mid] ** 2
    ref_line    = anchor / one_minus_gamma_ref ** 2
    ax.plot(one_minus_gamma_ref, ref_line,
            "--", color="grey", linewidth=1.5, label=r"$(1-\gamma)^{-2}$")

    style_ax(ax)
    ax.set_title(rf"$d = {D}$", fontsize=13)

    figure_path = FIGURE_DIR / f"gamma_dependence_TD0_d{D}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {figure_path}")


if __name__ == "__main__":
    main()

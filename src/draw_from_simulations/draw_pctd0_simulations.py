"""
Visualisation for the PCTD0 multi-gamma simulation.

Loads the .npz archive produced by pctd0_simulations.py and draws one MSE
curve per discount factor on a single log-log plot:

    γ₁ = 0.9
    γ₂ = 0.99
    γ₃ = 1.0
    γ₄ = (2 − 2g) / (2 − g)   (spectral-gap-dependent)

Set MSE_KEY to choose which metric to display:
    "MSE_values"             — centred value MSE  (well-defined for all γ)
    "MSE_constant"           — (learned_r̄ − r̄)² / (1−γ)²  (NaN/Inf for γ=1)
    "MSE_with_constant_part" — full error including constant component  (NaN/Inf for γ=1)

Plot parameters (N, D, batch_size) are read from parameters_simulations.py
so that this script stays in sync with pctd0_simulations.py automatically.

The figure is saved as a .png in experiments/figures/pctd0/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulations"))

import parameters_simulations as P


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT      = Path(__file__).resolve().parent.parent.parent
_PCTD0_DIR = _ROOT / "experiments" / "pctd0"
FIGURE_DIR = _ROOT / "experiments" / "figures" / "pctd0"

_SUBDIR = {
    (True,  True):  "total_MSE",
    (True,  False): "MSE_bias_term_only",
    (False, True):  "MSE_variance_term_only",
}

# Which MSE metric to display.
# "MSE_values", "MSE_constant", or "MSE_with_constant_part".
MSE_KEY = "MSE_values"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path() -> Path:
    subdir = _SUBDIR[(P.WITH_BIAS_TERM, P.WITH_VARIANCE_TERM)]
    return _PCTD0_DIR / subdir / f"pctd0_multi_gamma_N{P.N}_D{P.D}_B{P.BATCH_SIZE}.npz"


def style_ax(ax: plt.Axes, xlim: tuple, ylim: tuple, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("MSE", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #ax.set_title(title, fontsize=16)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    path = data_path()
    data = np.load(path)

    print(f"Loaded: {path}")
    print(f"Computation time: {data['computation_time_seconds']:.1f} s")

    checkpoints  = data["mse_checkpoints"]
    gammas       = data["gammas"]        # (4,)
    alphas       = data["alphas"]        # (4,)
    spectral_gap = float(data["spectral_gap"])
    mse          = data[MSE_KEY]         # (len(checkpoints), 4)

    metric_title = {
        "MSE_values":             "Values MSE",
        "MSE_constant":           "Constant MSE",
        "MSE_with_constant_part": "MSE with constant part",
    }.get(MSE_KEY, MSE_KEY)

    title = (
        rf"PCTD0 — {metric_title}  —  $N={P.N}$,  $D={P.D}$,  batch$={P.BATCH_SIZE}$"
        "\n"
        rf"spectral gap $g={spectral_gap:.4f}$"
    )

    gamma_labels = [
        rf"$\gamma=0.9$,  $\alpha={alphas[0]:.2f}$",
        rf"$\gamma=0.99$,  $\alpha={alphas[1]:.2f}$",
        rf"$\gamma=1$,  $\alpha={alphas[2]:.2f}$",
        (rf"$\gamma=1.15$,"
         rf"  $\alpha={alphas[3]:.2f}$"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for i, label in enumerate(gamma_labels):
        ax.plot(checkpoints, mse[:, i], label=label)

    style_ax(ax, (10, 1e6), (1e-6, 1), title)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / path.with_suffix(".png").name
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {figure_path}")


if __name__ == "__main__":
    main()

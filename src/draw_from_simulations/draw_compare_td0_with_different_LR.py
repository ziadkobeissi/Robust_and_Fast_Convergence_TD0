"""
Visualisation for the LR-schedule comparison experiment.

Loads the .npz archive produced by compare_td0_with_different_LR.py and
draws the MSE curves for both settings (uniform LR vs. split LR) on a
single log-log plot.

Plot parameters (N, D, γ, batch_size) are read from parameters_simulations.py
so that this script stays in sync with compare_td0_with_different_LR.py automatically.

The figure is saved as a .png in experiments/figures/compare_td0_with_different_LR/.
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
DATA_DIR   = _ROOT / "experiments" / "data"
FIGURE_DIR = _ROOT / "experiments" / "figures" / "compare_td0_with_different_LR"

# Which MSE metric to display.  One of "MSE_values", "MSE_constant", "MSE_non_constant".
MSE_KEY = "MSE_values"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path() -> Path:
    return DATA_DIR / f"compare_LR_N{P.N}_D{P.D}_gamma_{P.GAMMA}_B{P.BATCH_SIZE}.npz"

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

    alpha_u = float(data["alpha_uniform"])
    alpha_f = float(data["alpha_first_coord"])
    alpha_r = float(data["alpha_other_coords"])

    checkpoints = data["mse_checkpoints"]
    mse_uniform = data[f"{MSE_KEY}_uniform"]
    mse_split   = data[f"{MSE_KEY}_split"]

    metric_title = {
        "MSE_values":       "Values MSE",
        "MSE_constant":     "Constant MSE",
        "MSE_non_constant": "Non-constant MSE",
    }.get(MSE_KEY, MSE_KEY)

    title = (
        rf"{metric_title}  —  $\gamma={P.GAMMA}$,  $N={P.N}$,  $D={P.D}$,  batch$={P.BATCH_SIZE}$"
        "\n"
        rf"uniform: $\alpha={alpha_u:.4f}$   |   "
        rf"split: $\alpha_1={alpha_f:.4f},\ \alpha_{{rest}}={alpha_r:.4f}$"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")

    ax.plot(checkpoints, mse_uniform, label=rf"Standard TD(0)", color="tab:blue")
    ax.plot(checkpoints, mse_split,
            label=rf"TD(0) with two LRs", color="tab:orange")

    style_ax(
        ax,
        xlim=(checkpoints[0], checkpoints[-1]),
        ylim=(1e-3, 2*1e4),
        title=title,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / path.with_suffix(".png").name
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {figure_path}")


if __name__ == "__main__":
    main()

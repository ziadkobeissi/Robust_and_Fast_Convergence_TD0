"""
Visualisation for the TD0 vs PCTD0 comparison experiment.

Loads the .npz archive produced by compare_pctd0_and_td0.py and draws MSE
curves on a single log-log plot:

    · MSE_values_pctd0        — centred value MSE for PCTD0
    · MSE_values_td0          — value MSE for TD0 (uniform LR)
    · MSE_non_constant_td0    — non-constant component of TD0's value MSE
    · MSE_values_td0_split    — value MSE for TD0 with split LR  (if present)

Plot parameters (N, D, γ, batch_size) are read from parameters_simulations.py
so that this script stays in sync with compare_pctd0_and_td0.py automatically.

The figure is saved as a .png in experiments/figures/compare_pctd0_and_td0/.
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
FIGURE_DIR = _ROOT / "experiments" / "figures" / "compare_pctd0_and_td0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path() -> Path:
    return DATA_DIR / f"compare_pctd0_td0_N{P.N}_D{P.D}_gamma_{P.GAMMA}_B{P.BATCH_SIZE}.npz"


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

    alpha       = float(data["alpha"])
    checkpoints = data["mse_checkpoints"]

    mse_values_pctd0 = data["MSE_values_pctd0"]
    run_td0_first    = bool(data["run_td0_first"])
    run_td0_split    = bool(data["run_td0_split"]) if "run_td0_split" in data else False

    title = (
        rf"TD0 vs PCTD0  —  $\gamma={P.GAMMA}$,  $N={P.N}$,  $D={P.D}$,  "
        rf"$\alpha={alpha:.4f}$,  batch$={P.BATCH_SIZE}$"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")


    if run_td0_first:
        mse_values_td0       = data["MSE_values_td0"]
        mse_non_constant_td0 = data["MSE_non_constant_td0"]
        ax.plot(checkpoints, mse_values_td0,
                label=r"TD0 — $\mathrm{MSE}_{\mathrm{values}}$", color="tab:blue", linestyle="dashed")
        ax.plot(checkpoints, mse_non_constant_td0,
                label=r"TD0 — $\widehat{\mathrm{MSE}}_{\mathrm{values}}$", color="tab:blue")

        if False:#run_td0_split:
            alpha_f              = float(data["alpha_first_coord"])
            mse_values_td0_split = data["MSE_values_td0_split"]
            mse_non_constant_td0_split = data["MSE_non_constant_td0_split"]
            ax.plot(checkpoints, mse_values_td0_split,
                    label=(rf"TD0 with 2 LR — "
                           r"$\mathrm{MSE}_{\mathrm{values}}$"), color="tab:orange",linestyle="dashed")
            ax.plot(checkpoints, mse_non_constant_td0_split,
                    label=(rf"TD0 with 2 LR — "
                           r"$\widehat{\mathrm{MSE}}_{\mathrm{values}}$"), color="tab:orange")

    ax.plot(checkpoints, mse_values_pctd0,
            label=r"PCTD0 — $\widehat{\mathrm{MSE}}_{\mathrm{values}}$", color="tab:green")

    style_ax(ax, (10, 1e6), (1e-6, 1e4), title)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / path.with_suffix(".png").name
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {figure_path}")


if __name__ == "__main__":
    main()

"""
Visualisation for the controlled-EV experiment (d=3).

Loads a .npz archive produced by Controlled_EV_d3_clean.py and draws the
MSE curves for each eigenvalue-spread scenario on a log-log plot, together
with a reference O(1/t) upper-bound line.

The figure is saved as a .png in experiments/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Plot parameters  (must match the values used in Controlled_EV_d3_clean.py)
# ---------------------------------------------------------------------------

D              = 3
GAMMA          = 0.9
ALPHA          = (1.0 - GAMMA) / 4.0
BATCH_SIZE     = 1      # must match Controlled_EV_d3.py
WITH_BIAS      = True
STD_DEV_NOISE  = 0.0

# Which MSE metric to display.  One of "MSE_values", "MSE_constant", "MSE_non_constant".
MSE_KEY = "MSE_values"

_ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = _ROOT / "experiments" / "data"
FIGURE_DIR = _ROOT / "experiments" / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_filename(d: int, gamma: float, alpha: float,
                  with_bias: bool, std_dev_noise: float,
                  batch_size: int = 1) -> Path:
    """Reconstruct the .npz path from experiment parameters."""
    stem = (
        f"CEV_TD0_d{d}_gamma_{gamma}_alpha_{alpha}"
        f"{'_WB' if with_bias else '_WoutB'}"
        f"{'_WV' if std_dev_noise > 0 else '_WoutV'}"
        f"_B{batch_size}"
    )
    return DATA_DIR / (stem + ".npz")


def sci_label(x: float, precision: int = 1) -> str:
    """Format a float as a LaTeX scientific-notation string, e.g. r'1.0 \times 10^{-3}'."""
    exp   = int(np.floor(np.log10(abs(x))))
    coeff = x / 10 ** exp
    return rf"{coeff:.{precision}f} \times 10^{{{exp}}}"


def style_ax(ax: plt.Axes, xlim: tuple, ylim: tuple, title: str) -> None:
    """Apply consistent log-log styling to an axes object."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.legend(loc="lower left", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=16)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = data_filename(D, GAMMA, ALPHA, WITH_BIAS, STD_DEV_NOISE, BATCH_SIZE)
    data      = np.load(data_path)

    print(f"Loaded: {data_path}")
    print(f"Computation time: {data['computation_time_seconds']:.1f} s")

    # Shape: (n_k, n_checkpoints)
    mse          = data[MSE_KEY]
    checkpoints  = data["mse_checkpoints"]
    omegas       = data["omegas"]

    # O(1/t) reference line: scale C so it passes through the maximum of all curves
    C            = np.max(checkpoints * mse.max(axis=0))
    upper_bound  = C / checkpoints

    # Determine title from the metric name
    metric_title = {"MSE_values": "Values MSE", "MSE_constant": "Constant MSE",
                    "MSE_non_constant": "Non-constant MSE"}.get(MSE_KEY, MSE_KEY)
    title = f"{metric_title} — bias={'yes' if WITH_BIAS else 'no'}, σ={STD_DEV_NOISE}"

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    for k_idx, omega in enumerate(omegas):
        ax.plot(checkpoints, mse[k_idx], label=rf"$\omega = {sci_label(omega)}$")

    ax.plot(checkpoints, upper_bound, "--", linewidth=2.5,
            color="pink", label=r"$C / t$")

    style_ax(ax, xlim=(checkpoints[0], checkpoints[-1]), ylim=(1e-6, 1e0), title=title)

    figure_path = FIGURE_DIR / data_path.with_suffix(".png").name
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {figure_path}")


if __name__ == "__main__":
    main()

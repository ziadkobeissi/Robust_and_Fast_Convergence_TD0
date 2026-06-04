"""
Sanity check for the gamma-dependence experiment.

For a chosen variant, plots the MSE(t) curve for each γ value together with
its individual upper-bound line C(γ)/t, where

    C(γ) = max_t  t · MSE_γ(t)

is the tightest O(1/t) constant estimated from the data.

Interpretation:
  - Each solid curve is MSE_γ(t).
  - Each dashed curve of the same colour is the C(γ)/t bound.
  - If the experiment ran enough iterations, the solid curve should be
    tangent to (and below) its dashed counterpart at the last logged step.
  - If the solid curve is still well below its dashed line at the end,
    the constant C(γ) is over-estimated and more iterations are needed.

Curves are coloured by γ via a continuous colormap; a colorbar replaces
the legend to avoid clutter when many γ values are compared.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration  (must match a variant run in gamma_dependence.py)
# ---------------------------------------------------------------------------

D             = 20
BATCH_SIZE    = 1      # must match gamma_dependence.py
WITH_BIAS     = True
STD_DEV_NOISE = 1.0

MSE_KEY = "MSE_values"   # one of "MSE_values", "MSE_constant", "MSE_non_constant"

_ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = _ROOT / "experiments" / "data"
FIGURE_DIR = _ROOT / "experiments" / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path(d: int, with_bias: bool, std_dev_noise: float,
              batch_size: int = 1) -> Path:
    tag = ("_WB" if with_bias else "_WoutB") + ("_WV" if std_dev_noise > 0 else "_WoutV")
    return DATA_DIR / f"gamma_dependence_TD0_d{d}{tag}_B{batch_size}.npz"


def sci_label(x: float, precision: int = 1) -> str:
    """Format a float as a LaTeX scientific-notation string."""
    exp   = int(np.floor(np.log10(abs(x))))
    coeff = x / 10 ** exp
    return rf"{coeff:.{precision}f} \times 10^{{{exp}}}"


def style_ax(ax: plt.Axes, xlim: tuple, ylim: tuple) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    path = data_path(D, WITH_BIAS, STD_DEV_NOISE, BATCH_SIZE)
    data = np.load(path)

    print(f"Loaded: {path}")
    print(f"Computation time: {data['computation_time_seconds']:.1f} s")

    mse         = data[MSE_KEY]            # (n_checkpoints, n_gammas)
    checkpoints = data["mse_checkpoints"]  # (n_checkpoints,)
    gamma_vect  = data["gamma_vect"]       # (n_gammas,)

    # Per-γ upper-bound constant: C(γ) = max_t ( t · MSE_γ(t) )
    # mse.T shape: (n_gammas, n_checkpoints)
    C = np.max(checkpoints * mse.T, axis=1)   # (n_gammas,)

    # Map γ values to colours via a perceptually-uniform colormap
    norm   = plt.Normalize(vmin=gamma_vect.min(), vmax=gamma_vect.max())
    cmap   = plt.cm.viridis
    colors = cmap(norm(gamma_vect))

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")

    for k, (gamma, c_k, color) in enumerate(zip(gamma_vect, C, colors)):
        ax.plot(checkpoints, mse[:, k], color=color, linewidth=1.2)
        ax.plot(checkpoints, c_k / checkpoints,
                "--", color=color, linewidth=0.8, alpha=0.7)

    # Colorbar encodes γ, replacing a cluttered legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=r"$\gamma$")
    cbar.ax.tick_params(labelsize=10)

    style_ax(
        ax,
        xlim=(checkpoints[0], checkpoints[-1]),
        ylim=(1e-6, max(mse.max(), C.max()) * 2),
    )

    title = (
        rf"Sanity check — $d={D}$, "
        rf"bias={'yes' if WITH_BIAS else 'no'}, $\sigma={STD_DEV_NOISE}$"
        "\nsolid = MSE$(t)$,  dashed = $C(\gamma)/t$"
    )
    ax.set_title(title, fontsize=11)

    figure_path = FIGURE_DIR / ("SC_" + path.with_suffix(".png").name)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {figure_path}")


if __name__ == "__main__":
    main()

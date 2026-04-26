from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_parameter_stability(rolling_parameters: pd.DataFrame) -> pd.DataFrame:
    """Return compact stability stats for best L/S over rolling windows."""
    if rolling_parameters.empty:
        return pd.DataFrame(
            {
                "metric": ["n_windows", "L_mean", "L_std", "S_mean", "S_std", "L_nunique", "S_nunique"],
                "value": [0.0, float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0.0],
            }
        )

    l = rolling_parameters["best_L"].astype(float)
    s = rolling_parameters["best_S"].astype(float)
    return pd.DataFrame(
        {
            "metric": ["n_windows", "L_mean", "L_std", "S_mean", "S_std", "L_nunique", "S_nunique"],
            "value": [
                float(len(rolling_parameters)),
                float(l.mean()),
                float(l.std(ddof=1)) if len(l) > 1 else 0.0,
                float(s.mean()),
                float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                float(l.nunique()),
                float(s.nunique()),
            ],
        }
    )


def plot_parameter_evolution(
    rolling_parameters: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Plot time evolution of best L and best S in the walk-forward process."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_l, ax_s = axes

    if rolling_parameters.empty:
        ax_l.set_title("Parameter Evolution (no rolling windows)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    x = rolling_parameters["oos_start"]
    ax_l.plot(x, rolling_parameters["best_L"], marker="o", linewidth=1.0, color="tab:blue")
    ax_l.set_ylabel("Best L")
    ax_l.set_title("Rolling Parameter Evolution")
    ax_l.grid(True, alpha=0.25)

    ax_s.plot(x, rolling_parameters["best_S"], marker="o", linewidth=1.0, color="tab:orange")
    ax_s.set_ylabel("Best S")
    ax_s.set_xlabel("OOS Window Start")
    ax_s.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


#!/usr/bin/env python3
"""Post-process matched reduced-order and dq0-EMT frequency results.

Run after:
  1) python ieee39_reduced_order.py
  2) julia ieee39_main_run.jl

The comparison intentionally shows two EMT observables:
  - local GFL PLL frequency, and
  - the analytical frequency-divider H_G applied to EMT SG speeds.
This separates plant-model differences from the frequency-observable difference.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PY_FREQ = SCRIPT_DIR / "results_dynamic_local_filter" / "csv" / "gfl_frequency_filter_off.csv"
EMT_FREQ = SCRIPT_DIR / "plots" / "python_style_cbf" / "comparison" / "emt_pll_vs_analytical_divider.csv"
OUT_DIR = SCRIPT_DIR / "comparison_reduced_vs_emt"
GFL_BUSES = [30, 33, 35, 31, 38]
COLORS = {
    30: "#0072B2",
    33: "#3A923A",
    35: "#CC6677",
    31: "#009E73",
    38: "#7E57C2",
}
FREQ_LIMIT = -0.05


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No data in {path}")
    return {
        name: np.asarray([float(r[name]) for r in rows], dtype=float)
        for name in reader.fieldnames or []
    }


def main() -> None:
    py = read_csv(PY_FREQ)
    emt = read_csv(EMT_FREQ)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_py = py["t_s"]
    t_emt = emt["t_s"]

    summary_rows: list[list[object]] = []
    fig, axes = plt.subplots(len(GFL_BUSES), 1, figsize=(7.1, 9.5), sharex=True)

    for ax, bus in zip(np.atleast_1d(axes), GFL_BUSES):
        f_py = py[f"gfl_bus{bus}_filter_off_hz"]
        f_pll = np.interp(t_py, t_emt, emt[f"bus{bus}_emt_pll_hz"])
        f_div = np.interp(t_py, t_emt, emt[f"bus{bus}_emt_sg_divider_hz"])

        post = t_py >= 0.25
        rms_pll = float(np.sqrt(np.mean((f_pll[post] - f_py[post]) ** 2)))
        rms_div = float(np.sqrt(np.mean((f_div[post] - f_py[post]) ** 2)))
        summary_rows.append([
            bus,
            float(np.min(f_py)),
            float(np.min(f_pll)),
            float(np.min(f_div)),
            rms_pll,
            rms_div,
        ])

        c = COLORS[bus]
        ax.plot(t_py, f_py, color=c, linewidth=1.55, label="Reduced analytical")
        ax.plot(t_py, f_div, color=c, linewidth=1.15, linestyle="--", label="EMT SG-divider")
        ax.plot(t_py, f_pll, color="0.25", linewidth=1.10, label="EMT PLL")
        ax.axhline(FREQ_LIMIT, color="#CC6677", linewidth=0.9, linestyle="--")
        ax.axvline(0.25, color="0.45", linewidth=0.8, linestyle=":")
        ax.text(0.985, 0.90, f"Bus {bus}", transform=ax.transAxes, ha="right", va="top")
        ax.set_ylabel(r"$\Delta f$ [Hz]")
        ax.grid(True, alpha=0.20)

    axes[-1].set_xlabel(r"$t$ [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_DIR / "reduced_vs_emt_frequency_diagnostic.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "reduced_vs_emt_frequency_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (OUT_DIR / "frequency_comparison_summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "bus",
            "reduced_nadir_hz",
            "emt_pll_nadir_hz",
            "emt_sg_divider_nadir_hz",
            "rms_emt_pll_minus_reduced_hz",
            "rms_emt_divider_minus_reduced_hz",
        ])
        writer.writerows(summary_rows)

    print(f"Saved comparison outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

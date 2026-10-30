#!/usr/bin/env python3
"""Publication heatmap for the dq0-EMT epsilon sensitivity sweep."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize


# =============================================================================
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SAFETY_CSV_ROOT = SCRIPT_DIR / "plots" / "python_style_cbf" / "csv"
EPSILON_SWEEP_CSV_ROOT = SAFETY_CSV_ROOT / "epsilon_sweep"
EPSILON_HEATMAP_CSV = EPSILON_SWEEP_CSV_ROOT / "emt_epsilon_time_violation_map.csv"
DISTURBANCE_CSV = SAFETY_CSV_ROOT / "disturbance_schedule.csv"
FIGURE_EXPORT_DIR = SCRIPT_DIR / "plots" / "publication_figures"

EPSILON_HEATMAP_TMAX_S = 1.5
EPSILON_HEATMAP_RESAMPLE_DT_S = 0.01
EPSILON_HEATMAP_VIOLATION_TOL_HZ = 1e-6
SHOW_PLOTS = True
USE_EXTERNAL_LATEX = False

# =============================================================================
# Publication style
# =============================================================================

FIGURE_WIDTH = 7.10
AXIS_LW = 0.82
TICK_LW = 0.74
FONT_BASE = 10.0
FONT_XLABEL = 14.0
FONT_YLABEL = 12.8
PNG_DPI = 600
PDF_DPI = 600

COLOR_AXIS = "#343434"
COLOR_INK = "#252525"
COLOR_GRID = "#D8DDE3"
COLOR_EVENT = "#7D8590"
COLOR_LIMIT = "#CC6677"
COLOR_UNSAFE_FILL = "#F5DEDF"


def configure_matplotlib() -> None:
    use_tex = bool(USE_EXTERNAL_LATEX and shutil.which("latex"))
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_YLABEL,
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "axes.linewidth": AXIS_LW,
        "axes.edgecolor": COLOR_AXIS,
        "axes.labelcolor": COLOR_INK,
        "text.color": COLOR_INK,
        "xtick.color": COLOR_AXIS,
        "ytick.color": COLOR_AXIS,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": TICK_LW,
        "ytick.major.width": TICK_LW,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def compact_tick(value, _pos=None) -> str:
    x = float(value)
    a = abs(x)
    if a < 5e-13:
        return "0"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if a >= 0.1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    if a >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if a >= 0.001:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.1e}"


def event_times() -> list[float]:
    if not DISTURBANCE_CSV.is_file():
        return [0.25]
    df = pd.read_csv(DISTURBANCE_CSV)
    if "event_time_s" not in df.columns:
        return [0.25]
    return sorted(float(x) for x in np.unique(df["event_time_s"].to_numpy(float)))


def save_figure(fig, stem: str) -> list[Path]:
    FIGURE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, dpi in (("pdf", PDF_DPI), ("png", PNG_DPI)):
        path = FIGURE_EXPORT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        out.append(path)
        print(f"Saved: {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_heatmap() -> list[Path]:
    if not EPSILON_HEATMAP_CSV.is_file():
        raise FileNotFoundError(
            "EMT epsilon-sweep CSV was not found:\n"
            f"  {EPSILON_HEATMAP_CSV}\n\n"
            "Run ieee39_main_run.jl first."
        )

    df = pd.read_csv(EPSILON_HEATMAP_CSV)
    required = {
        "epsilon_s",
        "t_s",
        "worst_gfl_barrier_hz",
        "safety_violation_hz",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df = df.copy()
    df = df[
        np.isfinite(df["epsilon_s"].to_numpy(float))
        & np.isfinite(df["t_s"].to_numpy(float))
    ]
    df = df[df["t_s"] <= EPSILON_HEATMAP_TMAX_S + 1e-12]
    if df.empty:
        raise ValueError("The EMT epsilon heatmap CSV contains no usable data.")

    eps = np.sort(df["epsilon_s"].unique().astype(float))

    # -----------------------------------------------------------------
    # Reconstruct one common rectangular time grid.
    #
    # EMT/DAE result exports can differ by tiny endpoint roundoff or can
    # contain a few missing samples even when every simulation is nominally
    # exported at the same interval. A heatmap requires a rectangular array,
    # so each epsilon case is interpolated independently onto the same
    # publication grid. This does NOT interpolate in epsilon; only in time.
    # -----------------------------------------------------------------
    dt_plot = float(EPSILON_HEATMAP_RESAMPLE_DT_S)
    if dt_plot <= 0.0:
        raise ValueError("EPSILON_HEATMAP_RESAMPLE_DT_S must be positive.")

    n_steps = int(round(EPSILON_HEATMAP_TMAX_S / dt_plot))
    times = np.linspace(
        0.0,
        float(EPSILON_HEATMAP_TMAX_S),
        n_steps + 1,
        dtype=float,
    )

    violation = np.empty((len(eps), len(times)), dtype=float)
    barrier = np.empty((len(eps), len(times)), dtype=float)

    repaired_cases = []

    def _resample_one_case(case_df: pd.DataFrame, value_col: str, epsilon_s: float):
        local = case_df[["t_s", value_col]].copy()
        local["t_s"] = pd.to_numeric(local["t_s"], errors="coerce")
        local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
        local = local.replace([np.inf, -np.inf], np.nan).dropna()

        if local.empty:
            raise ValueError(
                f"epsilon={epsilon_s:.9g} s has no finite {value_col} samples."
            )

        # Collapse duplicate time stamps, if any.
        local = (
            local.groupby("t_s", as_index=False, sort=True)[value_col]
            .mean()
            .sort_values("t_s")
        )

        t_raw = local["t_s"].to_numpy(dtype=float)
        y_raw = local[value_col].to_numpy(dtype=float)

        if len(t_raw) < 2:
            raise ValueError(
                f"epsilon={epsilon_s:.9g} s has fewer than two finite "
                f"{value_col} samples."
            )

        # np.interp uses endpoint values outside the source interval.
        # Permit this only for small endpoint discrepancies caused by the
        # adaptive EMT solver / CSV export. Large coverage gaps are rejected.
        endpoint_tol = max(2.0 * dt_plot, 5e-4)
        if t_raw[0] > times[0] + endpoint_tol:
            raise ValueError(
                f"epsilon={epsilon_s:.9g} s begins at t={t_raw[0]:.6g} s, "
                f"too late for the requested heatmap grid starting at 0 s."
            )
        if t_raw[-1] < times[-1] - endpoint_tol:
            raise ValueError(
                f"epsilon={epsilon_s:.9g} s ends at t={t_raw[-1]:.6g} s, "
                f"before the requested heatmap end time {times[-1]:.6g} s. "
                "This is a genuinely incomplete EMT simulation, not merely a "
                "non-rectangular CSV grid. Do not extrapolate it in Python; "
                "rerun the Julia epsilon sweep with the epsilon-aware solver "
                "step settings in the updated ieee39_main_run.jl."
            )

        y_new = np.interp(times, t_raw, y_raw)

        if not np.all(np.isfinite(y_new)):
            raise ValueError(
                f"Non-finite values remain after resampling epsilon="
                f"{epsilon_s:.9g} s, column={value_col}."
            )

        return y_new, len(t_raw)

    for i, epsilon_s in enumerate(eps):
        # isclose prevents CSV text-rounding of epsilon from splitting one case.
        mask = np.isclose(
            df["epsilon_s"].to_numpy(dtype=float),
            float(epsilon_s),
            rtol=0.0,
            atol=max(1e-14, 1e-12 * max(1.0, abs(float(epsilon_s)))),
        )
        case_df = df.loc[mask].copy()

        v_i, nv = _resample_one_case(
            case_df,
            "safety_violation_hz",
            float(epsilon_s),
        )
        h_i, nh = _resample_one_case(
            case_df,
            "worst_gfl_barrier_hz",
            float(epsilon_s),
        )

        violation[i, :] = v_i
        barrier[i, :] = h_i

        if nv != len(times) or nh != len(times):
            repaired_cases.append(float(epsilon_s))

    if repaired_cases:
        print(
            "[heatmap] Reconstructed a common time grid for "
            f"{len(repaired_cases)} epsilon case(s)."
        )
        print(
            "[heatmap] This is time interpolation only; epsilon values "
            "are not interpolated."
        )

    tol = float(EPSILON_HEATMAP_VIOLATION_TOL_HZ)
    violation_masked = np.ma.masked_less_equal(violation, tol)

    unsafe_cmap = LinearSegmentedColormap.from_list(
        "paper_unsafe_red",
        [COLOR_UNSAFE_FILL, "#E8AEB4", COLOR_LIMIT, "#8F2E3D"],
        N=256,
    )
    unsafe_cmap.set_bad("white")

    positive = violation[violation > tol]
    vmax = float(np.max(positive)) if positive.size else 1e-4

    Tm, Em = np.meshgrid(times, eps)
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 3.35))

    mesh = ax.pcolormesh(
        Tm,
        Em,
        violation_masked,
        shading="auto",
        cmap=unsafe_cmap,
        norm=Normalize(vmin=0.0, vmax=vmax),
        rasterized=True,
        zorder=1,
    )

    # h=0 gives the actual safe/unsafe demarcation.
    safety_boundary = None
    if np.nanmin(barrier) <= 0.0 <= np.nanmax(barrier):
        safety_boundary = ax.contour(
            Tm,
            Em,
            barrier,
            levels=[0.0],
            colors=[COLOR_INK],
            linewidths=1.15,
            linestyles="-",
            alpha=0.96,
            zorder=4,
        )

    for t_event in event_times():
        if 0.0 <= t_event <= EPSILON_HEATMAP_TMAX_S:
            ax.axvline(
                t_event,
                color=COLOR_EVENT,
                linestyle=(0, (1.0, 1.7)),
                linewidth=0.90,
                alpha=0.90,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_xlim(0.0, EPSILON_HEATMAP_TMAX_S)
    ax.set_ylim(float(np.min(eps)), float(np.max(eps)))
    ax.set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_XLABEL, labelpad=4.0)
    ax.set_ylabel(r"$\epsilon\,[\mathrm{s}]$", fontsize=FONT_YLABEL, labelpad=5.0)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LW)
        spine.set_color(COLOR_AXIS)

    ax.tick_params(
        direction="in",
        top=False,
        right=True,
        labeltop=False,
        labelright=False,
        length=3.7,
        width=TICK_LW,
        pad=2.5,
    )
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.42, alpha=0.42, zorder=0)

    ax.text(
        0.025,
        0.065,
        "Safe region",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=COLOR_INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
        zorder=6,
    )

    if safety_boundary is not None:
        paths = safety_boundary.get_paths()
        if paths:
            verts = max(
                (path.vertices for path in paths if path.vertices.size),
                key=lambda a: a.shape[0],
                default=None,
            )
            if verts is not None and len(verts):
                j = int(0.72 * (len(verts) - 1))
                xb, yb = verts[j]
                ax.annotate(
                    "Safety boundary",
                    xy=(float(xb), float(yb)),
                    xytext=(7, 7),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8.3,
                    color=COLOR_INK,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.84,
                        "pad": 0.7,
                    },
                    zorder=6,
                )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        location="top",
        pad=0.025,
        fraction=0.085,
        aspect=35,
    )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Safety violation [Hz]", fontsize=9.5, labelpad=3.0)
    cbar.ax.tick_params(direction="in", labelsize=8.5, length=3.0, width=0.7, pad=1.8)
    cbar.outline.set_linewidth(0.72)
    cbar.outline.set_edgecolor(COLOR_AXIS)

    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.170, top=0.835)

    print(f"Epsilon cases found: {len(eps)}")
    print(f"Epsilon range: {eps.min():.6g} s to {eps.max():.6g} s")
    print(f"Heatmap time window: 0 to {EPSILON_HEATMAP_TMAX_S:.3f} s")

    return save_figure(fig, "dq0_emt_epsilon_time_safety_violation_heatmap")


def main() -> None:
    configure_matplotlib()
    plot_heatmap()


if __name__ == "__main__":
    main()

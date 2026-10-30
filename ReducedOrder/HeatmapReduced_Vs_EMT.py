#!/usr/bin/env python3
"""
Paper-ready epsilon-sensitivity figure: PEAK SAFETY VIOLATION versus epsilon.

For every filter time constant epsilon of the sweep, the time series of the
worst GFL-bus safety violation
    nu(t; eps) = max{0, max_{i in F} [omega_low - omega_i(t; eps)]}   [Hz]
(column `safety_violation_hz` of the CSVs) is collapsed to its peak over time,
    max_t nu(t; eps),
and plotted against epsilon (log axis) for the reduced model and the dq0-EMT
model on ONE axes. The sampled epsilon interval with no observed violations
in either model is shaded. The figure has NO epsilon-star annotations or
vertical threshold markers, and the horizontal axis is labeled epsilon only.
The shading does not imply an all-time safety guarantee.

Figure geometry and typography follow the manuscript's single-column figures
(3.5 in wide, STIX text and math at printed size, no outer box). Inserted as
    \\includegraphics[width=\\columnwidth]{...}
with no height, so the fonts print at exactly the sizes set below.

The x axis starts exactly at the FIRST epsilon present in the selected CSV
sweep and ends at 1e-1 s. The epsilon grid is read from the CSVs unless
EPSILON_SWEEP_VALUES_S is given.

Inputs (auto-discovered, same as before):
    reduced_epsilon_time_violation_map.csv
    emt_epsilon_time_violation_map.csv
The optional 1x2 heatmap of the previous version is kept behind MAKE_HEATMAP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, MaxNLocator


# =============================================================================
# User settings
# =============================================================================

# Optional direct CSV overrides. Leave as None for auto-discovery.
REDUCED_EPSILON_HEATMAP_CSV_OVERRIDE = None
EMT_EPSILON_HEATMAP_CSV_OVERRIDE = None

# Epsilon values of the sweep. None = take every epsilon present in the CSV
# (recommended: a new sweep is picked up automatically). Give an explicit
# array to enforce a particular grid, e.g.
#     np.unique(np.concatenate([np.geomspace(0.001, 0.1, 19), [0.015]]))
EPSILON_SWEEP_VALUES_S = None

# Peak-figure epsilon range (log scale).
# None means start exactly at the first positive epsilon in the CSV sweep.
# Upper endpoint is fixed at 1e-1 s; any larger epsilon samples are excluded.
PEAK_EPS_RANGE_S = (None, 1.0e-1)

EPSILON_HEATMAP_TMAX_S = 1.5            # peak is taken over 0 <= t <= this
EPSILON_HEATMAP_RESAMPLE_DT_S = 0.01
EPSILON_HEATMAP_VIOLATION_TOL_HZ = 1e-6  # peak below this counts as "no violation"

SHOW_PLOTS = True
SAVE_PDF = True
SAVE_PNG = True
SAVE_SVG = True                           # Editable vector graphics for collaborators
MAKE_PEAK_FIGURE = True                 # the new figure
MAKE_HEATMAP = False                    # the previous 1x2 heatmap (unchanged)

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_EXPORT_DIR = SCRIPT_DIR / "publication_figures"
FIGURE_STEM = "epsilon_sensitivity_peak_violation"
HEATMAP_FIGURE_STEM = "epsilon_sensitivity_reduced_vs_dq0_emt_1x2_common_scale"

# ---- peak-violation figure: geometry (inches) and printed point sizes -------
# Same conventions as the manuscript's 3x1 "EMT model" figure.
PEAK_FIG_WIDTH_IN = 3.5                 # IEEE single column
PEAK_PANEL_HEIGHT_IN = 1.70             # axes height
PEAK_TOP_BAND_IN = 0.08                 # above the axes (no heading)
PEAK_BOTTOM_BAND_IN = 0.34              # tick numbers + epsilon axis label
PEAK_MARGIN_LEFT = 0.150                # figure fraction; auto-widened if a label would leave the page
PEAK_MARGIN_RIGHT = 0.975
PEAK_Y_LABEL_PAD_PT = 1.0               # y label flush against the tick numbers
PEAK_AXIS_LABEL_FONT_PT = 8.4 * 1.20    # axis labels (10.08 pt)
PEAK_TICK_FONT_PT = 7.2 * 1.10          # tick numbers (7.92 pt)
PEAK_LEGEND_FONT_PT = 6.4               # legend / annotations
PEAK_LEGEND_LOC = "upper left"
PAGE_TEXT_PAD_IN = 0.04

# Curves: reduced model black solid + circles, EMT crimson dashed + squares
# (the colour pair used for "reduced" / "EMT" elsewhere in the paper).
PEAK_COLOR_REDUCED = "#1A1A1A"
PEAK_COLOR_EMT = "#C8102E"
PEAK_LW = 0.9
PEAK_MARKER_SIZE = 2.6
PEAK_DASHES_EMT = (3.2, 1.8)
PEAK_SAFE_FILL = "#E4F1E4"              # epsilon range with zero violation (both models)
PEAK_SAFE_ALPHA = 0.65
PEAK_SHOW_SAFE_REGION = True
PEAK_SAFE_LABEL_Y = 0.55                # safe-region label height (fraction of the largest peak)
PEAK_Y_PAD = 0.10                       # y head-room above the largest peak (fraction of span)
PEAK_Y_TICKS = 4

# Text (manuscript notation).
LABEL_EPS = r"$\epsilon$"
LABEL_PEAK = r"$\max_t\,\nu(t;\epsilon)\;[\mathrm{Hz}]$"
LABEL_REDUCED = "Reduced model"
LABEL_EMT = r"EMT model"
LABEL_SAFE = "No observed\nsafety violations"

# ---- heatmap settings (previous figure, kept) -------------------------------
COMMON_VMIN_HZ = 0.0
COMMON_VMAX_HZ = 0.25
PNG_DPI = 600
PDF_DPI = 600

COLOR_AXIS = "#000000"
COLOR_INK = "#1A1A1A"
COLOR_GRID = "0.82"
COLOR_EVENT = "0.45"
TEXT_COLOR = "0.10"


# =============================================================================
# Publication style (STIX text + STIX math, TrueType embedding, printed sizes)
# =============================================================================

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        "font.size": 8.0,
        "axes.labelsize": PEAK_AXIS_LABEL_FONT_PT,
        "xtick.labelsize": PEAK_TICK_FONT_PT,
        "ytick.labelsize": PEAK_TICK_FONT_PT,
        "legend.fontsize": PEAK_LEGEND_FONT_PT,
        "axes.linewidth": 0.5,
        "axes.edgecolor": COLOR_AXIS,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": COLOR_AXIS,
        "ytick.color": COLOR_AXIS,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.size": 1.3,
        "ytick.minor.size": 1.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.pad": 1.6,
        "ytick.major.pad": 1.4,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Keep ordinary labels as editable text objects in SVG files.
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": None,           # exact figure size: 1:1 at width=\columnwidth
        "savefig.pad_inches": 0.0,
    })


# =============================================================================
# Helpers
# =============================================================================

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


def heatmap_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.00, "#FFFFFF"), (0.03, "#F8ECEC"), (0.15, "#EEC9CD"), (0.35, "#E29CA8"),
        (0.55, "#D66F83"), (0.75, "#C64E66"), (1.00, "#9B2438"),
    ]
    return LinearSegmentedColormap.from_list("unsafe_heatmap", colors)


def script_search_roots() -> list[Path]:
    roots = [SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd()]
    out, seen = [], set()
    for r in roots:
        key = str(r.resolve()).lower() if r.exists() else str(r).lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def discover_csv(filename: str, override: str | None) -> Path:
    if override is not None:
        path = Path(override).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"CSV override does not exist:\n  {path}")

    candidates = []
    for root in script_search_roots():
        candidates.extend([
            root / filename,
            root / "csv_exports_normal" / filename,
            root / "plots" / "python_style_cbf" / "csv" / "epsilon_sweep" / filename,
            root / "Full_order" / "plots" / "python_style_cbf" / "csv" / "epsilon_sweep" / filename,
            root / "Full_Order" / "plots" / "python_style_cbf" / "csv" / "epsilon_sweep" / filename,
            root / "ReducedOrder" / "csv_exports_normal" / filename,
            root / "Reduced_Order" / "csv_exports_normal" / filename,
        ])
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    for root in script_search_roots():
        try:
            hits = list(root.rglob(filename))
        except (OSError, PermissionError):
            hits = []
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find required CSV:\n  {filename}\n"
        "Set the corresponding *_CSV_OVERRIDE path explicitly."
    )


def load_heatmap_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unchanged loader: (eps, times, barrier[eps, t], violation[eps, t])."""
    df = pd.read_csv(csv_path)
    required = {"epsilon_s", "t_s", "worst_gfl_barrier_hz", "safety_violation_hz"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    df = df.copy()
    for col in ("epsilon_s", "t_s", "worst_gfl_barrier_hz", "safety_violation_hz"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["t_s"] <= EPSILON_HEATMAP_TMAX_S + 1e-12]
    if df.empty:
        raise ValueError(f"{csv_path.name} contains no usable data in the requested time window.")

    if EPSILON_SWEEP_VALUES_S is None:
        eps = np.unique(np.round(df["epsilon_s"].to_numpy(dtype=float), 12))
    else:
        eps = np.asarray(EPSILON_SWEEP_VALUES_S, dtype=float)
    dt_plot = float(EPSILON_HEATMAP_RESAMPLE_DT_S)
    n_steps = int(round(EPSILON_HEATMAP_TMAX_S / dt_plot))
    times = np.linspace(0.0, EPSILON_HEATMAP_TMAX_S, n_steps + 1, dtype=float)
    barrier = np.empty((len(eps), len(times)), dtype=float)
    violation = np.empty((len(eps), len(times)), dtype=float)

    def _resample_one_case(case_df: pd.DataFrame, value_col: str, epsilon_s: float) -> np.ndarray:
        local = case_df[["t_s", value_col]].copy()
        local = local.groupby("t_s", as_index=False, sort=True)[value_col].mean().sort_values("t_s")
        t_raw = local["t_s"].to_numpy(dtype=float)
        y_raw = local[value_col].to_numpy(dtype=float)
        if len(t_raw) < 2:
            raise ValueError(f"{csv_path.name}: epsilon={epsilon_s:.9g} has fewer than two samples for {value_col}.")
        endpoint_tol = max(2.0 * dt_plot, 5e-4)
        if t_raw[0] > times[0] + endpoint_tol:
            raise ValueError(f"{csv_path.name}: epsilon={epsilon_s:.9g} begins too late (t0={t_raw[0]:.6g} s).")
        if t_raw[-1] < times[-1] - endpoint_tol:
            raise ValueError(f"{csv_path.name}: epsilon={epsilon_s:.9g} ends too early "
                             f"(t_end={t_raw[-1]:.6g} s < {times[-1]:.6g} s).")
        y_new = np.interp(times, t_raw, y_raw)
        if not np.all(np.isfinite(y_new)):
            raise ValueError(f"{csv_path.name}: non-finite values remain after resampling "
                             f"epsilon={epsilon_s:.9g}, column={value_col}.")
        return y_new

    for i, epsilon_s in enumerate(eps):
        mask = np.isclose(df["epsilon_s"].to_numpy(dtype=float), float(epsilon_s), rtol=0.0,
                          atol=max(1e-14, 1e-12 * max(1.0, abs(float(epsilon_s)))))
        case_df = df.loc[mask].copy()
        if case_df.empty:
            raise ValueError(f"{csv_path.name}: epsilon={epsilon_s:.9g} is missing from the CSV.")
        violation[i, :] = _resample_one_case(case_df, "safety_violation_hz", float(epsilon_s))
        barrier[i, :] = _resample_one_case(case_df, "worst_gfl_barrier_hz", float(epsilon_s))
    return eps, times, barrier, violation


def peak_violation(violation: np.ndarray, tol: float) -> np.ndarray:
    """max_t nu(t; eps) per epsilon row; values below tol are reported as exactly 0."""
    peak = np.max(violation, axis=1)
    peak[peak <= tol] = 0.0
    return peak


def fit_left_margin(fig, axes, min_x_in: float) -> None:
    """Shift the axes right so that no y label / tick number starts left of
    min_x_in (inches from the page edge); measured on the drawn figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xs = []
    for ax in axes:
        xs.append(ax.yaxis.label.get_window_extent(r).x0)
        for lab in ax.get_yticklabels():
            if lab.get_visible() and lab.get_text():
                xs.append(lab.get_window_extent(r).x0)
    need = min_x_in * fig.dpi - min(xs) if xs else 0.0
    if need > 0:
        dx = need / (fig.get_figwidth() * fig.dpi)
        for ax in axes:
            x0, y0, w, h = ax.get_position().bounds
            ax.set_position((x0 + dx, y0, w - dx, h))
        fig.canvas.draw()


def save_figure(fig, stem: str, *, exact_size: bool) -> list[Path]:
    FIGURE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    kw = {} if exact_size else {"bbox_inches": "tight", "pad_inches": 0.035}
    if SAVE_PDF:
        pdf_path = FIGURE_EXPORT_DIR / f"{stem}.pdf"
        fig.savefig(pdf_path, dpi=PDF_DPI, facecolor="white", **kw)
        saved.append(pdf_path)
        print(f"Saved: {pdf_path}")
    if SAVE_PNG:
        png_path = FIGURE_EXPORT_DIR / f"{stem}.png"
        fig.savefig(png_path, dpi=PNG_DPI, facecolor="white", **kw)
        saved.append(png_path)
        print(f"Saved: {png_path}")
    if SAVE_SVG:
        svg_path = FIGURE_EXPORT_DIR / f"{stem}.svg"
        fig.savefig(svg_path, format="svg", facecolor="white", **kw)
        saved.append(svg_path)
        print(f"Saved: {svg_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return saved


def load_both() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reduced_csv = discover_csv("reduced_epsilon_time_violation_map.csv", REDUCED_EPSILON_HEATMAP_CSV_OVERRIDE)
    emt_csv = discover_csv("emt_epsilon_time_violation_map.csv", EMT_EPSILON_HEATMAP_CSV_OVERRIDE)
    print(f"[Reduced] Using CSV: {reduced_csv}")
    print(f"[EMT]     Using CSV: {emt_csv}")
    eps_r, times_r, barrier_r, violation_r = load_heatmap_csv(reduced_csv)
    eps_e, times_e, barrier_e, violation_e = load_heatmap_csv(emt_csv)
    if len(eps_r) != len(eps_e) or not np.allclose(eps_r, eps_e, atol=1e-12, rtol=0.0):
        raise ValueError("Reduced and EMT epsilon grids do not match.")
    if len(times_r) != len(times_e) or not np.allclose(times_r, times_e, atol=1e-12, rtol=0.0):
        raise ValueError("Reduced and EMT time grids do not match.")
    return eps_r, times_r, barrier_r, violation_r, barrier_e, violation_e


# =============================================================================
# Peak-violation figure
# =============================================================================

def plot_peak_violation_vs_epsilon(eps: np.ndarray, violation_r: np.ndarray,
                                   violation_e: np.ndarray) -> list[Path]:
    tol = float(EPSILON_HEATMAP_VIOLATION_TOL_HZ)
    configured_lo, configured_hi = PEAK_EPS_RANGE_S
    hi_e = float(configured_hi)
    if hi_e <= 0.0:
        raise ValueError("The logarithmic x-axis upper limit must be positive.")
    # Use the first *actual* positive sampled epsilon as the left x-axis
    # endpoint, rather than padding the axis down to 1e-3 s.
    # An explicitly configured lower limit still filters the samples, but
    # the displayed lower limit always equals the first retained sample.
    keep = (np.isfinite(eps) & (eps > 0.0) &
            (eps <= hi_e * (1 + 1e-9)))
    if configured_lo is not None:
        keep &= eps >= float(configured_lo) * (1 - 1e-9)
    if not keep.any():
        raise ValueError(f"No positive sweep epsilon inside PEAK_EPS_RANGE_S = {PEAK_EPS_RANGE_S}; "
                         f"CSV epsilon range is [{eps.min():.4g}, {eps.max():.4g}] s.")
    eps = eps[keep]
    violation_r = violation_r[keep]
    violation_e = violation_e[keep]
    lo_e = float(np.min(eps))
    if lo_e >= hi_e:
        raise ValueError(f"Need at least one epsilon below {hi_e:g} s for a logarithmic x-axis. ")
    print(f"[peak] Displaying epsilon range [{lo_e:.6g}, {hi_e:.6g}] s; "
          f"left edge equals first CSV sample.")
    peak_r = peak_violation(violation_r, tol)
    peak_e = peak_violation(violation_e, tol)
    for e, pr, pe in zip(eps, peak_r, peak_e):
        print(f"       eps = {e:8.4g} s   reduced = {pr:.4f} Hz   EMT = {pe:.4f} Hz")

    W = PEAK_FIG_WIDTH_IN
    height = PEAK_PANEL_HEIGHT_IN + PEAK_TOP_BAND_IN + PEAK_BOTTOM_BAND_IN
    fig = plt.figure(figsize=(W, height), facecolor="white")
    ax = fig.add_axes((PEAK_MARGIN_LEFT, PEAK_BOTTOM_BAND_IN / height,
                       PEAK_MARGIN_RIGHT - PEAK_MARGIN_LEFT, PEAK_PANEL_HEIGHT_IN / height))

    # Curves (markers = the simulated epsilon values).
    ax.plot(eps, peak_r, color=PEAK_COLOR_REDUCED, lw=PEAK_LW, marker="o", ms=PEAK_MARKER_SIZE,
            mfc=PEAK_COLOR_REDUCED, mec=PEAK_COLOR_REDUCED, zorder=4)
    ax.plot(eps, peak_e, color=PEAK_COLOR_EMT, lw=PEAK_LW, ls="--", dashes=PEAK_DASHES_EMT,
            marker="s", ms=PEAK_MARKER_SIZE - 0.2, mfc=PEAK_COLOR_EMT, mec=PEAK_COLOR_EMT, zorder=5)

    # Axes ranges.
    ax.set_xscale("log")
    ax.set_xlim(lo_e, hi_e)
    top = max(float(peak_r.max()), float(peak_e.max()), 1e-3)
    ax.set_ylim(-0.03 * top, top * (1.0 + PEAK_Y_PAD))

    # Shade only the leading interval where BOTH models have zero observed
    # violations at each sampled epsilon. This does not assert safety for every
    # unsampled epsilon between those points, or outside the simulation window.
    # No threshold markers or epsilon-star labels are drawn.
    if PEAK_SHOW_SAFE_REGION:
        jointly_nonviolating = (peak_r == 0.0) & (peak_e == 0.0)
        first_violating_indices = np.flatnonzero(~jointly_nonviolating)
        if len(first_violating_indices) == 0:
            safe_edge = float(eps[-1])
        elif first_violating_indices[0] == 0:
            safe_edge = None
        else:
            safe_edge = float(eps[first_violating_indices[0] - 1])
        if safe_edge is not None and safe_edge > lo_e:
            ax.axvspan(lo_e, safe_edge, color=PEAK_SAFE_FILL,
                       alpha=PEAK_SAFE_ALPHA, lw=0, zorder=0.5)
            ax.text(np.sqrt(lo_e * safe_edge), PEAK_SAFE_LABEL_Y * top,
                    LABEL_SAFE, ha="center", va="center",
                    multialignment="center", fontsize=PEAK_LEGEND_FONT_PT - 0.5,
                    linespacing=0.95, color="0.35", zorder=6)

    # Ticks, grid, labels (same style as the manuscript's time-domain figures).
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=PEAK_Y_TICKS, min_n_ticks=3, steps=[1, 2, 2.5, 5]))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True,
                   labelsize=PEAK_TICK_FONT_PT, length=2.2, width=0.5, pad=1.8)
    ax.tick_params(axis="x", which="minor", direction="in", top=True, length=1.3, width=0.4)
    ax.grid(True, which="major", color=COLOR_GRID, alpha=0.35, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="0.6", lw=0.45, zorder=1)
    ax.set_xlabel(LABEL_EPS, fontsize=PEAK_AXIS_LABEL_FONT_PT, labelpad=1.0)
    ax.set_ylabel(LABEL_PEAK, fontsize=PEAK_AXIS_LABEL_FONT_PT, labelpad=PEAK_Y_LABEL_PAD_PT)

    # Legend inside the axes (upper left is empty: the curves rise to the right).
    handles = [
        Line2D([0], [0], color=PEAK_COLOR_REDUCED, lw=PEAK_LW, marker="o", ms=PEAK_MARKER_SIZE, label=LABEL_REDUCED),
        Line2D([0], [0], color=PEAK_COLOR_EMT, lw=PEAK_LW, ls="--", dashes=PEAK_DASHES_EMT, marker="s",
               ms=PEAK_MARKER_SIZE - 0.2, label=LABEL_EMT),
    ]
    leg = ax.legend(handles=handles, loc=PEAK_LEGEND_LOC, frameon=True, fancybox=False, framealpha=0.92,
                    prop={"size": PEAK_LEGEND_FONT_PT}, handlelength=2.2, handletextpad=0.5,
                    borderpad=0.3, labelspacing=0.3, borderaxespad=0.35)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.set_zorder(7)

    fit_left_margin(fig, [ax], PAGE_TEXT_PAD_IN)
    return save_figure(fig, FIGURE_STEM, exact_size=True)


# =============================================================================
# Previous 1x2 heatmap (kept, unchanged in content; off by default)
# =============================================================================

def plot_reduced_vs_emt_heatmap(eps, times, violation_r, violation_e) -> list[Path]:
    cmap = heatmap_cmap()
    norm = Normalize(vmin=COMMON_VMIN_HZ, vmax=COMMON_VMAX_HZ, clip=True)
    tol = float(EPSILON_HEATMAP_VIOLATION_TOL_HZ)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.8, wspace=0.08)
    for ax, violation, title in zip(axes, (violation_r, violation_e), ("(a) Reduced model", r"(b) $dq0$-EMT model")):
        mesh = ax.pcolormesh(times, eps, np.ma.masked_less_equal(violation, tol), shading="auto",
                             cmap=cmap, norm=norm, rasterized=True, zorder=2)
        ax.axvline(0.25, color=COLOR_EVENT, linestyle=(0, (1.0, 1.6)), linewidth=0.5, zorder=3)
        ax.set_xlim(0.0, EPSILON_HEATMAP_TMAX_S)
        ax.set_ylim(float(eps.min()), float(eps.max()))
        ax.set_yscale("log")
        ax.set_xlabel(r"$t\;[\mathrm{s}]$", labelpad=1.0)
        ax.set_title(title, fontsize=7.15)
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
        ax.grid(True, color=COLOR_GRID, alpha=0.35, linewidth=0.3)
    axes[0].set_ylabel(LABEL_EPS, labelpad=1.0)
    cb = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.02, location="top")
    cb.set_label("Safety violation [Hz]", fontsize=7.15)
    cb.ax.tick_params(labelsize=PEAK_TICK_FONT_PT, length=2.0, width=0.5)
    return save_figure(fig, HEATMAP_FIGURE_STEM, exact_size=False)


def main() -> None:
    configure_matplotlib()
    eps, times, _barrier_r, violation_r, _barrier_e, violation_e = load_both()
    if MAKE_PEAK_FIGURE:
        plot_peak_violation_vs_epsilon(eps, violation_r, violation_e)
    if MAKE_HEATMAP:
        plot_reduced_vs_emt_heatmap(eps, times, violation_r, violation_e)


if __name__ == "__main__":
    main()
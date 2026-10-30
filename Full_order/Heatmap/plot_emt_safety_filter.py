#!/usr/bin/env python3
"""Publication plots from Julia dq0-EMT safety-filter CSV exports."""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize


# =============================================================================
# Paths and selections
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
JULIA_ROOT = SCRIPT_DIR
SAFETY_CSV_ROOT = JULIA_ROOT / "plots" / "python_style_cbf" / "csv"
FIGURE_EXPORT_DIR = JULIA_ROOT / "plots" / "publication_figures"
EPSILON_SWEEP_CSV_ROOT = SAFETY_CSV_ROOT / "epsilon_sweep"
EPSILON_HEATMAP_CSV = EPSILON_SWEEP_CSV_ROOT / "emt_epsilon_time_violation_map.csv"

GFL_BUSES = [30, 34, 36, 31, 38]
SG_BUSES = [32, 33, 35, 37, 39]
RESOURCE_FREQUENCY_BUSES = [30, 31, 34,36,38, 32, 33, 35, 37, 39]

FREQ_LOWER_HZ = -0.07
FREQ_UPPER_HZ = +0.07
ACTIVE_SAFETY_BOUND = "auto"

SHOW_PLOTS = True
USE_EXTERNAL_LATEX = False

MAKE_FREQUENCY_SAFETY_FIGURE = True
MAKE_BARRIER_FIGURE = True
MAKE_FILTER_STATE_FIGURE = True
MAKE_GFL_POWER_DECOMPOSITION_FIGURE = True
MAKE_EPSILON_VIOLATION_HEATMAP = True

EPSILON_HEATMAP_TMAX_S = 1.5
EPSILON_HEATMAP_VIOLATION_TOL_HZ = 1e-6


# =============================================================================
# Publication style: matched to the reduced-order plotting code
# =============================================================================

FIGURE_WIDTH = 7.10
AXIS_LW = 0.82
TICK_LW = 0.74
FONT_BASE = 10.0
FONT_XLABEL = 14.0
FONT_YLABEL = 12.8
FONT_LEGEND = 9.5
FONT_STATUS = 9.8
PNG_DPI = 600
PDF_DPI = 600

COLOR_AXIS = "#343434"
COLOR_INK = "#252525"
COLOR_GRID = "#D8DDE3"
COLOR_EVENT = "#7D8590"
COLOR_ZERO = "#98A1AA"
COLOR_LIMIT = "#CC6677"
COLOR_FILTER = "#D55E00"
COLOR_APPLIED = "#009E73"
COLOR_FFR = "#0072B2"
COLOR_FILL = "#E69F00"
COLOR_UNSAFE_FILL = "#F5DEDF"

BUS_COLORS = {
    30: "#0072B2",
    31: "#009E73",
    32: "#E69F00",
    33: "#3A923A",
    34: "#56B4E9",
    35: "#CC6677",
    36: "#D55E00",
    37: "#AA4499",
    38: "#7E57C2",
    39: "#008B8B",
}


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
        "legend.fontsize": FONT_LEGEND,
        "axes.linewidth": AXIS_LW,
        "axes.edgecolor": COLOR_AXIS,
        "axes.labelcolor": COLOR_INK,
        "text.color": COLOR_INK,
        "xtick.color": COLOR_AXIS,
        "ytick.color": COLOR_AXIS,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": TICK_LW,
        "ytick.major.width": TICK_LW,
        "axes.unicode_minus": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _compact_tick(value, _pos=None):
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


# =============================================================================
# CSV helpers
# =============================================================================

def ensure_dir(path) -> None:
    os.makedirs(os.fspath(path), exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def case_dir(tag: str) -> Path:
    return SAFETY_CSV_ROOT / tag


def gfl_path(tag: str, bus: int) -> Path:
    return case_dir(tag) / f"gfl_bus{bus:02d}.csv"


def sg_path(tag: str, bus: int) -> Path:
    return case_dir(tag) / f"sg_bus{bus:02d}.csv"


def required_gfl_paths(tag: str) -> list[Path]:
    return [gfl_path(tag, b) for b in GFL_BUSES]


def missing_case_files(tag: str) -> list[Path]:
    return [p for p in required_gfl_paths(tag) if not p.is_file()]


def case_available(tag: str) -> bool:
    return case_dir(tag).is_dir() and not missing_case_files(tag)


def matched_cases_available() -> bool:
    return case_available("filter_off") and case_available("filter_on")


def gfl_df(tag: str, bus: int) -> pd.DataFrame:
    return read_csv(gfl_path(tag, bus))


def sg_df(tag: str, bus: int) -> pd.DataFrame:
    return read_csv(sg_path(tag, bus))


def event_times() -> list[float]:
    path = SAFETY_CSV_ROOT / "disturbance_schedule.csv"
    if not path.is_file():
        return [0.25]
    df = pd.read_csv(path)
    if "event_time_s" not in df.columns:
        return [0.25]
    return sorted(float(x) for x in np.unique(df["event_time_s"].to_numpy()))


def frequency_trace(tag: str, bus: int) -> tuple[np.ndarray, np.ndarray]:
    df = gfl_df(tag, bus) if bus in GFL_BUSES else sg_df(tag, bus)
    return (
        df["t_s"].to_numpy(float),
        df["frequency_deviation_hz"].to_numpy(float),
    )


def _time_limits(tags: list[str], buses: list[int]) -> tuple[float, float]:
    lo = np.inf
    hi = -np.inf
    for tag in tags:
        for bus in buses:
            path = gfl_path(tag, bus) if bus in GFL_BUSES else sg_path(tag, bus)
            if not path.is_file():
                continue
            t, _ = frequency_trace(tag, bus)
            if len(t):
                lo = min(lo, float(np.nanmin(t)))
                hi = max(hi, float(np.nanmax(t)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    return lo, hi


# =============================================================================
# Reduced-order-matched plotting helpers
# =============================================================================

def _style_axis(ax, xlim: tuple[float, float], ybins: int = 5, zero: bool = False):
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LW)
        spine.set_color(COLOR_AXIS)

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        length=3.7,
        width=TICK_LW,
        pad=2.5,
    )
    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.42,
        alpha=0.40,
        zorder=-5,
    )
    ax.margins(x=0.0)

    if zero:
        ax.axhline(0.0, color=COLOR_ZERO, linewidth=0.72, zorder=0)


def _event_lines(ax, events: list[float]):
    for t_ev in sorted(set(float(t) for t in events)):
        ax.axvline(
            t_ev,
            color=COLOR_EVENT,
            linestyle=(0, (1.0, 1.7)),
            linewidth=0.78,
            alpha=0.88,
            zorder=0,
        )


def _set_ylim(ax, arrays, include_zero=False, lower_pad=0.10, upper_pad=0.14):
    vals = []
    for a in arrays:
        x = np.asarray(a, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)

    if include_zero:
        vals.append(np.array([0.0]))

    if not vals:
        return

    x = np.concatenate(vals)
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo

    if span < 1e-8:
        span = max(abs(lo), abs(hi), 1.0) * 0.05
        lo -= span
        hi += span
        span = hi - lo

    ax.set_ylim(lo - lower_pad * span, hi + upper_pad * span)


def _safe_save(fig, path, **kwargs) -> Path | None:
    """
    Save robustly on Windows/OneDrive.

    If Windows/Pillow rejects an existing PNG path, the figure is first written
    to a local temporary file and then copied back. If that exact destination is
    still unavailable, an automatically numbered filename is used instead.
    """
    path = Path(os.path.normpath(os.fspath(path)))
    ensure_dir(path.parent)

    try:
        fig.savefig(str(path), **kwargs)
        return path
    except OSError as first_error:
        warnings.warn(
            f"Direct save failed for {path}: {first_error}. "
            "Retrying through a temporary local file."
        )

    suffix = path.suffix if path.suffix else ".tmp"
    fd, tmp_name = tempfile.mkstemp(prefix="cbf_plot_", suffix=suffix)
    os.close(fd)
    tmp = Path(tmp_name)

    try:
        fig.savefig(str(tmp), **kwargs)

        try:
            shutil.copyfile(str(tmp), str(path))
            return path
        except OSError:
            # A stale/open OneDrive file can block replacement of only one name.
            for idx in range(1, 100):
                alt = path.with_name(f"{path.stem}_{idx}{path.suffix}")
                try:
                    shutil.copyfile(str(tmp), str(alt))
                    warnings.warn(
                        f"Could not write {path.name}; saved as {alt.name} instead."
                    )
                    return alt
                except OSError:
                    continue

            warnings.warn(
                f"Could not copy the figure into {path.parent}. "
                f"A temporary copy remains at {tmp}."
            )
            return tmp
    finally:
        # Remove the temporary file only when it has been copied successfully.
        try:
            if tmp.exists():
                # If we returned tmp itself above, preserve it.
                same_as_tmp = False
                # This branch is intentionally conservative.
                if path.exists() or any(
                    path.with_name(f"{path.stem}_{i}{path.suffix}").exists()
                    for i in range(1, 100)
                ):
                    tmp.unlink(missing_ok=True)
        except OSError:
            pass


def save_figure(fig, stem: str) -> list[Path]:
    ensure_dir(FIGURE_EXPORT_DIR)
    common = dict(
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )

    saved = []
    pdf = _safe_save(
        fig,
        FIGURE_EXPORT_DIR / f"{stem}.pdf",
        format="pdf",
        dpi=PDF_DPI,
        **common,
    )
    if pdf is not None:
        saved.append(pdf)
        print(f"Saved: {pdf}")

    png = _safe_save(
        fig,
        FIGURE_EXPORT_DIR / f"{stem}.png",
        format="png",
        dpi=PNG_DPI,
        **common,
    )
    if png is not None:
        saved.append(png)
        print(f"Saved: {png}")

    # Match the reduced-order code: show each completed figure as well as save it.
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return saved


def _resource_style(bus: int):
    if bus in GFL_BUSES:
        return "-", None, 1.50
    if bus in SG_BUSES:
        return "--", (5.5, 2.8), 1.45
    return "-.", None, 1.45


def _active_side(tags: list[str]) -> str:
    arrays = []
    for tag in tags:
        if not case_available(tag):
            continue
        for bus in GFL_BUSES:
            _, y = frequency_trace(tag, bus)
            arrays.append(y)

    if not arrays:
        return "lower"

    data = np.concatenate(arrays)
    mode = str(ACTIVE_SAFETY_BOUND).strip().lower()

    if mode in {"lower", "upper"}:
        return mode
    if mode != "auto":
        raise ValueError('ACTIVE_SAFETY_BOUND must be "auto", "lower", or "upper".')

    fmin = float(np.min(data))
    fmax = float(np.max(data))
    lower = max(0.0, -fmin) / max(abs(FREQ_LOWER_HZ), 1e-12)
    upper = max(0.0, fmax) / max(abs(FREQ_UPPER_HZ), 1e-12)
    return "lower" if lower >= upper else "upper"


def _frequency_bound(ax, side: str):
    value = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    label = r"$\Delta f_{\min}$" if side == "lower" else r"$\Delta f_{\max}$"

    ax.axhline(
        value,
        color=COLOR_LIMIT,
        linestyle=(0, (4.0, 2.2)),
        linewidth=1.0,
        zorder=1,
    )
    ax.text(
        0.986,
        value,
        label,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom" if side == "lower" else "top",
        fontsize=8.2,
        color=COLOR_LIMIT,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 0.6,
        },
    )


def _status(ax, enabled: bool):
    ax.text(
        0.985,
        0.955,
        "Filter ON" if enabled else "Filter OFF",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_STATUS,
        color=COLOR_APPLIED if enabled else COLOR_FILTER,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "pad": 0.6,
        },
    )


def _dqemt_label(ax):
    # This is the only visual identifier that distinguishes these figures
    # from the reduced-order publication plots.
    ax.text(
        0.015,
        0.955,
        r"$dq0$-EMT",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_STATUS,
        color=COLOR_INK,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.6,
        },
    )


# =============================================================================
# Publication figures
# =============================================================================

def plot_frequency_safety(events: list[float]) -> list[Path]:
    compare = matched_cases_available()
    tags = ["filter_off", "filter_on"] if compare else [
        "filter_on" if case_available("filter_on") else "filter_off"
    ]
    side = _active_side(tags)
    xlim = _time_limits(tags, RESOURCE_FREQUENCY_BUSES)

    rows = 2 if compare else 1
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(FIGURE_WIDTH, 4.85 if compare else 2.95),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    handles = []
    labels = []
    all_arrays = []

    for bus in RESOURCE_FREQUENCY_BUSES:
        path_off = gfl_path("filter_off", bus) if bus in GFL_BUSES else sg_path("filter_off", bus)
        path_on = gfl_path("filter_on", bus) if bus in GFL_BUSES else sg_path("filter_on", bus)

        if compare and (not path_off.is_file() or not path_on.is_file()):
            continue

        if compare:
            t_off, f_off = frequency_trace("filter_off", bus)
            t_on, f_on = frequency_trace("filter_on", bus)
        else:
            tag = tags[0]
            t_on, f_on = frequency_trace(tag, bus)
            t_off, f_off = t_on, f_on

        ls, dashes, lw = _resource_style(bus)
        kwargs = dict(
            color=BUS_COLORS.get(bus, COLOR_INK),
            linestyle=ls,
            linewidth=lw,
        )
        if dashes is not None:
            kwargs["dashes"] = dashes

        line, = axes[0].plot(
            t_off if compare else t_on,
            f_off if compare else f_on,
            **kwargs,
        )
        if compare:
            axes[1].plot(t_on, f_on, **kwargs)

        handles.append(line)
        labels.append(f"Bus {bus} ({'GFL' if bus in GFL_BUSES else 'SG'})")
        all_arrays.extend([f_off, f_on])

    bound = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    vals = np.concatenate(
        [np.asarray(a) for a in all_arrays]
        + [np.array([0.0, bound])]
    )
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(hi - lo, 1e-3)
    ylim = (lo - 0.06 * span, hi + 0.08 * span)

    for i, ax in enumerate(axes):
        _frequency_bound(ax, side)
        _event_lines(ax, events)
        _style_axis(ax, xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(
            r"$\Delta f_i\,[\mathrm{Hz}]$",
            fontsize=FONT_YLABEL,
            labelpad=5.0,
        )
        _status(ax, False if compare and i == 0 else True)
        _dqemt_label(ax)

    axes[-1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(5, max(1, len(labels))),
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.105 if compare else 0.17,
        top=0.80 if compare else 0.76,
        hspace=0.065,
    )

    return save_figure(
        fig,
        "dq0_emt_frequency_safety_filter_off_on"
        if compare
        else "dq0_emt_frequency_safety",
    )


def _barrier_from_df(df: pd.DataFrame, side: str) -> np.ndarray:
    if "active_barrier_hz" in df.columns:
        return df["active_barrier_hz"].to_numpy(float)

    freq = df["frequency_deviation_hz"].to_numpy(float)
    if side == "lower":
        return freq - FREQ_LOWER_HZ
    return FREQ_UPPER_HZ - freq


def plot_barrier_safety(events: list[float]) -> list[Path]:
    compare = matched_cases_available()
    tags = ["filter_off", "filter_on"] if compare else [
        "filter_on" if case_available("filter_on") else "filter_off"
    ]
    side = _active_side(tags)
    xlim = _time_limits(tags, GFL_BUSES)

    rows = 2 if compare else 1
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(FIGURE_WIDTH, 4.55 if compare else 2.80),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    handles = []
    labels = []
    arrays = []

    for bus in GFL_BUSES:
        if compare:
            df0 = gfl_df("filter_off", bus)
            df1 = gfl_df("filter_on", bus)
            h0 = _barrier_from_df(df0, side)
            h1 = _barrier_from_df(df1, side)
            t0 = df0["t_s"].to_numpy(float)
            t1 = df1["t_s"].to_numpy(float)
        else:
            tag = tags[0]
            df1 = gfl_df(tag, bus)
            h1 = _barrier_from_df(df1, side)
            t1 = df1["t_s"].to_numpy(float)
            h0 = h1
            t0 = t1

        color = BUS_COLORS.get(bus, COLOR_INK)
        line, = axes[0].plot(
            t0 if compare else t1,
            h0 if compare else h1,
            color=color,
            linewidth=1.50,
        )
        if compare:
            axes[1].plot(t1, h1, color=color, linewidth=1.50)

        handles.append(line)
        labels.append(f"Bus {bus}")
        arrays.extend([h0, h1])

    tmp = np.concatenate(arrays + [np.array([0.0])])
    lo = float(np.min(tmp))
    hi = float(np.max(tmp))
    span = max(hi - lo, 1e-3)
    ylim = (lo - 0.10 * span, hi + 0.12 * span)

    ylabel = (
        r"$h_i^{\min}\,[\mathrm{Hz}]$"
        if side == "lower"
        else r"$h_i^{\max}\,[\mathrm{Hz}]$"
    )

    for i, ax in enumerate(axes):
        ax.axhspan(
            ylim[0],
            0.0,
            color=COLOR_UNSAFE_FILL,
            alpha=0.45,
            linewidth=0.0,
            zorder=-4,
        )
        ax.axhline(
            0.0,
            color=COLOR_LIMIT,
            linestyle=(0, (3.0, 2.0)),
            linewidth=0.95,
        )
        _event_lines(ax, events)
        _style_axis(ax, xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=FONT_YLABEL, labelpad=5.0)
        _status(ax, False if compare and i == 0 else True)
        _dqemt_label(ax)

    axes[-1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=FONT_LEGEND,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.105 if compare else 0.17,
        top=0.84 if compare else 0.80,
        hspace=0.065,
    )

    return save_figure(
        fig,
        "dq0_emt_gfl_barrier_safety_filter_off_on"
        if compare
        else "dq0_emt_gfl_barrier_safety",
    )


def plot_filter_state(events: list[float]) -> list[Path]:
    if not case_available("filter_on"):
        return []

    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.80))
    handles = []
    labels = []
    arrays = []
    xlo = np.inf
    xhi = -np.inf

    for bus in GFL_BUSES:
        df = gfl_df("filter_on", bus)
        t = df["t_s"].to_numpy(float)
        z = df["z_command_system_pu"].to_numpy(float)

        line, = ax.plot(
            t,
            z,
            color=BUS_COLORS.get(bus, COLOR_INK),
            linewidth=1.52,
        )
        handles.append(line)
        labels.append(f"Bus {bus}")
        arrays.append(z)

        if len(t):
            xlo = min(xlo, float(np.nanmin(t)))
            xhi = max(xhi, float(np.nanmax(t)))

    xlim = (xlo, xhi) if np.isfinite(xlo) and np.isfinite(xhi) else (0.0, 1.0)

    _event_lines(ax, events)
    _style_axis(ax, xlim, zero=True)
    _set_ylim(
        ax,
        arrays,
        include_zero=True,
        lower_pad=0.12,
        upper_pad=0.18,
    )
    ax.set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )
    ax.set_ylabel(
        r"$p_{{\rm s},i}=z_i\,[\mathrm{p.u.}]$",
        fontsize=FONT_YLABEL,
        labelpad=5.0,
    )
    _status(ax, True)
    _dqemt_label(ax)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=FONT_LEGEND,
    )
    fig.subplots_adjust(
        left=0.112,
        right=0.988,
        bottom=0.18,
        top=0.79,
    )

    return save_figure(fig, "dq0_emt_local_safety_filter_corrections")


def plot_gfl_power_decomposition(events: list[float]) -> list[Path]:
    if not case_available("filter_on"):
        return []

    first = gfl_df("filter_on", GFL_BUSES[0])
    t = first["t_s"].to_numpy(float)

    dffr = np.zeros_like(t)
    zsum = np.zeros_like(t)
    applied = np.zeros_like(t)

    for bus in GFL_BUSES:
        df = gfl_df("filter_on", bus)
        ti = df["t_s"].to_numpy(float)

        if len(ti) != len(t) or not np.allclose(ti, t):
            ffr_i = np.interp(
                t,
                ti,
                df["delta_pF_ffr_system_pu"].to_numpy(float),
            )
            z_i = np.interp(
                t,
                ti,
                df["z_command_system_pu"].to_numpy(float),
            )
            applied_i = np.interp(
                t,
                ti,
                df["delta_pF_system_pu"].to_numpy(float),
            )
        else:
            ffr_i = df["delta_pF_ffr_system_pu"].to_numpy(float)
            z_i = df["z_command_system_pu"].to_numpy(float)
            applied_i = df["delta_pF_system_pu"].to_numpy(float)

        dffr += ffr_i
        zsum += z_i
        applied += applied_i

    xlim = (
        float(np.nanmin(t)),
        float(np.nanmax(t)),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH, 4.55),
        sharex=True,
    )

    l0, = axes[0].plot(
        t,
        dffr,
        color=COLOR_FFR,
        linestyle="--",
        linewidth=1.40,
    )
    l1, = axes[0].plot(
        t,
        applied,
        color=COLOR_APPLIED,
        linewidth=1.70,
    )
    fill = axes[0].fill_between(
        t,
        dffr,
        applied,
        color=COLOR_FILL,
        alpha=0.30,
        linewidth=0.0,
    )
    axes[0].legend(
        [l0, l1, fill],
        [
            r"$\Delta P_F^{\rm FFR}$",
            r"$\Delta P_F$",
            r"$\sum_i z_i$",
        ],
        loc="upper right",
        fontsize=FONT_LEGEND,
    )

    l2, = axes[1].plot(
        t,
        dffr,
        color=COLOR_FFR,
        linewidth=1.55,
    )
    l3, = axes[1].plot(
        t,
        zsum,
        color=COLOR_FILTER,
        linewidth=1.55,
    )
    axes[1].legend(
        [l2, l3],
        [
            r"$\Delta P_F^{\rm FFR}$",
            r"$\sum_i z_i$",
        ],
        loc="upper right",
        fontsize=FONT_LEGEND,
    )

    for ax in axes:
        _event_lines(ax, events)
        _style_axis(ax, xlim, zero=True)
        ax.set_ylabel(
            r"$\Delta P_F\,[\mathrm{p.u.}]$",
            fontsize=FONT_YLABEL,
            labelpad=5.0,
        )

    _set_ylim(
        axes[0],
        [dffr, applied],
        include_zero=True,
        upper_pad=0.20,
    )
    _set_ylim(
        axes[1],
        [dffr, zsum],
        include_zero=True,
        upper_pad=0.20,
    )

    axes[1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )

    _dqemt_label(axes[0])

    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.105,
        top=0.985,
        hspace=0.075,
    )

    return save_figure(fig, "dq0_emt_aggregate_gfl_power_decomposition")



# =============================================================================
# EMT epsilon-vs-time safety-violation heatmap
# =============================================================================

def epsilon_heatmap_available() -> bool:
    return EPSILON_HEATMAP_CSV.is_file()


def plot_epsilon_violation_heatmap(events: list[float]) -> list[Path]:
    """
    Plot the time-resolved dq0-EMT safety violation over the epsilon sweep.

    The Julia sweep exports

        v(t, epsilon) = max(0, -min_i h_i(t; epsilon)),

    so white corresponds to the safe region and the unsafe region uses the
    same pink/red visual language as the other safety-filter figures.
    """
    if not epsilon_heatmap_available():
        print(
            "Skipping epsilon heatmap because the Julia sweep CSV was not found:\n"
            f"  {EPSILON_HEATMAP_CSV}"
        )
        return []

    df = read_csv(EPSILON_HEATMAP_CSV)
    required = {
        "epsilon_s",
        "t_s",
        "worst_gfl_barrier_hz",
        "safety_violation_hz",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns in {EPSILON_HEATMAP_CSV.name}: {missing}"
        )

    df = df.copy()
    df = df[np.isfinite(df["epsilon_s"]) & np.isfinite(df["t_s"])]
    df = df[df["t_s"] <= EPSILON_HEATMAP_TMAX_S + 1e-12]

    if df.empty:
        raise ValueError("The EMT epsilon heatmap CSV contains no usable rows.")

    eps = np.sort(df["epsilon_s"].unique().astype(float))
    times = np.sort(df["t_s"].unique().astype(float))

    violation_pivot = df.pivot_table(
        index="epsilon_s",
        columns="t_s",
        values="safety_violation_hz",
        aggfunc="first",
    ).reindex(index=eps, columns=times)

    barrier_pivot = df.pivot_table(
        index="epsilon_s",
        columns="t_s",
        values="worst_gfl_barrier_hz",
        aggfunc="first",
    ).reindex(index=eps, columns=times)

    violation = violation_pivot.to_numpy(dtype=float)
    barrier = barrier_pivot.to_numpy(dtype=float)

    if np.isnan(violation).any() or np.isnan(barrier).any():
        raise ValueError(
            "The epsilon heatmap grid is incomplete. Each epsilon case must "
            "contain the same exported time grid."
        )

    tol = float(EPSILON_HEATMAP_VIOLATION_TOL_HZ)
    violation_masked = np.ma.masked_less_equal(violation, tol)

    # Same unsafe color family as the other paper figures.
    unsafe_cmap = LinearSegmentedColormap.from_list(
        "paper_unsafe_red",
        [
            COLOR_UNSAFE_FILL,
            "#E8AEB4",
            COLOR_LIMIT,
            "#8F2E3D",
        ],
        N=256,
    )
    unsafe_cmap.set_bad("white")

    positive = violation[violation > tol]
    vmax = float(np.max(positive)) if positive.size else 1e-4

    Tm, Em = np.meshgrid(times, eps)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(FIGURE_WIDTH, 3.35),
    )

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

    # The h=0 contour is the physically meaningful safe/unsafe boundary.
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

    for t_event in events:
        if 0.0 <= float(t_event) <= EPSILON_HEATMAP_TMAX_S:
            ax.axvline(
                float(t_event),
                color=COLOR_EVENT,
                linestyle=(0, (1.0, 1.7)),
                linewidth=0.90,
                alpha=0.90,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_xlim(0.0, EPSILON_HEATMAP_TMAX_S)
    ax.set_ylim(float(np.min(eps)), float(np.max(eps)))

    ax.set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.0,
    )
    ax.set_ylabel(
        r"$\epsilon\,[\mathrm{s}]$",
        fontsize=FONT_YLABEL,
        labelpad=5.0,
    )

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

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
    ax.grid(
        axis="x",
        color=COLOR_GRID,
        linewidth=0.42,
        alpha=0.42,
        zorder=0,
    )

    # Directly label the unshaded region rather than using "white = safe".
    ax.text(
        0.025,
        0.065,
        "Safe region",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=COLOR_INK,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
            "pad": 0.8,
        },
        zorder=6,
    )

    # Put a compact label next to the h=0 demarcation curve.
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
    cbar.set_label(
        r"Safety violation [Hz]",
        fontsize=9.5,
        labelpad=3.0,
    )
    cbar.ax.tick_params(
        direction="in",
        labelsize=8.5,
        length=3.0,
        width=0.7,
        pad=1.8,
    )
    cbar.outline.set_linewidth(0.72)
    cbar.outline.set_edgecolor(COLOR_AXIS)

    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.170,
        top=0.835,
    )

    return save_figure(
        fig,
        "dq0_emt_epsilon_time_safety_violation_heatmap",
    )


# =============================================================================
# Report and main
# =============================================================================

def write_report(generated: list[Path]) -> Path:
    ensure_dir(FIGURE_EXPORT_DIR)
    path = FIGURE_EXPORT_DIR / "plot_report.txt"
    lines = [
        "Julia dq0-EMT publication plots",
        f"Safety CSV root: {SAFETY_CSV_ROOT}",
        f"Filter OFF available: {case_available('filter_off')}",
        f"Filter ON available: {case_available('filter_on')}",
        f"Epsilon heatmap CSV available: {epsilon_heatmap_available()}",
        "",
        "Generated:",
    ]
    lines += [f"  {p.name}" for p in generated] if generated else ["  none"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_inputs() -> None:
    if not SAFETY_CSV_ROOT.is_dir():
        raise FileNotFoundError(
            "Julia safety CSV root was not found:\n"
            f"  {SAFETY_CSV_ROOT}"
        )

    standard_requested = any([
        MAKE_FREQUENCY_SAFETY_FIGURE,
        MAKE_BARRIER_FIGURE,
        MAKE_FILTER_STATE_FIGURE,
        MAKE_GFL_POWER_DECOMPOSITION_FIGURE,
    ])

    if not standard_requested:
        if MAKE_EPSILON_VIOLATION_HEATMAP and epsilon_heatmap_available():
            return
    elif case_available("filter_off") or case_available("filter_on"):
        return

    off_missing = missing_case_files("filter_off")
    on_missing = missing_case_files("filter_on")

    details = [
        "Julia safety-filter CSVs are incomplete.",
        "",
        "Expected safety CSV root:",
        f"  {SAFETY_CSV_ROOT}",
        "",
        "Missing FILTER-OFF GFL files:",
    ]
    details += [f"  {p}" for p in off_missing] or ["  none"]
    details += ["", "Missing FILTER-ON GFL files:"]
    details += [f"  {p}" for p in on_missing] or ["  none"]

    raise FileNotFoundError("\n".join(details))


def main() -> None:
    configure_matplotlib()
    _validate_inputs()
    ensure_dir(FIGURE_EXPORT_DIR)

    events = event_times()
    generated: list[Path] = []

    if MAKE_FREQUENCY_SAFETY_FIGURE:
        generated += plot_frequency_safety(events)

    if MAKE_BARRIER_FIGURE:
        generated += plot_barrier_safety(events)

    if MAKE_FILTER_STATE_FIGURE:
        generated += plot_filter_state(events)

    if MAKE_GFL_POWER_DECOMPOSITION_FIGURE:
        generated += plot_gfl_power_decomposition(events)

    if MAKE_EPSILON_VIOLATION_HEATMAP:
        generated += plot_epsilon_violation_heatmap(events)

    report = write_report(generated)

    print(f"\nGenerated {len(generated)} figure files in:")
    print(f"  {FIGURE_EXPORT_DIR}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

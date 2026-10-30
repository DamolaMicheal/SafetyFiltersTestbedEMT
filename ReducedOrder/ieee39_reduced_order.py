#!/usr/bin/env python3
"""IEEE 39-bus reduced-order simulation with a dynamic local safety filter."""

from __future__ import annotations

import copy
import csv
import importlib
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.transforms import blended_transform_factory


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Simulation settings
# =============================================================================

SYS_BASE_MVA = 100.0
F_BASE_HZ = 60.0
OMEGA_BASE = 2.0 * np.pi * F_BASE_HZ

DT = 0.002
T_FINAL = 6.0
T_GRID = np.arange(0.0, T_FINAL + 0.5 * DT, DT)

SLACK_BUS_EXT = 39

# Overlapping analytical/EMT network settings. The full-order EMT model uses
# differential transmission-line states and requires the validated small
# resistance floor for a stable fast electrical subsystem. Apply the same
# branch-R rule here so the reduced network coupling uses the same effective
# line data. The analytical model still treats background load deviations as
# power disturbances because it has no voltage-magnitude states.
APPLY_LINE_RESISTANCE_FLOOR = True
LINE_R_MIN_PU = 0.002
LINE_XR_CAP = 6.0
BACKGROUND_LOAD_CONSTANT_POWER = True

# Every online generator bus must be assigned once; bus 39 remains SG.
SG_BUSES_EXT = [32, 33, 35, 37, 39]
GFL_BUSES_EXT = [30, 34, 36, 31, 38]

# Reduced-model frequency convention:
# GFL frequency is obtained from the quasi-static frequency divider H_G.
# The full-order Julia EMT model instead uses a Kaura PLL with
# GFL_PLL_KP = 0.02 and GFL_PLL_KI = 2.5. Those PLL gains are deliberately
# not inserted here because this reduced model has no PLL states.

# Master supervision switch.
ENABLE_SAFETY_FILTER =True

# When enabled, also run the nominal case for comparison.
RUN_NOMINAL_COMPARISON =True

# Positive load steps increase demand.
LOAD_STEP_EVENTS = [
    {"time_s": 0.25, "bus": 22, "mw":100.0},
   # {"time_s": 0.25, "bus": 2, "mw": 150.0},
    #{"time_s": 0.25, "bus": 29, "mw": 150.0},
    #{"time_s": 0.25, "bus": 22, "mw": 100.0},
]

SG_REFERENCE_EVENTS = []
GFL_REFERENCE_EVENTS = []

# Delta p_F^FFR = -K_F omega_F.
K_FFR_DEFAULT = 0.0
# Device-base frequency--watt gain used at every controlled GFL bus.
# The reduced model converts this gain to the 100 MVA system base using
# each inverter MVA base, matching the EMT outer-control implementation.
K_FFR_MAP_EXT = {
    30: 0.0,
    33: 0.0,
    35: 0.0,
    31: 0.0,
    38: 0.0,
}

FREQ_LOWER_HZ = -0.08
FREQ_UPPER_HZ = +0.08
ALPHA_CBF = 2.0

FILTER_EPSILON_S = 0.035
ROCOF_MODE = "dirty"       # "dirty" or "exact"
ROCOF_TAU_S = 0.02

# Bus-dependent proximal gains only:
#   D_Gamma = diag(Gamma_ii),
#   rho = PROX_STEP_FRACTION / lambda_max(D_Gamma^(-1/2) Gamma D_Gamma^(-1/2)),
#   k_i = rho / Gamma_ii.
#
# Thus each bus is scaled by its local self-sensitivity Gamma_ii, while the
# full normalized Gamma sets the single common contraction fraction rho.
USE_LOCAL_GAMMA_SENSITIVITY_FOR_K = True
PROX_STEP_FRACTION = 0.8
PROX_K_MAP_EXT = {}

# The Julia EMT implementation now identifies its own Gamma directly from the
# EMT plant. Therefore the reduced model does not need to export analytical
# Gamma/H_G/matched-parameter CSV files for Julia at runtime.
EXPORT_SHARED_CBF_DESIGN = False
SHARED_DESIGN_DIR = os.path.join(SCRIPT_DIR, "shared_design")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_dynamic_local_filter")
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
REPORT_FILE = os.path.join(RESULTS_DIR, "simulation_report.txt")
EXPORT_PF_CSV = True
SHOW_PLOTS = True
USE_EXTERNAL_LATEX = False

# Plot bus selection. None means all valid buses for that figure.
PLOT_FREQUENCY_BUSES_EXT = [32, 34, 36, 37, 39]
PLOT_CBF_BUSES_EXT = None
PLOT_BARRIER_BUSES_EXT = None
PLOT_BARRIER_OVERLAY_BUSES_EXT = None
PLOT_FILTER_BUSES_EXT = None
PLOT_FILTER_OVERLAY_BUSES_EXT = None
PLOT_GFL_POWER_BUSES_EXT = None
PLOT_SG_POWER_BUSES_EXT = None
PLOT_POWER_OVERLAY_BUSES_EXT = [30, 33, 35, 31, 38]

# Additional mixed-resource overlays. None means all configured GFL and SG buses.
# GFL traces are solid; SG traces are dashed.
PLOT_RESOURCE_FREQUENCY_BUSES_EXT = None
PLOT_RESOURCE_POWER_BUSES_EXT = None

# One bus panel per row for all multi-bus subplot figures.
BUS_SUBPLOT_COLUMNS = 1

# Use one common vertical scale for all z_i panels and the z_i overlay.
FILTER_STATE_COMMON_YLIM = True
PRINT_FILTER_STATE_DIAGNOSTICS = True

MAKE_FREQUENCY_FIGURE = True
MAKE_CBF_CONSTRAINT_FIGURE = True
MAKE_BARRIER_FIGURE = True
MAKE_BARRIER_OVERLAY_FIGURE = True
MAKE_FILTER_STATE_FIGURE = True
MAKE_FILTER_STATE_OVERLAY_FIGURE = True
MAKE_FILTER_TRACKING_FIGURE = False
MAKE_GFL_POWER_FIGURE = True
MAKE_GFL_AGGREGATE_FIGURE = True
MAKE_SG_POWER_FIGURE = True
MAKE_POWER_OVERLAY_FIGURE = True

# Additional publication overlays containing both GFL and SG resources.
MAKE_RESOURCE_FREQUENCY_OVERLAY_FIGURE = True
MAKE_RESOURCE_POWER_OVERLAY_FIGURE = True

ACTIVE_SAFETY_BOUND = "auto"   # "auto", "lower", or "upper"
SHOW_GFL_REFERENCE_DEVIATION = True

# =============================================================================
# MATPOWER column indices
# =============================================================================

BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
BUS_AREA = 6
VM = 7
VA = 8
BASE_KV = 9
ZONE = 10
VMAX = 11
VMIN = 12

GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
VG = 5
MBASE_COL = 6
GEN_STATUS = 7
PMAX = 8
PMIN = 9

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
RATE_B = 6
RATE_C = 7
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12


@dataclass
class ReducedModel:
    A: np.ndarray
    B: np.ndarray
    Gs: np.ndarray
    Cw: np.ndarray
    Gamma: np.ndarray
    HG: np.ndarray
    Kf: np.ndarray
    Yx: np.ndarray
    Yu: np.ndarray
    Yz: np.ndarray
    I0: np.ndarray
    SbarG: np.ndarray
    SbarF: np.ndarray
    SbarL: np.ndarray
    SG_int: list[int]
    SG_nr_int: list[int]
    GFL_int: list[int]
    L_int: list[int]
    SG_ext: list[int]
    GFL_ext: list[int]
    nG: int
    nF: int
    nL: int
    nx: int
    nu: int
    M_diag: np.ndarray
    D_diag: np.ndarray
    base_mva: float
    prox_k: np.ndarray
    lower_pu: np.ndarray
    upper_pu: np.ndarray
    alpha_low: np.ndarray
    alpha_up: np.ndarray


# =============================================================================
# Utilities and power-flow initialization
# =============================================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def deepcopy_case(mpc: dict) -> dict:
    out = {}
    for key, value in mpc.items():
        out[key] = np.array(value, dtype=float, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return out


def sanitize_case_for_runpf(mpc: dict):
    mpc_s = deepcopy_case(mpc)
    bus = np.asarray(mpc_s["bus"], dtype=float)
    gen = np.asarray(mpc_s["gen"], dtype=float)
    branch = np.asarray(mpc_s["branch"], dtype=float)

    bus_ids = bus[:, BUS_I].astype(int)
    if np.unique(bus_ids).size != bus.shape[0]:
        raise ValueError("Bus identifiers are not unique.")

    order = np.argsort(bus_ids)
    bus = bus[order, :]
    bus_ids_sorted = bus[:, BUS_I].astype(int)
    id_map = {int(old): int(new) for new, old in enumerate(bus_ids_sorted, start=1)}

    bus[:, BUS_I] = np.array([id_map[int(x)] for x in bus_ids_sorted], dtype=float)
    gen[:, GEN_BUS] = np.array([id_map[int(x)] for x in gen[:, GEN_BUS].astype(int)], dtype=float)
    branch[:, F_BUS] = np.array([id_map[int(x)] for x in branch[:, F_BUS].astype(int)], dtype=float)
    branch[:, T_BUS] = np.array([id_map[int(x)] for x in branch[:, T_BUS].astype(int)], dtype=float)

    mpc_s["bus"] = bus
    mpc_s["gen"] = gen
    mpc_s["branch"] = branch
    return mpc_s, id_map


def load_case_from_py() -> dict:
    """Prefer the user's case39_modified_slack39.py; fall back to PYPOWER case39."""
    try:
        mod = importlib.import_module("case39_modified_slack39")
        if hasattr(mod, "get_mpc"):
            return mod.get_mpc()
        if hasattr(mod, "load_case"):
            return mod.load_case()
        if hasattr(mod, "mpc"):
            return mod.mpc
        raise RuntimeError(
            "case39_modified_slack39.py must export get_mpc(), load_case(), or mpc."
        )
    except ModuleNotFoundError:
        try:
            from pypower.api import case39
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PYPOWER is required. Install it with 'pip install PYPOWER', and place "
                "case39_modified_slack39.py beside this script if the modified case is required."
            ) from exc
        warnings.warn(
            "case39_modified_slack39.py was not found; using PYPOWER's standard case39 instead."
        )
        return case39()


def build_ybus_dense_matpower(base_mva: float, bus: np.ndarray, branch: np.ndarray) -> np.ndarray:
    nb = bus.shape[0]
    busnums = bus[:, BUS_I].astype(int)
    bus_to_idx = {int(busnum): i for i, busnum in enumerate(busnums)}

    Y = np.zeros((nb, nb), dtype=complex)
    gs = bus[:, GS] / base_mva
    bs = bus[:, BS] / base_mva
    Y[np.arange(nb), np.arange(nb)] += gs + 1j * bs

    for row in branch:
        if int(row[BR_STATUS]) == 0:
            continue

        fb = int(row[F_BUS])
        tb = int(row[T_BUS])
        i = bus_to_idx[fb]
        j = bus_to_idx[tb]

        z = float(row[BR_R]) + 1j * float(row[BR_X])
        if abs(z) < 1e-14:
            raise ValueError(f"Zero branch impedance encountered on {fb}-{tb}.")

        y = 1.0 / z
        ysh = 1j * float(row[BR_B]) / 2.0
        tap = float(row[TAP]) if abs(float(row[TAP])) > 1e-14 else 1.0
        shift = np.deg2rad(float(row[SHIFT]))
        tap_complex = tap * np.exp(1j * shift)

        Y[i, i] += (y + ysh) / (tap_complex * np.conj(tap_complex))
        Y[i, j] += -y / np.conj(tap_complex)
        Y[j, i] += -y / tap_complex
        Y[j, j] += y + ysh

    return Y


def network_laplacian_from_ybus(Ybus: np.ndarray) -> np.ndarray:
    """DC-network Laplacian using series susceptances; shunt terms are removed."""
    B = -np.imag(Ybus).astype(float)
    B = 0.5 * (B + B.T)
    np.fill_diagonal(B, 0.0)
    np.fill_diagonal(B, -np.sum(B, axis=1))
    return B


def apply_line_resistance_floor_to_case(mpc: dict) -> dict:
    """Apply the same optional branch-R rule used by the EMT script."""
    out = deepcopy_case(mpc)
    if not APPLY_LINE_RESISTANCE_FLOOR:
        return out
    branch = np.asarray(out["branch"], dtype=float)
    for k, row in enumerate(branch):
        if int(row[BR_STATUS]) == 0:
            continue
        # Match the Julia helper, which modifies only Line components.  In the
        # MATPOWER table the transformer branches carry a nonzero tap ratio.
        if abs(float(row[TAP])) > 1e-14:
            continue
        r = float(row[BR_R])
        x = abs(float(row[BR_X]))
        branch[k, BR_R] = max(r, LINE_R_MIN_PU, x / LINE_XR_CAP)
    out["branch"] = branch
    return out


def run_pf_init(mpc_ext: dict):
    mpc_ext = apply_line_resistance_floor_to_case(mpc_ext)
    try:
        from pypower.api import ppoption, runpf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PYPOWER is required for the AC power-flow initialization.") from exc

    mpc_base, id_map = sanitize_case_for_runpf(mpc_ext)
    inv_map = {v: k for k, v in id_map.items()}
    base_mva = float(mpc_base.get("baseMVA", SYS_BASE_MVA))

    if SLACK_BUS_EXT not in id_map:
        raise ValueError(f"Slack bus {SLACK_BUS_EXT} is not present in the case.")

    slack_bus_int = id_map[SLACK_BUS_EXT]
    mpc_pf = deepcopy_case(mpc_base)

    ppopt = ppoption(OUT_ALL=0, VERBOSE=0, PF_ALG=1, PF_MAX_IT=80, PF_TOL=1e-10)
    t0 = time.perf_counter()
    pf_results, success = runpf(mpc_pf, ppopt)
    elapsed = time.perf_counter() - t0
    if not success:
        raise RuntimeError("The AC power flow did not converge.")

    Ybus = build_ybus_dense_matpower(
        base_mva,
        np.asarray(mpc_base["bus"], dtype=float),
        np.asarray(mpc_base["branch"], dtype=float),
    )

    print(
        f"AC power flow: converged in {elapsed:.3f} s | "
        f"baseMVA={base_mva:.1f} | slack bus={SLACK_BUS_EXT}"
    )

    return mpc_base, pf_results, Ybus, id_map, inv_map, slack_bus_int, base_mva


def generator_mbase_by_bus(gen: np.ndarray, N: int) -> np.ndarray:
    out = np.full(N, SYS_BASE_MVA, dtype=float)
    seen = np.zeros(N, dtype=bool)
    for row in gen:
        if int(row[GEN_STATUS]) == 0:
            continue
        bus = int(row[GEN_BUS])
        mbase = float(row[MBASE_COL]) if float(row[MBASE_COL]) > 0.0 else SYS_BASE_MVA
        if not seen[bus - 1]:
            out[bus - 1] = mbase
            seen[bus - 1] = True
        else:
            out[bus - 1] += mbase
    return out


def export_power_flow_csvs(pf_results: dict, inv_map: dict[int, int]) -> None:
    ensure_dir(CSV_DIR)

    bus = np.asarray(pf_results["bus"], dtype=float).copy()
    gen = np.asarray(pf_results["gen"], dtype=float).copy()
    branch = np.asarray(pf_results["branch"], dtype=float).copy()

    bus[:, BUS_I] = [inv_map.get(int(x), int(x)) for x in bus[:, BUS_I]]
    gen[:, GEN_BUS] = [inv_map.get(int(x), int(x)) for x in gen[:, GEN_BUS]]
    branch[:, F_BUS] = [inv_map.get(int(x), int(x)) for x in branch[:, F_BUS]]
    branch[:, T_BUS] = [inv_map.get(int(x), int(x)) for x in branch[:, T_BUS]]

    np.savetxt(
        os.path.join(CSV_DIR, "power_flow_bus.csv"),
        bus,
        delimiter=",",
        header="BUS_I,BUS_TYPE,PD,QD,GS,BS,BUS_AREA,VM,VA,BASE_KV,ZONE,VMAX,VMIN",
        comments="",
    )
    np.savetxt(
        os.path.join(CSV_DIR, "power_flow_gen.csv"),
        gen,
        delimiter=",",
        header="GEN_BUS,PG,QG,QMAX,QMIN,VG,MBASE,GEN_STATUS,PMAX,PMIN",
        comments="",
    )
    np.savetxt(
        os.path.join(CSV_DIR, "power_flow_branch.csv"),
        branch,
        delimiter=",",
        header="F_BUS,T_BUS,BR_R,BR_X,BR_B,RATE_A,RATE_B,RATE_C,TAP,SHIFT,BR_STATUS,ANGMIN,ANGMAX",
        comments="",
    )


# =============================================================================
# Reduced DAE, frequency divider, and ODE elimination
# =============================================================================


def selector(N: int, buses_int: list[int]) -> np.ndarray:
    S = np.zeros((N, len(buses_int)), dtype=float)
    for j, bus in enumerate(buses_int):
        S[bus - 1, j] = 1.0
    return S


def reference_angle_embedding(N: int, slack_int: int) -> tuple[np.ndarray, list[int]]:
    non_ref = [bus for bus in range(1, N + 1) if bus != slack_int]
    I0 = np.zeros((N, N - 1), dtype=float)
    for col, bus in enumerate(non_ref):
        I0[bus - 1, col] = 1.0
    return I0, non_ref


def reduced_selector(non_ref: list[int], buses_int: list[int]) -> np.ndarray:
    pos = {bus: i for i, bus in enumerate(non_ref)}
    Sbar = np.zeros((len(non_ref), len(buses_int)), dtype=float)
    for col, bus in enumerate(buses_int):
        if bus not in pos:
            raise ValueError(f"Reference bus {bus} cannot appear in a reduced-angle selector.")
        Sbar[pos[bus], col] = 1.0
    return Sbar


def validate_resource_partition(
    mpc_base: dict,
    id_map: dict[int, int],
    inv_map: dict[int, int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    sg_ext = list(dict.fromkeys(int(b) for b in SG_BUSES_EXT))
    gfl_ext = list(dict.fromkeys(int(b) for b in GFL_BUSES_EXT))

    if set(sg_ext) & set(gfl_ext):
        raise ValueError(f"SG/GFL overlap: {sorted(set(sg_ext) & set(gfl_ext))}")
    if SLACK_BUS_EXT not in sg_ext:
        raise ValueError(f"Slack bus {SLACK_BUS_EXT} must remain in SG_BUSES_EXT.")
    if len(sg_ext) < 2:
        raise ValueError("At least two SG buses, including the reference SG, are required.")
    if len(gfl_ext) < 1:
        raise ValueError("At least one GFL bus is required for the safety-filter simulation.")

    gen = np.asarray(mpc_base["gen"], dtype=float)
    online_resource_ext = sorted(
        {
            int(inv_map[int(row[GEN_BUS])])
            for row in gen
            if int(row[GEN_STATUS]) != 0
        }
    )

    assigned = set(sg_ext) | set(gfl_ext)
    expected = set(online_resource_ext)
    if assigned != expected:
        raise ValueError(
            "SG_BUSES_EXT and GFL_BUSES_EXT must partition all online generator buses. "
            f"Missing={sorted(expected - assigned)}, extra={sorted(assigned - expected)}"
        )

    missing_from_case = [b for b in assigned if b not in id_map]
    if missing_from_case:
        raise ValueError(f"Configured resource buses are absent from the case: {missing_from_case}")

    # Keep the configured resource order; place the reference SG last.
    sg_ext_ordered = [b for b in sg_ext if b != SLACK_BUS_EXT] + [SLACK_BUS_EXT]
    gfl_ext_ordered = list(gfl_ext)

    sg_int = [id_map[b] for b in sg_ext_ordered]
    gfl_int = [id_map[b] for b in gfl_ext_ordered]
    return sg_ext_ordered, gfl_ext_ordered, sg_int, gfl_int

def build_frequency_divider(
    Bnet: np.ndarray,
    I0: np.ndarray,
    SG_int: list[int],
    GFL_int: list[int],
    L_int: list[int],
    non_ref: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nG = len(SG_int)
    nF = len(GFL_int)
    SG_nr = SG_int[:-1]

    SFL = np.hstack([selector(Bnet.shape[0], GFL_int), selector(Bnet.shape[0], L_int)])
    SbarG = reduced_selector(non_ref, SG_nr)
    SbarF = reduced_selector(non_ref, GFL_int)
    SbarL = reduced_selector(non_ref, L_int) if L_int else np.zeros((len(non_ref), 0))
    SbarFL = np.hstack([SbarF, SbarL])

    Bbar = Bnet @ I0
    A_FL = SFL.T @ Bbar @ SbarFL
    B_FL_G = SFL.T @ Bbar @ SbarG

    E_G = np.hstack([np.eye(nG - 1), -np.ones((nG - 1, 1))])
    e_ref = np.zeros((nG, 1))
    e_ref[-1, 0] = 1.0

    X = la.solve(A_FL, B_FL_G, assume_a="gen")
    H_FL = np.ones((nF + len(L_int), 1)) @ e_ref.T - X @ E_G
    C_F = np.hstack([np.eye(nF), np.zeros((nF, len(L_int)))])
    H_G = C_F @ H_FL

    return H_G, SbarG, SbarF, SbarL


def build_reduced_model(
    mpc_base: dict,
    Ybus: np.ndarray,
    id_map: dict[int, int],
    inv_map: dict[int, int],
    slack_int: int,
    base_mva: float,
) -> ReducedModel:
    N = np.asarray(mpc_base["bus"]).shape[0]
    SG_ext, GFL_ext, SG_int, GFL_int = validate_resource_partition(mpc_base, id_map, inv_map)
    SG_nr_int = SG_int[:-1]
    L_int = sorted(set(range(1, N + 1)) - set(SG_int) - set(GFL_int))

    nG = len(SG_int)
    nF = len(GFL_int)
    nL = len(L_int)

    Bnet = network_laplacian_from_ybus(Ybus)
    I0, non_ref = reference_angle_embedding(N, slack_int)
    Bbar = Bnet @ I0

    SG_selector = selector(N, SG_int)
    F_selector = selector(N, GFL_int)
    L_selector = selector(N, L_int)

    H_G, SbarG, SbarF, SbarL = build_frequency_divider(
        Bnet, I0, SG_int, GFL_int, L_int, non_ref
    )

    E_G = np.hstack([np.eye(nG - 1), -np.ones((nG - 1, 1))])

    gen_case = np.asarray(mpc_base["gen"], dtype=float)
    mbase = generator_mbase_by_bus(gen_case, N)

    H_map = {
        30: 8.90,
        31: 5.44,
        32: 4.95,
        33: 5.67,
        34: 4.86,
        35: 4.86,
        36: 4.33,
        37: 4.08,
        38: 3.90,
        39: 5.00,
    }
    D_map = {
        30: 40.0, 31: 40.0, 32: 40.0, 33: 40.0, 34: 40.0,
        35: 40.0, 36: 40.0, 37: 40.0, 38: 40.0, 39: 48.0,
    }

    M_diag = []
    D_diag = []
    invR_diag = []
    tau_diag = []
    
    for bus_int in SG_int:
        bus_ext = int(inv_map[bus_int])
    
        scale = float(mbase[bus_int - 1]) / SYS_BASE_MVA
    
        M_diag.append(
            2.0 * H_map.get(bus_ext, 4.0) * scale
        )
    
        D_diag.append(
            D_map.get(bus_ext, 40.0) * scale
        )
    
        invR_diag.append(
            (1.0 / 0.05) * scale
        )
    
        tau_diag.append(0.50)

    M = np.diag(M_diag)
    D = np.diag(D_diag)
    Rinv = np.diag(invR_diag)
    Tm = np.diag(tau_diag)

    Kf_values = []
    for bus_int in GFL_int:
        bus_ext = int(inv_map[bus_int])
        scale = float(mbase[bus_int - 1]) / SYS_BASE_MVA
        Kf_values.append(K_FFR_MAP_EXT.get(bus_ext, K_FFR_DEFAULT) * scale)
    Kf = np.diag(Kf_values)

    Minv = np.diag(1.0 / np.diag(M))
    Tinv = np.diag(1.0 / np.diag(Tm))

    ntheta = nG - 1
    nx = ntheta + nG + nG
    nu = nG + nF + N
    ny = nF + nL + nF

    Z_tg_tg = np.zeros((ntheta, ntheta))
    Z_tg_pm = np.zeros((ntheta, nG))
    Z_g_tg = np.zeros((nG, ntheta))
    Z_g_g = np.zeros((nG, nG))

    A11 = np.block(
        [
            [Z_tg_tg, OMEGA_BASE * E_G, Z_tg_pm],
            [
                -Minv @ SG_selector.T @ Bbar @ SbarG,
                -Minv @ D,
                Minv,
            ],
            [Z_g_tg, -Tinv @ Rinv, -Tinv],
        ]
    )

    A12 = np.block(
        [
            [
                np.zeros((ntheta, nF)),
                np.zeros((ntheta, nL)),
                np.zeros((ntheta, nF)),
            ],
            [
                -Minv @ SG_selector.T @ Bbar @ SbarF,
                -Minv @ SG_selector.T @ Bbar @ SbarL,
                np.zeros((nG, nF)),
            ],
            [
                np.zeros((nG, nF)),
                np.zeros((nG, nL)),
                np.zeros((nG, nF)),
            ],
        ]
    )

    A21 = np.block(
        [
            [
                -F_selector.T @ Bbar @ SbarG,
                np.zeros((nF, nG)),
                np.zeros((nF, nG)),
            ],
            [
                -L_selector.T @ Bbar @ SbarG,
                np.zeros((nL, nG)),
                np.zeros((nL, nG)),
            ],
            [
                np.zeros((nF, ntheta)),
                -H_G,
                np.zeros((nF, nG)),
            ],
        ]
    )

    A22 = np.block(
        [
            [
                -F_selector.T @ Bbar @ SbarF,
                -F_selector.T @ Bbar @ SbarL,
                -Kf,
            ],
            [
                -L_selector.T @ Bbar @ SbarF,
                -L_selector.T @ Bbar @ SbarL,
                np.zeros((nL, nF)),
            ],
            [
                np.zeros((nF, nF)),
                np.zeros((nF, nL)),
                np.eye(nF),
            ],
        ]
    )

    B1 = np.block(
        [
            [
                np.zeros((ntheta, nG)),
                np.zeros((ntheta, nF)),
                np.zeros((ntheta, N)),
            ],
            [
                np.zeros((nG, nG)),
                np.zeros((nG, nF)),
                -Minv @ SG_selector.T,
            ],
            [
                Tinv,
                np.zeros((nG, nF)),
                np.zeros((nG, N)),
            ],
        ]
    )

    B2 = np.block(
        [
            [
                np.zeros((nF, nG)),
                np.eye(nF),
                -F_selector.T,
            ],
            [
                np.zeros((nL, nG)),
                np.zeros((nL, nF)),
                -L_selector.T,
            ],
            [
                np.zeros((nF, nG)),
                np.zeros((nF, nF)),
                np.zeros((nF, N)),
            ],
        ]
    )

    E2 = np.vstack(
        [
            np.eye(nF),
            np.zeros((nL, nF)),
            np.zeros((nF, nF)),
        ]
    )

    if A22.shape != (ny, ny):
        raise RuntimeError("Internal DAE assembly error: A22 has the wrong dimension.")

    cond_A22 = np.linalg.cond(A22)
    if not np.isfinite(cond_A22) or cond_A22 > 1e12:
        raise np.linalg.LinAlgError(
            f"The algebraic block A22 is ill-conditioned (cond={cond_A22:.3e})."
        )

    A22_A21 = la.solve(A22, A21, assume_a="gen")
    A22_B2 = la.solve(A22, B2, assume_a="gen")
    A22_E2 = la.solve(A22, E2, assume_a="gen")

    A = A11 - A12 @ A22_A21
    B = B1 - A12 @ A22_B2
    Gs = -A12 @ A22_E2

    Yx = -A22_A21
    Yu = -A22_B2
    Yz = -A22_E2

    Cw = np.hstack(
        [
            np.zeros((nF, ntheta)),
            H_G,
            np.zeros((nF, nG)),
        ]
    )
    Gamma = Cw @ Gs

    asym = la.norm(Gamma - Gamma.T, ord="fro") / max(la.norm(Gamma, ord="fro"), 1e-14)
    Gamma_sym = 0.5 * (Gamma + Gamma.T)
    eigvals = la.eigvalsh(Gamma_sym)
    singular_H = la.svdvals(H_G)

    gamma_scale = max(float(np.max(np.abs(eigvals))), 1.0)
    gamma_tol = max(1e-12, 1e-10 * gamma_scale)
    rank_H = int(np.sum(singular_H > max(1e-12, 1e-10 * singular_H[0])))
    rank_Gamma = int(np.sum(eigvals > gamma_tol))

    if asym > 1e-8:
        raise ValueError(
            "Gamma is not numerically symmetric for the selected partition. "
            f"Relative asymmetry={asym:.3e}."
        )
    if rank_Gamma < nF or eigvals[0] <= gamma_tol:
        raise ValueError(
            "The selected SG/GFL partition does not satisfy Gamma >> 0. "
            f"rank(H_G)={rank_H}/{nF}, rank(Gamma)={rank_Gamma}/{nF}, "
            f"lambda_min(Gamma)={eigvals[0]:.3e}. Choose another partition or run the diagnostic script."
        )

    gamma_lmax = float(eigvals[-1])
    gamma_diag = np.diag(Gamma_sym).copy()
    if np.any(gamma_diag <= gamma_tol):
        bad = [GFL_ext[i] for i in np.where(gamma_diag <= gamma_tol)[0]]
        raise ValueError(
            "Positive local RoCoF self-sensitivities Gamma_ii are required. "
            f"Invalid GFL buses: {bad}."
        )

    if not (0.0 < PROX_STEP_FRACTION < 2.0):
        raise ValueError("PROX_STEP_FRACTION must lie strictly between 0 and 2.")

    mode = "local_sensitivity" if USE_LOCAL_GAMMA_SENSITIVITY_FOR_K else "spectral"

    if mode == "spectral":
        prox_k = np.full(nF, PROX_STEP_FRACTION / gamma_lmax, dtype=float)
        local_scale = None
        normalized_lmax = None
    else:
        Dinvhalf = np.diag(1.0 / np.sqrt(gamma_diag))
        Gamma_normalized = Dinvhalf @ Gamma_sym @ Dinvhalf
        normalized_lmax = float(la.eigvalsh(Gamma_normalized)[-1])
        local_scale = PROX_STEP_FRACTION / normalized_lmax
        prox_k = local_scale / gamma_diag

    gain_source = ["auto"] * nF
    for i, bus_ext in enumerate(GFL_ext):
        if bus_ext in PROX_K_MAP_EXT:
            prox_k[i] = float(PROX_K_MAP_EXT[bus_ext])
            gain_source[i] = "manual"

    if np.any(prox_k <= 0.0):
        raise ValueError("All proximal gains k_i must be positive.")

    Khalf = np.diag(np.sqrt(prox_k))
    weighted_lmax = float(la.eigvalsh(Khalf @ Gamma_sym @ Khalf)[-1])
    if weighted_lmax >= 2.0:
        warnings.warn(
            "The proximal gains violate lambda_max(K^(1/2) Gamma K^(1/2)) < 2; "
            f"value={weighted_lmax:.4f}."
        )

    lower = np.full(nF, FREQ_LOWER_HZ / F_BASE_HZ, dtype=float)
    upper = np.full(nF, FREQ_UPPER_HZ / F_BASE_HZ, dtype=float)
    alpha_low = np.full(nF, ALPHA_CBF, dtype=float)
    alpha_up = np.full(nF, ALPHA_CBF, dtype=float)


    print("\nReduced analytical model")
    print(f"  SG buses : {SG_ext}")
    print(f"  GFL buses: {GFL_ext}")
    print(f"  states={nx}, algebraic variables={ny}, cond(A22)={cond_A22:.3e}")
    print(
        f"  H_G rank={rank_H}/{nF}, sigma_min(H_G)={singular_H[-1]:.3e}"
    )
    print(
        f"  Gamma rank={rank_Gamma}/{nF}, lambda_min={eigvals[0]:.3e}, "
        f"lambda_max={eigvals[-1]:.3e}, cond={eigvals[-1] / eigvals[0]:.3e}"
    )

    print("\nProximal gains")
    if mode == "spectral":
        print(
            "  mode=spectral: "
            "k_i = PROX_STEP_FRACTION / lambda_max(Gamma)"
        )
    else:
        print(
            "  mode=local_sensitivity: "
            "k_i = rho / Gamma_ii"
        )
        print(
            f"  lambda_max(D_Gamma^(-1/2) Gamma D_Gamma^(-1/2))="
            f"{normalized_lmax:.6f}"
        )
        print(
            f"  rho=PROX_STEP_FRACTION/lambda_max(normalized Gamma)="
            f"{local_scale:.6f}"
        )
    print(f"  PROX_STEP_FRACTION={PROX_STEP_FRACTION:.4f}")
    print(f"  lambda_max(K^(1/2) Gamma K^(1/2))={weighted_lmax:.6f}")
    print("  Bus | Gamma_ii [1/s] | k_i [s] | k_i Gamma_ii | source")
    print("  " + "-" * 64)
    for i, bus_ext in enumerate(GFL_ext):
        print(
            f"  {bus_ext:>3d} | {gamma_diag[i]:>14.7e} | "
            f"{prox_k[i]:>9.5f} | {prox_k[i] * gamma_diag[i]:>12.6f} | "
            f"{gain_source[i]}"
        )

    print(
        f"\nDynamic filter: epsilon={FILTER_EPSILON_S:.3f} s, "
        f"RoCoF={ROCOF_MODE}, tau_d={ROCOF_TAU_S:.3f} s"
    )

    return ReducedModel(
        A=A,
        B=B,
        Gs=Gs,
        Cw=Cw,
        Gamma=Gamma,
        HG=H_G,
        Kf=Kf,
        Yx=Yx,
        Yu=Yu,
        Yz=Yz,
        I0=I0,
        SbarG=SbarG,
        SbarF=SbarF,
        SbarL=SbarL,
        SG_int=SG_int,
        SG_nr_int=SG_nr_int,
        GFL_int=GFL_int,
        L_int=L_int,
        SG_ext=SG_ext,
        GFL_ext=GFL_ext,
        nG=nG,
        nF=nF,
        nL=nL,
        nx=nx,
        nu=nu,
        M_diag=np.asarray(M_diag, dtype=float),
        D_diag=np.asarray(D_diag, dtype=float),
        base_mva=base_mva,
        prox_k=prox_k,
        lower_pu=lower,
        upper_pu=upper,
        alpha_low=alpha_low,
        alpha_up=alpha_up,
    )


def export_shared_cbf_design(model: ReducedModel) -> None:
    """Export the analytical divider and Gamma used by the EMT validation."""
    if not EXPORT_SHARED_CBF_DESIGN:
        return

    ensure_dir(SHARED_DESIGN_DIR)

    gamma_path = os.path.join(SHARED_DESIGN_DIR, "Gamma_analytical.csv")
    gamma_header = ["bus"] + [f"bus_{b}" for b in model.GFL_ext]
    gamma_rows = np.column_stack([
        np.asarray(model.GFL_ext, dtype=float),
        np.asarray(model.Gamma, dtype=float),
    ])
    np.savetxt(
        gamma_path,
        gamma_rows,
        delimiter=",",
        header=",".join(gamma_header),
        comments="",
        fmt="%.12g",
    )

    hg_path = os.path.join(SHARED_DESIGN_DIR, "H_G_analytical.csv")
    hg_header = ["gfl_bus"] + [f"sg_{b}" for b in model.SG_ext]
    hg_rows = np.column_stack([
        np.asarray(model.GFL_ext, dtype=float),
        np.asarray(model.HG, dtype=float),
    ])
    np.savetxt(
        hg_path,
        hg_rows,
        delimiter=",",
        header=",".join(hg_header),
        comments="",
        fmt="%.12g",
    )

    params_path = os.path.join(SHARED_DESIGN_DIR, "matched_parameters.csv")
    rows = [
        ("system_base_mva", SYS_BASE_MVA),
        ("base_frequency_hz", F_BASE_HZ),
        ("simulation_final_time_s", T_FINAL),
        ("frequency_lower_hz", FREQ_LOWER_HZ),
        ("frequency_upper_hz", FREQ_UPPER_HZ),
        ("cbf_alpha", ALPHA_CBF),
        ("filter_epsilon_s", FILTER_EPSILON_S),
        ("rocof_tau_s", ROCOF_TAU_S),
        ("prox_step_fraction", PROX_STEP_FRACTION),
        ("total_load_step_mw", sum(float(e["mw"]) for e in LOAD_STEP_EVENTS)),
        ("line_resistance_floor_enabled", float(APPLY_LINE_RESISTANCE_FLOOR)),
        ("line_r_min_pu", LINE_R_MIN_PU),
        ("line_xr_cap", LINE_XR_CAP),
        ("background_load_constant_power", float(BACKGROUND_LOAD_CONSTANT_POWER)),
    ]
    with open(params_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["quantity", "value"])
        writer.writerows(rows)

    print(f"Shared analytical CBF design exported to: {SHARED_DESIGN_DIR}")


# =============================================================================
# Disturbances and local filter
# =============================================================================


def make_input_event_vector(
    model: ReducedModel,
    id_map: dict[int, int],
    kind: str,
    bus_ext: int,
    mw: float,
) -> np.ndarray:
    u = np.zeros(model.nu, dtype=float)
    bus_int = id_map[int(bus_ext)]
    value_pu = float(mw) / model.base_mva

    if kind == "load":
        offset = model.nG + model.nF
        u[offset + bus_int - 1] = value_pu
    elif kind == "sg_ref":
        if bus_ext not in model.SG_ext:
            raise ValueError(f"Bus {bus_ext} is not an SG bus.")
        u[model.SG_ext.index(bus_ext)] = value_pu
    elif kind == "gfl_ref":
        if bus_ext not in model.GFL_ext:
            raise ValueError(f"Bus {bus_ext} is not a GFL bus.")
        u[model.nG + model.GFL_ext.index(bus_ext)] = value_pu
    else:
        raise ValueError(f"Unknown event type '{kind}'.")

    return u


def build_events(model: ReducedModel, id_map: dict[int, int]):
    events = []

    for event in LOAD_STEP_EVENTS:
        events.append(
            {
                "time_s": float(event["time_s"]),
                "label": f"+{float(event['mw']):g} MW load at Bus {int(event['bus'])}",
                "u": make_input_event_vector(model, id_map, "load", int(event["bus"]), float(event["mw"])),
            }
        )

    for event in SG_REFERENCE_EVENTS:
        events.append(
            {
                "time_s": float(event["time_s"]),
                "label": f"{float(event['mw']):+g} MW SG reference at Bus {int(event['bus'])}",
                "u": make_input_event_vector(model, id_map, "sg_ref", int(event["bus"]), float(event["mw"])),
            }
        )

    for event in GFL_REFERENCE_EVENTS:
        events.append(
            {
                "time_s": float(event["time_s"]),
                "label": f"{float(event['mw']):+g} MW GFL reference at Bus {int(event['bus'])}",
                "u": make_input_event_vector(model, id_map, "gfl_ref", int(event["bus"]), float(event["mw"])),
            }
        )

    events.sort(key=lambda item: item["time_s"])
    return events


def input_at_time(t: float, model: ReducedModel, events: list[dict]) -> np.ndarray:
    u = np.zeros(model.nu, dtype=float)
    for event in events:
        if t + 1e-12 >= event["time_s"]:
            u += event["u"]
    return u


def project_zero_interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if np.any(b < a - 1e-12):
        raise RuntimeError("The local proximal interval is not ordered.")
    return np.where(a > 0.0, a, np.where(b < 0.0, b, 0.0))


def local_filter_target(
    omega_f: np.ndarray,
    rocof_hat: np.ndarray,
    z: np.ndarray,
    model: ReducedModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local law: no phi_i, Gamma row, or remote z_j is used here."""
    a = z + model.prox_k * (
        -model.alpha_low * (omega_f - model.lower_pu) - rocof_hat
    )
    b = z + model.prox_k * (
        +model.alpha_up * (model.upper_pu - omega_f) - rocof_hat
    )
    target = project_zero_interval(a, b)
    return target, a, b


# =============================================================================
# Time-domain simulation
# =============================================================================


def rk4_step(state: np.ndarray, u: np.ndarray, h: float, rhs) -> np.ndarray:
    k1 = rhs(state, u)
    k2 = rhs(state + 0.5 * h * k1, u)
    k3 = rhs(state + 0.5 * h * k2, u)
    k4 = rhs(state + h * k3, u)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_without_safety(model: ReducedModel, events: list[dict]):
    x = np.zeros(model.nx, dtype=float)
    X = np.zeros((len(T_GRID), model.nx), dtype=float)

    def rhs(x_state, u):
        return model.A @ x_state + model.B @ u

    for k, t in enumerate(T_GRID):
        X[k, :] = x
        if k == len(T_GRID) - 1:
            break
        u = input_at_time(t, model, events)
        x = rk4_step(x, u, DT, rhs)

    traj = reconstruct_trajectory(model, events, X, None, None)
    rocof = np.zeros((len(T_GRID), model.nF), dtype=float)
    for k in range(len(T_GRID)):
        xk = traj["x"][k, :]
        uk = traj["u"][k, :]
        rocof[k, :] = model.Cw @ (model.A @ xk + model.B @ uk)

    traj["rocof_exact"] = rocof
    return traj


def simulate_with_dynamic_filter(model: ReducedModel, events: list[dict]):
    nF = model.nF
    q = np.zeros(model.nx + 2 * nF, dtype=float)

    X = np.zeros((len(T_GRID), model.nx), dtype=float)
    Z = np.zeros((len(T_GRID), nF), dtype=float)
    CHI = np.zeros((len(T_GRID), nF), dtype=float)
    TARGET = np.zeros((len(T_GRID), nF), dtype=float)
    AINT = np.zeros((len(T_GRID), nF), dtype=float)
    BINT = np.zeros((len(T_GRID), nF), dtype=float)
    ROCOF_HAT = np.zeros((len(T_GRID), nF), dtype=float)
    ROCOF_EXACT = np.zeros((len(T_GRID), nF), dtype=float)

    mode = ROCOF_MODE.lower().strip()
    if mode not in {"dirty", "exact"}:
        raise ValueError("ROCOF_MODE must be 'dirty' or 'exact'.")

    def rhs(q_state, u):
        x = q_state[: model.nx]
        z = q_state[model.nx : model.nx + nF]
        chi = q_state[model.nx + nF :]

        xdot = model.A @ x + model.B @ u + model.Gs @ z
        omega_f = model.Cw @ x
        exact_rocof = model.Cw @ xdot

        chi_dot = (-chi + omega_f) / ROCOF_TAU_S
        rocof_hat = exact_rocof if mode == "exact" else (omega_f - chi) / ROCOF_TAU_S

        target, _, _ = local_filter_target(omega_f, rocof_hat, z, model)
        zdot = (-z + target) / FILTER_EPSILON_S

        return np.concatenate([xdot, zdot, chi_dot])

    for k, t in enumerate(T_GRID):
        x = q[: model.nx]
        z = q[model.nx : model.nx + nF]
        chi = q[model.nx + nF :]
        u = input_at_time(t, model, events)

        xdot = model.A @ x + model.B @ u + model.Gs @ z
        omega_f = model.Cw @ x
        exact_rocof = model.Cw @ xdot
        rocof_hat = exact_rocof if mode == "exact" else (omega_f - chi) / ROCOF_TAU_S
        target, a, b = local_filter_target(omega_f, rocof_hat, z, model)

        X[k, :] = x
        Z[k, :] = z
        CHI[k, :] = chi
        TARGET[k, :] = target
        AINT[k, :] = a
        BINT[k, :] = b
        ROCOF_HAT[k, :] = rocof_hat
        ROCOF_EXACT[k, :] = exact_rocof

        if k == len(T_GRID) - 1:
            break

        q = rk4_step(q, u, DT, rhs)

    traj = reconstruct_trajectory(model, events, X, Z, CHI)
    traj.update(
        {
            "target": TARGET,
            "a": AINT,
            "b": BINT,
            "rocof_hat": ROCOF_HAT,
            "rocof_exact": ROCOF_EXACT,
        }
    )
    return traj


def reconstruct_trajectory(
    model: ReducedModel,
    events: list[dict],
    X: np.ndarray,
    Z: np.ndarray | None,
    CHI: np.ndarray | None,
):
    """Reconstruct zero-started state and power deviations."""
    nt = X.shape[0]
    nF = model.nF
    z_hist = np.zeros((nt, nF), dtype=float) if Z is None else Z

    omega_f = X @ model.Cw.T
    omega_g = X[:, model.nG - 1 : model.nG - 1 + model.nG]
    delta_pm = X[:, model.nG - 1 + model.nG :]

    theta_full = np.zeros((nt, model.I0.shape[0]), dtype=float)
    delta_pF_ref = np.zeros((nt, nF), dtype=float)
    delta_pF_ffr = np.zeros((nt, nF), dtype=float)
    delta_pF_nom = np.zeros((nt, nF), dtype=float)
    delta_pF = np.zeros((nt, nF), dtype=float)
    u_hist = np.zeros((nt, model.nu), dtype=float)

    kf_diag = np.diag(model.Kf)

    for k, t in enumerate(T_GRID):
        u = input_at_time(t, model, events)
        u_hist[k, :] = u
        x = X[k, :]
        z = z_hist[k, :]

        y = model.Yx @ x + model.Yu @ u + model.Yz @ z
        theta_f = y[:nF]
        theta_l = y[nF : nF + model.nL]
        theta_g = x[: model.nG - 1]

        theta_red = model.SbarG @ theta_g + model.SbarF @ theta_f + model.SbarL @ theta_l
        theta_full[k, :] = model.I0 @ theta_red

        delta_pF_ref[k, :] = u[model.nG : model.nG + nF]
        delta_pF_ffr[k, :] = -kf_diag * omega_f[k, :]
        delta_pF_nom[k, :] = delta_pF_ref[k, :] + delta_pF_ffr[k, :]
        delta_pF[k, :] = delta_pF_nom[k, :] + z

    return {
        "x": X,
        "z": z_hist,
        "chi": CHI,
        "omega_f": omega_f,
        "omega_g": omega_g,
        "delta_pm": delta_pm,
        "theta_full": theta_full,
        "delta_pF_ref": delta_pF_ref,
        "delta_pF_ffr": delta_pF_ffr,
        "delta_pF_nom": delta_pF_nom,
        "delta_pF": delta_pF,
        "u": u_hist,
    }


# =============================================================================
# Diagnostics and exports
# =============================================================================


def exact_rocof(model: ReducedModel, traj: dict) -> np.ndarray:
    if "rocof_exact" in traj:
        return traj["rocof_exact"]

    out = np.zeros_like(traj["omega_f"])
    for k in range(len(T_GRID)):
        x = traj["x"][k, :]
        u = traj["u"][k, :]
        z = traj["z"][k, :]
        out[k, :] = model.Cw @ (model.A @ x + model.B @ u + model.Gs @ z)
    return out


def cbf_diagnostics(model: ReducedModel, traj: dict):
    omega = traj["omega_f"]
    rocof = exact_rocof(model, traj)
    low_res = rocof + model.alpha_low * (omega - model.lower_pu)
    up_res = model.alpha_up * (model.upper_pu - omega) - rocof
    return low_res, up_res


def print_performance_summary(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    filter_enabled: bool,
) -> None:
    side = _active_safety_side(model, baseline, active)
    bound = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    metric = "nadir" if side == "lower" else "zenith"

    print("\nFrequency-safety summary")
    print(f"  Dynamic safety filter = {'ON' if filter_enabled else 'OFF'}")
    print(f"  active constraint: {side} ({metric}), Delta f limit={bound:+.4f} Hz")

    if filter_enabled:
        print("  Bus | filter OFF [Hz] | filter ON [Hz] | peak |z_i| [p.u.]")
        print("  " + "-" * 61)
        for i, bus in enumerate(model.GFL_ext):
            f0 = F_BASE_HZ * baseline["omega_f"][:, i]
            fa = F_BASE_HZ * active["omega_f"][:, i]
            m0 = np.min(f0) if side == "lower" else np.max(f0)
            ma = np.min(fa) if side == "lower" else np.max(fa)
            zp = np.max(np.abs(_filter_state_trace(model, active, bus)))
            print(f"  {bus:>3d} | {m0:>+12.4f} | {ma:>+14.4f} | {zp:>15.4f}")
    else:
        print("  Bus | filter OFF [Hz]")
        print("  " + "-" * 24)
        for i, bus in enumerate(model.GFL_ext):
            f = F_BASE_HZ * active["omega_f"][:, i]
            m = np.min(f) if side == "lower" else np.max(f)
            print(f"  {bus:>3d} | {m:>+12.4f}")

    low_res, up_res = cbf_diagnostics(model, active)
    active_res = low_res if side == "lower" else up_res
    print(f"  minimum active-CBF residual: {np.min(active_res):+.6e} p.u./s")


def validate_filter_state_trajectory(
    model: ReducedModel,
    traj: dict,
    filter_enabled: bool,
) -> None:
    z = np.asarray(traj.get("z"), dtype=float)
    expected = (len(T_GRID), model.nF)
    if z.shape != expected:
        raise ValueError(f"Filter-state array z has shape {z.shape}; expected {expected}.")
    if not np.all(np.isfinite(z)):
        raise FloatingPointError("Filter-state array z contains non-finite values.")

    dp_nom = np.asarray(traj.get("delta_pF_nom"), dtype=float)
    dp_applied = np.asarray(traj.get("delta_pF"), dtype=float)
    if dp_nom.shape != expected or dp_applied.shape != expected:
        raise ValueError("GFL power-deviation arrays are inconsistent with the GFL dimension.")

    identity_error = float(np.max(np.abs((dp_applied - dp_nom) - z)))
    tol = 5e-10
    if identity_error > tol:
        raise RuntimeError(
            "Filter-state consistency check failed: "
            f"max |Delta p_F - Delta p_F^nom - z| = {identity_error:.3e}."
        )

    if not filter_enabled:
        off_error = float(np.max(np.abs(z)))
        if off_error > tol:
            raise RuntimeError(
                "Safety filter is OFF but z is not identically zero: "
                f"max |z| = {off_error:.3e}."
            )

    if not PRINT_FILTER_STATE_DIAGNOSTICS:
        return

    print("\nFilter-state consistency")
    print(f"  GFL state-column order: {model.GFL_ext}")
    print(
        "  max |Delta p_F - Delta p_F^nom - z| = "
        f"{identity_error:.3e} p.u."
    )
    print("  Bus | column | peak z_i [p.u.] | t_peak [s] | final z_i [p.u.]")
    print("  " + "-" * 68)
    for bus in model.GFL_ext:
        i = _gfl_index(model, bus)
        zi = _filter_state_trace(model, traj, bus)
        k_peak = int(np.argmax(np.abs(zi)))
        print(
            f"  {bus:>3d} | {i:>6d} | {zi[k_peak]:>15.6f} | "
            f"{T_GRID[k_peak]:>10.4f} | {zi[-1]:>16.6e}"
        )


def _report_matrix(name: str, matrix: np.ndarray, row_labels=None, col_labels=None) -> list[str]:
    """Return a readable text block for a matrix used in the analytical model."""
    A = np.asarray(matrix, dtype=float)
    lines = [f"{name} shape={A.shape}"]
    if row_labels is not None:
        lines.append("  rows   : " + ", ".join(str(x) for x in row_labels))
    if col_labels is not None:
        lines.append("  columns: " + ", ".join(str(x) for x in col_labels))
    lines.append(
        np.array2string(
            A,
            precision=7,
            suppress_small=False,
            max_line_width=220,
            separator=", ",
        )
    )
    return lines


def write_simulation_report(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
    inv_map: dict[int, int],
) -> str:
    """Write a compact human-readable report for every successful simulation run."""
    ensure_dir(RESULTS_DIR)

    side = _active_safety_side(model, baseline, active)
    gamma_sym = 0.5 * (model.Gamma + model.Gamma.T)
    gamma_eigs = la.eigvalsh(gamma_sym)
    hg_svals = la.svdvals(model.HG)
    load_ext = [int(inv_map.get(int(bus), int(bus))) for bus in model.L_int]

    lines: list[str] = []
    add = lines.append
    add("IEEE 39-BUS DYNAMIC LOCAL SAFETY-FILTER REPORT")
    add("=" * 78)
    add("")
    add("RUN CONFIGURATION")
    add("-" * 78)
    add(f"System base                       : {SYS_BASE_MVA:.6g} MVA")
    add(f"Base frequency                    : {F_BASE_HZ:.6g} Hz")
    add(f"Simulation step                   : {DT:.6g} s")
    add(f"Final simulation time             : {T_FINAL:.6g} s")
    add(f"Slack/reference bus               : {SLACK_BUS_EXT}")
    add(f"Dynamic safety filter             : {'ON' if filter_enabled else 'OFF'}")
    add(f"Nominal comparison                : {'ON' if RUN_NOMINAL_COMPARISON else 'OFF'}")
    add(f"RoCoF mode                        : {ROCOF_MODE}")
    add(f"RoCoF differentiator tau_d        : {ROCOF_TAU_S:.6g} s")
    add(f"Dynamic-filter epsilon            : {FILTER_EPSILON_S:.6g} s")
    add(f"CBF alpha                         : {ALPHA_CBF:.6g}")
    add(f"Frequency lower bound             : {FREQ_LOWER_HZ:+.6g} Hz")
    add(f"Frequency upper bound             : {FREQ_UPPER_HZ:+.6g} Hz")
    add(f"Active safety side                : {side}")
    add("")

    add("RESOURCE PARTITION AND STATE DIMENSIONS")
    add("-" * 78)
    add(f"SG buses (external order)         : {model.SG_ext}")
    add(f"GFL buses (external order)        : {model.GFL_ext}")
    add(f"Load/algebraic buses              : {load_ext}")
    add(f"SG internal-bus order             : {model.SG_int}")
    add(f"GFL internal-bus order            : {model.GFL_int}")
    add(f"Number of SG buses nG             : {model.nG}")
    add(f"Number of GFL buses nF            : {model.nF}")
    add(f"Number of load buses nL           : {model.nL}")
    add(f"Differential-state dimension nx   : {model.nx}")
    add(f"Input dimension nu                : {model.nu}")
    add("")
    add("Bus role table")
    add("  Bus | Role | divider row/column")
    add("  " + "-" * 40)
    for i, bus in enumerate(model.GFL_ext):
        add(f"  {bus:>3d} | GFL  | H_G row {i}")
    for j, bus in enumerate(model.SG_ext):
        role = "SG (reference)" if bus == SLACK_BUS_EXT else "SG"
        add(f"  {bus:>3d} | {role:<14s} | H_G column {j}")
    add("")

    add("FREQUENCY DIVIDER")
    add("-" * 78)
    add("Interpretation: omega_F = H_G omega_G. Rows follow GFL_ext; columns follow SG_ext.")
    lines.extend(
        _report_matrix(
            "H_G",
            model.HG,
            row_labels=[f"Bus {b} (GFL)" for b in model.GFL_ext],
            col_labels=[f"Bus {b} (SG)" for b in model.SG_ext],
        )
    )
    add(f"rank(H_G)                         : {np.linalg.matrix_rank(model.HG)}/{model.nF}")
    add(f"singular values(H_G)              : {np.array2string(hg_svals, precision=7)}")
    add("")

    add("CORRECTIVE-INPUT SENSITIVITY")
    add("-" * 78)
    lines.extend(
        _report_matrix(
            "Gamma = C_omega G_s",
            model.Gamma,
            row_labels=[f"Bus {b} (GFL RoCoF)" for b in model.GFL_ext],
            col_labels=[f"Bus {b} (GFL correction)" for b in model.GFL_ext],
        )
    )
    add(f"Gamma symmetry error (Frobenius) : {la.norm(model.Gamma-model.Gamma.T, ord='fro'):.7e}")
    add(f"Gamma eigenvalues                 : {np.array2string(gamma_eigs, precision=7)}")
    add(f"lambda_min(Gamma)                 : {gamma_eigs[0]:.7e}")
    add(f"lambda_max(Gamma)                 : {gamma_eigs[-1]:.7e}")
    add(f"cond(Gamma)                       : {gamma_eigs[-1] / gamma_eigs[0]:.7e}")
    add("")

    add("RESOURCE PARAMETERS")
    add("-" * 78)
    add("SG bus | M_i [p.u.] | D_i [p.u.] | frequency role")
    add("  " + "-" * 58)
    for i, bus in enumerate(model.SG_ext):
        role = "reference" if bus == SLACK_BUS_EXT else "non-reference"
        add(f"  {bus:>3d}  | {model.M_diag[i]:>11.6f} | {model.D_diag[i]:>11.6f} | {role}")
    add("")
    add("GFL bus | K_FFR | Gamma_ii [1/s] | k_i [s] | k_i Gamma_ii")
    add("  " + "-" * 70)
    gamma_diag = np.diag(gamma_sym)
    kf_diag = np.diag(model.Kf)
    for i, bus in enumerate(model.GFL_ext):
        add(
            f"  {bus:>3d}  | {kf_diag[i]:>7.3f} | {gamma_diag[i]:>14.7e} | "
            f"{model.prox_k[i]:>9.6f} | {model.prox_k[i] * gamma_diag[i]:>13.7f}"
        )
    add("")

    add("DISTURBANCES")
    add("-" * 78)
    if events:
        for event in events:
            add(f"t={event['time_s']:.6f} s : {event['label']}")
    else:
        add("none")
    add("")

    add("FREQUENCY RESPONSE SUMMARY")
    add("-" * 78)
    add("Values are frequency deviations from 60 Hz.")
    add("Bus | Type | OFF min [Hz] | OFF max [Hz] | ON min [Hz] | ON max [Hz]")
    add("  " + "-" * 76)
    for bus in model.GFL_ext + model.SG_ext:
        f_off = _frequency_trace_hz(model, baseline, bus)
        f_on = _frequency_trace_hz(model, active, bus)
        role = "GFL" if bus in model.GFL_ext else "SG"
        add(
            f"{bus:>3d} | {role:>4s} | {np.min(f_off):>+12.6f} | {np.max(f_off):>+12.6f} | "
            f"{np.min(f_on):>+11.6f} | {np.max(f_on):>+11.6f}"
        )
    add("")

    add("ACTIVE-POWER RESPONSE SUMMARY")
    add("-" * 78)
    add("GFL entries are applied active-power deviations; SG entries are mechanical-power deviations.")
    add("Bus | Type | OFF min [pu] | OFF max [pu] | ON min [pu] | ON max [pu]")
    add("  " + "-" * 76)
    for bus in model.GFL_ext + model.SG_ext:
        p_off = _power_trace_pu(model, baseline, bus)
        p_on = _power_trace_pu(model, active, bus)
        role = "GFL" if bus in model.GFL_ext else "SG"
        add(
            f"{bus:>3d} | {role:>4s} | {np.min(p_off):>+12.6f} | {np.max(p_off):>+12.6f} | "
            f"{np.min(p_on):>+11.6f} | {np.max(p_on):>+11.6f}"
        )
    add("")

    if filter_enabled:
        add("DYNAMIC SAFETY-FILTER SUMMARY")
        add("-" * 78)
        add("Bus | peak |z_i| [pu] | t_peak [s] | final z_i [pu]")
        add("  " + "-" * 62)
        for bus in model.GFL_ext:
            z = _filter_state_trace(model, active, bus)
            k = int(np.argmax(np.abs(z)))
            add(f"{bus:>3d} | {np.max(np.abs(z)):>15.7f} | {T_GRID[k]:>10.6f} | {z[-1]:>+14.7e}")
        add("")

    low_res, up_res = cbf_diagnostics(model, active)
    add("CBF DIAGNOSTICS")
    add("-" * 78)
    add(f"minimum lower-CBF residual [pu/s] : {np.min(low_res):+.7e}")
    add(f"minimum upper-CBF residual [pu/s] : {np.min(up_res):+.7e}")
    add("")
    add("END OF REPORT")

    report_path = os.path.abspath(REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"Run report: {report_path}")
    return report_path


def write_csv(path: str, header: list[str], data: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(data.tolist())


def export_results(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    filter_enabled: bool,
) -> None:
    ensure_dir(CSV_DIR)
    tag = "filter_on" if filter_enabled else "filter_off"

    freq_header = ["t_s"]
    freq_cols = [T_GRID]
    for i, bus in enumerate(model.GFL_ext):
        freq_header.append(f"gfl_bus{bus}_filter_off_hz")
        freq_cols.append(F_BASE_HZ * baseline["omega_f"][:, i])
        if filter_enabled:
            freq_header.append(f"gfl_bus{bus}_filter_on_hz")
            freq_cols.append(F_BASE_HZ * active["omega_f"][:, i])
    write_csv(
        os.path.join(CSV_DIR, f"gfl_frequency_{tag}.csv"),
        freq_header,
        np.column_stack(freq_cols),
    )

    z_header = ["t_s"] + [f"bus{bus}_z_pu" for bus in model.GFL_ext]
    write_csv(
        os.path.join(CSV_DIR, f"filter_state_{tag}.csv"),
        z_header,
        np.column_stack([T_GRID, active["z"]]),
    )

    gfl_header = ["t_s"]
    gfl_cols = [T_GRID]
    for i, bus in enumerate(model.GFL_ext):
        gfl_header += [
            f"bus{bus}_delta_p_ref_pu",
            f"bus{bus}_delta_p_ffr_pu",
            f"bus{bus}_delta_p_nom_pu",
            f"bus{bus}_z_pu",
            f"bus{bus}_delta_p_F_pu",
        ]
        gfl_cols += [
            active["delta_pF_ref"][:, i],
            active["delta_pF_ffr"][:, i],
            active["delta_pF_nom"][:, i],
            active["z"][:, i],
            active["delta_pF"][:, i],
        ]
    write_csv(
        os.path.join(CSV_DIR, f"gfl_power_decomposition_{tag}.csv"),
        gfl_header,
        np.column_stack(gfl_cols),
    )

    sg_header = ["t_s"] + [f"bus{bus}_delta_p_m_pu" for bus in model.SG_ext]
    sg_cols = [T_GRID] + [
        active["delta_pm"][:, i] for i in range(len(model.SG_ext))
    ]
    write_csv(
        os.path.join(CSV_DIR, f"sg_mechanical_power_deviation_{tag}.csv"),
        sg_header,
        np.column_stack(sg_cols),
    )

    rocof = exact_rocof(model, active)
    r_header = ["t_s"]
    r_cols = [T_GRID]
    for i, bus in enumerate(model.GFL_ext):
        r_header.append(f"bus{bus}_rocof_exact_hz_s")
        r_cols.append(F_BASE_HZ * rocof[:, i])
        if "rocof_hat" in active:
            r_header.append(f"bus{bus}_rocof_hat_hz_s")
            r_cols.append(F_BASE_HZ * active["rocof_hat"][:, i])
    write_csv(
        os.path.join(CSV_DIR, f"gfl_rocof_{tag}.csv"),
        r_header,
        np.column_stack(r_cols),
    )

    low_res, up_res = cbf_diagnostics(model, active)
    cbf_header = ["t_s"]
    cbf_cols = [T_GRID]
    for i, bus in enumerate(model.GFL_ext):
        omega_hz = F_BASE_HZ * active["omega_f"][:, i]
        rocof_hz_s = F_BASE_HZ * rocof[:, i]
        low_bound = -model.alpha_low[i] * (omega_hz - FREQ_LOWER_HZ)
        up_bound = model.alpha_up[i] * (FREQ_UPPER_HZ - omega_hz)
        cbf_header += [
            f"bus{bus}_h_lower_hz",
            f"bus{bus}_h_upper_hz",
            f"bus{bus}_rocof_hz_s",
            f"bus{bus}_cbf_lower_bound_hz_s",
            f"bus{bus}_cbf_upper_bound_hz_s",
            f"bus{bus}_lower_residual_hz_s",
            f"bus{bus}_upper_residual_hz_s",
        ]
        cbf_cols += [
            omega_hz - FREQ_LOWER_HZ,
            FREQ_UPPER_HZ - omega_hz,
            rocof_hz_s,
            low_bound,
            up_bound,
            F_BASE_HZ * low_res[:, i],
            F_BASE_HZ * up_res[:, i],
        ]
    write_csv(
        os.path.join(CSV_DIR, f"cbf_diagnostics_{tag}.csv"),
        cbf_header,
        np.column_stack(cbf_cols),
    )

    np.savetxt(os.path.join(CSV_DIR, "H_G.csv"), model.HG, delimiter=",")
    np.savetxt(os.path.join(CSV_DIR, "Gamma.csv"), model.Gamma, delimiter=",")

# =============================================================================
# Publication figures
# =============================================================================

FIGURE_WIDTH_DOUBLE = 7.10
FIGURE_WIDTH_SINGLE = 3.45
LINE_WIDTH_MAIN = 1.50
LINE_WIDTH_AUX = 1.10
AXIS_LINE_WIDTH = 0.82
TICK_LINE_WIDTH = 0.74

FONT_SIZE_BASE = 10.0
FONT_SIZE_XLABEL = 14.0
FONT_SIZE_YLABEL = 12.8
FONT_SIZE_FILTER_YLABEL = 14.4
FONT_SIZE_BARRIER_YLABEL = 14.8
FONT_SIZE_TICK = 9.6
FONT_SIZE_LEGEND = 9.8
FONT_SIZE_BUS_TAG = 10.2
FONT_SIZE_STATUS = 10.0

FIGURE_PNG_DPI = 600
FIGURE_PDF_DPI = 600
FIGURE_SAVE_PAD_IN = 0.030

COLOR_INK = "#252525"
COLOR_AXIS = "#343434"
COLOR_EVENT = "#7D8590"
COLOR_ZERO = "#98A1AA"
COLOR_GRID = "#D8DDE3"

COLOR_BLUE = "#0072B2"
COLOR_SKY = "#56B4E9"
COLOR_TEAL = "#009E73"
COLOR_GREEN = "#3A923A"
COLOR_ORANGE = "#E69F00"
COLOR_VERMILLION = "#D55E00"
COLOR_ROSE = "#CC6677"
COLOR_PURPLE = "#7E57C2"
COLOR_INDIGO = "#5E60CE"
COLOR_CYAN = "#008B8B"

COLOR_SAFE_FILL = "#DDECF5"
COLOR_UNSAFE_FILL = "#F5DEDF"
COLOR_SUPPORT_FILL = "#F3CD85"
COLOR_TRACKING_FILL = "#DDD4EF"

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
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "cmr10",
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": FONT_SIZE_BASE,
            "axes.labelsize": FONT_SIZE_YLABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.edgecolor": COLOR_AXIS,
            "axes.labelcolor": COLOR_INK,
            "text.color": COLOR_INK,
            "xtick.color": COLOR_AXIS,
            "ytick.color": COLOR_AXIS,
            "lines.linewidth": LINE_WIDTH_MAIN,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": TICK_LINE_WIDTH,
            "ytick.major.width": TICK_LINE_WIDTH,
            "legend.frameon": False,
            "legend.handlelength": 2.20,
            "legend.handletextpad": 0.48,
            "legend.columnspacing": 0.90,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _resolve_plot_buses(requested, valid_buses: list[int], name: str) -> list[int]:
    valid = list(dict.fromkeys(int(b) for b in valid_buses))
    if requested is None:
        return valid
    chosen = list(dict.fromkeys(int(b) for b in requested))
    invalid = [b for b in chosen if b not in valid]
    if invalid:
        raise ValueError(f"{name} contains invalid buses {invalid}; valid buses are {valid}.")
    return chosen


def _gfl_index(model: ReducedModel, bus: int) -> int:
    bus = int(bus)
    if bus not in model.GFL_ext:
        raise ValueError(f"Bus {bus} is not a configured GFL bus.")
    return model.GFL_ext.index(bus)


def _gfl_trace(model: ReducedModel, traj: dict, key: str, bus: int) -> np.ndarray:
    if key not in traj:
        raise KeyError(f"Trajectory does not contain '{key}'.")
    data = np.asarray(traj[key], dtype=float)
    if data.ndim != 2 or data.shape != (len(T_GRID), model.nF):
        raise ValueError(
            f"Trajectory '{key}' has shape {data.shape}; "
            f"expected {(len(T_GRID), model.nF)}."
        )
    return data[:, _gfl_index(model, bus)]


def _filter_state_trace(model: ReducedModel, traj: dict, bus: int) -> np.ndarray:
    return _gfl_trace(model, traj, "z", bus)


def _frequency_trace_hz(model: ReducedModel, traj: dict, bus: int) -> np.ndarray:
    if bus in model.SG_ext:
        return F_BASE_HZ * traj["omega_g"][:, model.SG_ext.index(bus)]
    if bus in model.GFL_ext:
        return F_BASE_HZ * _gfl_trace(model, traj, "omega_f", bus)
    raise ValueError(f"Bus {bus} is not a configured SG or GFL resource bus.")


def _power_trace_pu(model: ReducedModel, traj: dict, bus: int) -> np.ndarray:
    if bus in model.SG_ext:
        return traj["delta_pm"][:, model.SG_ext.index(bus)]
    if bus in model.GFL_ext:
        return _gfl_trace(model, traj, "delta_pF", bus)
    raise ValueError(f"Bus {bus} is not a configured SG or GFL resource bus.")


def _bus_color(bus: int) -> str:
    return BUS_COLORS.get(int(bus), COLOR_BLUE)


def _case_tag(filter_enabled: bool) -> str:
    return "filter_on" if filter_enabled else "filter_off"


def _status_text(filter_enabled: bool) -> str:
    return f"Dynamic safety filter = {'ON' if filter_enabled else 'OFF'}"


def _event_lines(ax, events: list[dict]) -> None:
    for event in events:
        ax.axvline(
            event["time_s"],
            color=COLOR_EVENT,
            linestyle=(0, (1.0, 1.7)),
            linewidth=0.78,
            alpha=0.88,
            zorder=0,
        )


def _compact_tick(value, _position) -> str:
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


def _boxed_axis(ax, ybins: int = 4, zero_line: bool = False) -> None:
    ax.set_xlim(float(T_GRID[0]), float(T_GRID[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLOR_AXIS)
        spine.set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        length=3.7,
        width=TICK_LINE_WIDTH,
        pad=2.5,
    )
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.42, alpha=0.40, zorder=-5)
    ax.margins(x=0.0)
    if zero_line:
        ax.axhline(0.0, color=COLOR_ZERO, linewidth=0.72, zorder=0)


def _set_data_ylim(
    ax,
    arrays,
    *,
    include_zero: bool = False,
    lower_pad: float = 0.12,
    upper_pad: float = 0.18,
    min_span: float = 1e-4,
) -> None:
    values = []
    for array in arrays:
        a = np.asarray(array, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size:
            values.append(a)
    if include_zero:
        values.append(np.array([0.0]))
    if not values:
        return
    data = np.concatenate(values)
    lo = float(np.min(data))
    hi = float(np.max(data))
    span = hi - lo
    scale = max(abs(lo), abs(hi), 1.0)
    if span < max(min_span, 1e-7 * scale):
        pad = max(min_span, 0.05 * scale)
        lo -= pad
        hi += pad
        span = hi - lo
    ax.set_ylim(lo - lower_pad * span, hi + upper_pad * span)


def _quiet_side(ax, arrays) -> str:
    if not arrays:
        return "right"
    ylo, yhi = ax.get_ylim()
    yrange = max(yhi - ylo, 1e-12)
    left_peak = -np.inf
    right_peak = -np.inf
    for array in arrays:
        a = np.asarray(array, dtype=float).reshape(-1)
        if not a.size:
            continue
        m = max(2, int(0.18 * a.size))
        left_peak = max(left_peak, float(np.nanmax(a[:m])))
        right_peak = max(right_peak, float(np.nanmax(a[-m:])))
    left_clear = (yhi - left_peak) / yrange if np.isfinite(left_peak) else 1.0
    right_clear = (yhi - right_peak) / yrange if np.isfinite(right_peak) else 1.0
    return "left" if left_clear > right_clear + 0.04 else "right"


def _panel_bus_label(ax, bus: int, arrays=None) -> None:
    side = _quiet_side(ax, arrays)
    x = 0.018 if side == "left" else 0.982
    ax.text(
        x,
        0.972,
        f"Bus {bus}",
        transform=ax.transAxes,
        ha="left" if side == "left" else "right",
        va="top",
        fontsize=FONT_SIZE_BUS_TAG,
        color=COLOR_INK,
        zorder=40,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.7},
    )


def _status_label(ax, filter_enabled: bool, arrays=None) -> None:
    side = _quiet_side(ax, arrays)
    x = 0.018 if side == "left" else 0.982
    ax.text(
        x,
        0.972,
        _status_text(filter_enabled),
        transform=ax.transAxes,
        ha="left" if side == "left" else "right",
        va="top",
        fontsize=FONT_SIZE_STATUS,
        color=COLOR_TEAL if filter_enabled else COLOR_VERMILLION,
        zorder=40,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.7},
    )


def _barrier_symbol(side: str) -> str:
    if side == "lower":
        return r"$\frac{h}{\!}\!_i$"
    return r"$\overline{h}_i$"


def _barrier_ylabel(side: str) -> str:
    if side == "lower":
        return r"$\frac{h}{\!}\!_i\,[\mathrm{Hz}]$"
    return r"$\overline{h}_i\,[\mathrm{Hz}]$"


def _set_barrier_ylabel(ax, side: str) -> None:
    ax.set_ylabel(
        _barrier_ylabel(side),
        fontsize=FONT_SIZE_BARRIER_YLABEL,
        labelpad=5.0,
    )


def _annotate_safety_boundary(ax) -> None:
    ylo, yhi = ax.get_ylim()
    y_text = -0.018 * (yhi - ylo)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.985,
        y_text,
        "Safety boundary",
        transform=trans,
        ha="right",
        va="top",
        fontsize=FONT_SIZE_LEGEND,
        color=COLOR_ROSE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.25},
        zorder=7,
        clip_on=False,
    )


def _make_bus_grid(n: int, row_height: float = 1.66):
    if n < 1:
        raise ValueError("At least one bus must be selected for a figure.")
    if BUS_SUBPLOT_COLUMNS != 1:
        raise ValueError("BUS_SUBPLOT_COLUMNS must be 1 for the publication layout.")

    nrows = n
    width = FIGURE_WIDTH_SINGLE if n == 1 else FIGURE_WIDTH_DOUBLE
    height = 2.35 if n == 1 else max(3.0, 0.72 + row_height * nrows)
    fig, ax_array = plt.subplots(
        nrows,
        1,
        figsize=(width, height),
        sharex=nrows > 1,
        squeeze=False,
    )
    axes = [ax_array[row, 0] for row in range(nrows)]
    positions = [(row, 0, 1) for row in range(nrows)]
    return fig, axes, positions, nrows, 1


def _finish_bus_grid(
    fig,
    axes,
    positions,
    nrows: int,
    ylabel: str,
    legend_handles,
    legend_labels,
    legend_ncol: int = 3,
    ylabel_fontsize: float | None = None,
) -> None:
    ysize = FONT_SIZE_YLABEL if ylabel_fontsize is None else float(ylabel_fontsize)
    for ax, (row, col, _span) in zip(axes, positions):
        if row == nrows - 1:
            ax.set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=ysize, labelpad=5.0)

    ncol = min(max(1, legend_ncol), len(legend_labels))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.994),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONT_SIZE_LEGEND,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.065,
        top=0.965,
        hspace=0.115,
    )
    fig.align_ylabels()


def _safe_savefig(fig, path: str, **kwargs) -> str:
    target = os.path.abspath(os.fspath(path))
    ensure_dir(os.path.dirname(target))
    suffix = os.path.splitext(target)[1] or ".tmp"
    fd, staged = tempfile.mkstemp(prefix="cbf_fig_", suffix=suffix)
    os.close(fd)
    try:
        fig.savefig(staged, **kwargs)
        try:
            os.replace(staged, target)
            staged = None
        except OSError:
            shutil.copyfile(staged, target)
        return target
    except Exception as exc:
        raise RuntimeError(
            f"Could not save figure '{os.path.basename(target)}' to '{os.path.dirname(target)}'."
        ) from exc
    finally:
        if staged and os.path.exists(staged):
            try:
                os.remove(staged)
            except OSError:
                pass


def save_figure(fig, name: str) -> None:
    ensure_dir(FIGURE_DIR)
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)).strip("._ ")
    if not safe_name:
        safe_name = "figure"
    common = dict(bbox_inches="tight", pad_inches=FIGURE_SAVE_PAD_IN, facecolor="white")
    _safe_savefig(
        fig,
        os.path.join(FIGURE_DIR, f"{safe_name}.pdf"),
        dpi=FIGURE_PDF_DPI,
        **common,
    )
    _safe_savefig(
        fig,
        os.path.join(FIGURE_DIR, f"{safe_name}.png"),
        dpi=FIGURE_PNG_DPI,
        **common,
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _active_safety_side(model: ReducedModel, baseline: dict, active: dict) -> str:
    mode = str(ACTIVE_SAFETY_BOUND).strip().lower()
    if mode in {"lower", "upper"}:
        return mode
    if mode != "auto":
        raise ValueError('ACTIVE_SAFETY_BOUND must be "auto", "lower", or "upper".')
    data = np.concatenate(
        [
            F_BASE_HZ * baseline["omega_f"].reshape(-1),
            F_BASE_HZ * active["omega_f"].reshape(-1),
        ]
    )
    fmin = float(np.min(data))
    fmax = float(np.max(data))
    lower_activity = max(0.0, -fmin) / max(abs(FREQ_LOWER_HZ), 1e-9)
    upper_activity = max(0.0, fmax) / max(abs(FREQ_UPPER_HZ), 1e-9)
    return "lower" if lower_activity >= upper_activity else "upper"


def _frequency_ylim(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    buses: list[int],
    side: str,
):
    traces = []
    for bus in buses:
        traces.append(_frequency_trace_hz(model, baseline, bus))
        traces.append(_frequency_trace_hz(model, active, bus))
    data = np.concatenate(traces) if traces else np.array([0.0])
    bound = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    lo = min(float(np.min(data)), bound, 0.0)
    hi = max(float(np.max(data)), bound, 0.0)
    span = max(hi - lo, 1e-3)
    return lo - 0.06 * span, hi + 0.08 * span


def _draw_active_frequency_limit(ax, side: str) -> None:
    value = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    label = r"$\Delta f_{\min}$" if side == "lower" else r"$\Delta f_{\max}$"
    ax.axhline(
        value,
        color=COLOR_ROSE,
        linestyle=(0, (4.0, 2.2)),
        linewidth=1.00,
        zorder=1,
    )
    ax.text(
        0.987,
        value,
        label,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom" if side == "lower" else "top",
        fontsize=7.5,
        color=COLOR_ROSE,
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.7},
    )


def plot_frequency_response(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(
        PLOT_FREQUENCY_BUSES_EXT,
        model.SG_ext + model.GFL_ext,
        "PLOT_FREQUENCY_BUSES_EXT",
    )
    if not buses:
        return

    side = _active_safety_side(model, baseline, active)
    compare = bool(filter_enabled and RUN_NOMINAL_COMPARISON)
    if compare:
        fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_DOUBLE, 4.50), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_DOUBLE, 2.72))
        axes = np.atleast_1d(ax)

    handles, labels = [], []
    nominal_arrays, active_arrays = [], []
    for bus in buses:
        color = _bus_color(bus)
        f_nom = _frequency_trace_hz(model, baseline, bus)
        f_act = _frequency_trace_hz(model, active, bus)
        if compare:
            line, = axes[0].plot(T_GRID, f_nom, color=color, linewidth=1.38)
            axes[1].plot(T_GRID, f_act, color=color, linewidth=1.38)
            nominal_arrays.append(f_nom)
        else:
            line, = axes[0].plot(T_GRID, f_act, color=color, linewidth=1.42)
        active_arrays.append(f_act)
        handles.append(line)
        labels.append(f"Bus {bus}")

    ylo, yhi = _frequency_ylim(model, baseline, active, buses, side)
    for ax in axes:
        _draw_active_frequency_limit(ax, side)
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=5)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(r"$\Delta f_i\,[\mathrm{Hz}]$", fontsize=FONT_SIZE_YLABEL, labelpad=5.0)

    if compare:
        _status_label(axes[0], False, nominal_arrays)
        _status_label(axes[1], True, active_arrays)
        axes[1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.095, top=0.895, hspace=0.06)
        name = "frequency_response_comparison"
    else:
        _status_label(axes[0], filter_enabled, active_arrays)
        axes[0].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.16, top=0.84)
        name = f"frequency_response_{_case_tag(filter_enabled)}"

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.994),
        ncol=min(5, max(1, len(buses))),
        frameon=False,
        borderaxespad=0.0,
    )
    save_figure(fig, name)


def plot_power_overlay(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(
        PLOT_POWER_OVERLAY_BUSES_EXT,
        model.SG_ext + model.GFL_ext,
        "PLOT_POWER_OVERLAY_BUSES_EXT",
    )
    if not buses:
        return

    compare = bool(filter_enabled and RUN_NOMINAL_COMPARISON)
    if compare:
        fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_DOUBLE, 4.50), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_DOUBLE, 2.72))
        axes = np.atleast_1d(ax)

    handles, labels = [], []
    nominal_arrays, active_arrays = [], []
    for bus in buses:
        p_nom = _power_trace_pu(model, baseline, bus)
        p_act = _power_trace_pu(model, active, bus)
        color = _bus_color(bus)
        if compare:
            line, = axes[0].plot(T_GRID, p_nom, color=color, linewidth=1.42)
            axes[1].plot(T_GRID, p_act, color=color, linewidth=1.42)
            nominal_arrays.append(p_nom)
        else:
            line, = axes[0].plot(T_GRID, p_act, color=color, linewidth=1.45)
        active_arrays.append(p_act)
        handles.append(line)
        labels.append(f"Bus {bus}")

    all_arrays = nominal_arrays + active_arrays if compare else active_arrays
    _set_data_ylim(axes[0], all_arrays, include_zero=True, lower_pad=0.10, upper_pad=0.16)
    for ax in axes:
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=5, zero_line=True)
        ax.set_ylabel(r"$\Delta p_i\,[\mathrm{p.u.}]$", fontsize=FONT_SIZE_YLABEL, labelpad=5.0)

    if compare:
        _status_label(axes[0], False, nominal_arrays)
        _status_label(axes[1], True, active_arrays)
        axes[1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.095, top=0.895, hspace=0.06)
        name = "power_deviation_overlay_comparison"
    else:
        _status_label(axes[0], filter_enabled, active_arrays)
        axes[0].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.16, top=0.84)
        name = f"power_deviation_overlay_{_case_tag(filter_enabled)}"

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.994),
        ncol=min(5, max(1, len(buses))),
        frameon=False,
        borderaxespad=0.0,
    )
    save_figure(fig, name)


def _resource_plot_style(model: ReducedModel, bus: int):
    """Return line style used by the additional mixed-resource overlays."""
    if bus in model.GFL_ext:
        return "-", None, 1.48
    if bus in model.SG_ext:
        return "--", (5.5, 2.8), 1.42
    raise ValueError(f"Bus {bus} is not a configured SG or GFL resource bus.")


def _resource_label(model: ReducedModel, bus: int) -> str:
    role = "GFL" if bus in model.GFL_ext else "SG"
    return f"Bus {bus} ({role})"


def plot_resource_frequency_overlay(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    """Additional frequency overlay: GFL buses solid, SG buses dashed."""
    valid = model.GFL_ext + model.SG_ext
    buses = _resolve_plot_buses(
        PLOT_RESOURCE_FREQUENCY_BUSES_EXT,
        valid,
        "PLOT_RESOURCE_FREQUENCY_BUSES_EXT",
    )
    if not buses:
        return

    side = _active_safety_side(model, baseline, active)
    compare = bool(filter_enabled and RUN_NOMINAL_COMPARISON)

    if compare:
        fig, axes = plt.subplots(
            2, 1, figsize=(FIGURE_WIDTH_DOUBLE, 4.85), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_DOUBLE, 2.95))
        axes = np.atleast_1d(ax)

    handles, labels = [], []
    nominal_arrays, active_arrays = [], []

    for bus in buses:
        f_off = _frequency_trace_hz(model, baseline, bus)
        f_on = _frequency_trace_hz(model, active, bus)
        ls, dashes, lw = _resource_plot_style(model, bus)
        kwargs = dict(color=_bus_color(bus), linewidth=lw, linestyle=ls)
        if dashes is not None:
            kwargs["dashes"] = dashes

        if compare:
            line, = axes[0].plot(T_GRID, f_off, **kwargs)
            axes[1].plot(T_GRID, f_on, **kwargs)
            nominal_arrays.append(f_off)
        else:
            line, = axes[0].plot(T_GRID, f_on, **kwargs)
        active_arrays.append(f_on)
        handles.append(line)
        labels.append(_resource_label(model, bus))

    ylo, yhi = _frequency_ylim(model, baseline, active, buses, side)
    for ax in axes:
        _draw_active_frequency_limit(ax, side)
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=5)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(r"$\Delta f_i\,[\mathrm{Hz}]$", fontsize=FONT_SIZE_YLABEL, labelpad=5.0)

    if compare:
        _status_label(axes[0], False, nominal_arrays)
        _status_label(axes[1], True, active_arrays)
        axes[1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.105, top=0.845, hspace=0.065)
        name = "resource_frequency_overlay_gfl_sg_comparison"
    else:
        _status_label(axes[0], filter_enabled, active_arrays)
        axes[0].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.17, top=0.80)
        name = f"resource_frequency_overlay_gfl_sg_{_case_tag(filter_enabled)}"

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(5, max(1, len(buses))),
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONT_SIZE_LEGEND,
    )
    save_figure(fig, name)


def plot_resource_power_overlay(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    """Additional power overlay: GFL active power solid, SG mechanical power dashed."""
    valid = model.GFL_ext + model.SG_ext
    buses = _resolve_plot_buses(
        PLOT_RESOURCE_POWER_BUSES_EXT,
        valid,
        "PLOT_RESOURCE_POWER_BUSES_EXT",
    )
    if not buses:
        return

    compare = bool(filter_enabled and RUN_NOMINAL_COMPARISON)
    if compare:
        fig, axes = plt.subplots(
            2, 1, figsize=(FIGURE_WIDTH_DOUBLE, 4.85), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_DOUBLE, 2.95))
        axes = np.atleast_1d(ax)

    handles, labels = [], []
    nominal_arrays, active_arrays = [], []

    for bus in buses:
        p_off = _power_trace_pu(model, baseline, bus)
        p_on = _power_trace_pu(model, active, bus)
        ls, dashes, lw = _resource_plot_style(model, bus)
        kwargs = dict(color=_bus_color(bus), linewidth=lw, linestyle=ls)
        if dashes is not None:
            kwargs["dashes"] = dashes

        if compare:
            line, = axes[0].plot(T_GRID, p_off, **kwargs)
            axes[1].plot(T_GRID, p_on, **kwargs)
            nominal_arrays.append(p_off)
        else:
            line, = axes[0].plot(T_GRID, p_on, **kwargs)
        active_arrays.append(p_on)
        handles.append(line)
        labels.append(_resource_label(model, bus))

    all_arrays = nominal_arrays + active_arrays if compare else active_arrays
    _set_data_ylim(
        axes[0], all_arrays, include_zero=True, lower_pad=0.10, upper_pad=0.16
    )

    for ax in axes:
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=5, zero_line=True)
        ax.set_ylabel(r"$\Delta p\,[\mathrm{p.u.}]$", fontsize=FONT_SIZE_YLABEL, labelpad=5.0)

    if compare:
        _status_label(axes[0], False, nominal_arrays)
        _status_label(axes[1], True, active_arrays)
        axes[1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.105, top=0.845, hspace=0.065)
        name = "resource_power_overlay_gfl_sg_comparison"
    else:
        _status_label(axes[0], filter_enabled, active_arrays)
        axes[0].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.102, right=0.988, bottom=0.17, top=0.80)
        name = f"resource_power_overlay_gfl_sg_{_case_tag(filter_enabled)}"

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(5, max(1, len(buses))),
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONT_SIZE_LEGEND,
    )
    save_figure(fig, name)


def plot_cbf_constraints(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(PLOT_CBF_BUSES_EXT, model.GFL_ext, "PLOT_CBF_BUSES_EXT")
    if not buses:
        return

    side = _active_safety_side(model, active, active)
    rocof_all = F_BASE_HZ * exact_rocof(model, active)
    fig, axes, positions, nrows, _ = _make_bus_grid(len(buses), row_height=1.66)
    handles = labels = None

    for ax, bus in zip(axes, buses):
        i = model.GFL_ext.index(bus)
        rocof = rocof_all[:, i]
        freq = F_BASE_HZ * active["omega_f"][:, i]
        if side == "lower":
            bound = -model.alpha_low[i] * (freq - FREQ_LOWER_HZ)
            bound_label = "Lower CBF bound"
            bound_color = COLOR_VERMILLION
        else:
            bound = model.alpha_up[i] * (FREQ_UPPER_HZ - freq)
            bound_label = "Upper CBF bound"
            bound_color = COLOR_PURPLE

        _set_data_ylim(ax, [rocof, bound], include_zero=True, lower_pad=0.13, upper_pad=0.22)
        ylo, yhi = ax.get_ylim()
        if side == "lower":
            band = ax.fill_between(T_GRID, bound, yhi, color=COLOR_SAFE_FILL, alpha=0.58, linewidth=0.0, zorder=-3)
        else:
            band = ax.fill_between(T_GRID, ylo, bound, color=COLOR_SAFE_FILL, alpha=0.58, linewidth=0.0, zorder=-3)

        l0, = ax.plot(T_GRID, rocof, color=_bus_color(bus), linewidth=1.52)
        l1, = ax.plot(T_GRID, bound, color=bound_color, linestyle="--", linewidth=1.12)
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=4)
        _panel_bus_label(ax, bus, [rocof, bound])
        if handles is None:
            handles = [l0, l1, band]
            labels = [r"$\dot f_i$", bound_label, "Admissible"]

    _finish_bus_grid(fig, axes, positions, nrows, r"$\dot{f}_i\,[\mathrm{Hz/s}]$", handles, labels, 3)
    save_figure(fig, f"gfl_cbf_constraint_{_case_tag(filter_enabled)}")


def plot_barrier_functions(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(PLOT_BARRIER_BUSES_EXT, model.GFL_ext, "PLOT_BARRIER_BUSES_EXT")
    if not buses:
        return

    side = _active_safety_side(model, active, active)
    fig, axes, positions, nrows, _ = _make_bus_grid(len(buses), row_height=1.66)
    handles = labels = None
    symbol = _barrier_symbol(side)

    for ax, bus in zip(axes, buses):
        i = _gfl_index(model, bus)
        freq = F_BASE_HZ * _gfl_trace(model, active, "omega_f", bus)
        h = freq - FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ - freq
        _set_data_ylim(ax, [h], include_zero=True, lower_pad=0.12, upper_pad=0.24)
        lower_lim, _ = ax.get_ylim()
        unsafe = ax.axhspan(
            lower_lim, 0.0, color=COLOR_UNSAFE_FILL, alpha=0.50, linewidth=0.0, zorder=-3
        )
        line, = ax.plot(T_GRID, h, color=_bus_color(bus), linewidth=1.60)
        boundary = ax.axhline(
            0.0, color=COLOR_ROSE, linestyle=(0, (3.0, 2.0)), linewidth=1.00, zorder=1
        )
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=4)
        _panel_bus_label(ax, bus, [h])
        _annotate_safety_boundary(ax)
        if handles is None:
            handles = [line]
            labels = [symbol]

    # Apply the exact barrier notation after the generic grid formatting.
    _finish_bus_grid(
        fig,
        axes,
        positions,
        nrows,
        "",
        handles,
        labels,
        3,
        ylabel_fontsize=FONT_SIZE_BARRIER_YLABEL,
    )
    for ax in axes:
        _set_barrier_ylabel(ax, side)
    save_figure(fig, f"gfl_barrier_function_{_case_tag(filter_enabled)}")


def plot_barrier_overlay(
    model: ReducedModel,
    baseline: dict,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(
        PLOT_BARRIER_OVERLAY_BUSES_EXT,
        model.GFL_ext,
        "PLOT_BARRIER_OVERLAY_BUSES_EXT",
    )
    if not buses:
        return

    side = _active_safety_side(model, baseline, active)
    symbol = _barrier_symbol(side)
    compare = bool(filter_enabled and RUN_NOMINAL_COMPARISON)

    if compare:
        fig, axes = plt.subplots(
            2, 1, figsize=(FIGURE_WIDTH_DOUBLE, 4.50), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_DOUBLE, 2.72))
        axes = np.atleast_1d(ax)

    handles, labels = [], []
    nominal_arrays, active_arrays = [], []

    for bus in buses:
        i = model.GFL_ext.index(bus)
        f_nom = F_BASE_HZ * baseline["omega_f"][:, i]
        f_act = F_BASE_HZ * active["omega_f"][:, i]
        h_nom = f_nom - FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ - f_nom
        h_act = f_act - FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ - f_act
        color = _bus_color(bus)

        if compare:
            line, = axes[0].plot(T_GRID, h_nom, color=color, linewidth=1.42)
            axes[1].plot(T_GRID, h_act, color=color, linewidth=1.42)
            nominal_arrays.append(h_nom)
        else:
            line, = axes[0].plot(T_GRID, h_act, color=color, linewidth=1.45)

        active_arrays.append(h_act)
        handles.append(line)
        labels.append(f"Bus {bus}")

    all_arrays = nominal_arrays + active_arrays if compare else active_arrays
    _set_data_ylim(
        axes[0], all_arrays, include_zero=True, lower_pad=0.12, upper_pad=0.18
    )
    ylo, _ = axes[0].get_ylim()

    for ax in axes:
        unsafe = ax.axhspan(
            ylo, 0.0, color=COLOR_UNSAFE_FILL, alpha=0.42, linewidth=0.0, zorder=-4
        )
        boundary = ax.axhline(
            0.0,
            color=COLOR_ROSE,
            linestyle=(0, (3.0, 2.0)),
            linewidth=0.95,
            zorder=1,
        )
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=5)
        _set_barrier_ylabel(ax, side)
        _annotate_safety_boundary(ax)

    if compare:
        _status_label(axes[0], False, nominal_arrays)
        _status_label(axes[1], True, active_arrays)
        axes[1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.105, right=0.988, bottom=0.105, top=0.895, hspace=0.065)
        name = "gfl_barrier_overlay_comparison"
    else:
        _status_label(axes[0], filter_enabled, active_arrays)
        axes[0].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
        fig.subplots_adjust(left=0.105, right=0.988, bottom=0.175, top=0.865)
        name = f"gfl_barrier_overlay_{_case_tag(filter_enabled)}"

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.994),
        ncol=min(5, len(labels)),
        frameon=False,
        borderaxespad=0.0,
    )
    save_figure(fig, name)


def _filter_state_plot_limits(
    model: ReducedModel,
    active: dict,
    buses: list[int],
) -> tuple[float, float] | None:
    traces = [_filter_state_trace(model, active, bus) for bus in buses]
    values = [np.asarray(z, dtype=float).reshape(-1) for z in traces]
    values = [z[np.isfinite(z)] for z in values if z.size]
    if not values:
        return None
    data = np.concatenate(values + [np.array([0.0])])
    lo = float(np.min(data))
    hi = float(np.max(data))
    span = hi - lo
    scale = max(abs(lo), abs(hi), 1.0)
    if span < max(1e-4, 1e-7 * scale):
        pad = max(1e-4, 0.05 * scale)
        lo -= pad
        hi += pad
        span = hi - lo
    return lo - 0.14 * span, hi + 0.24 * span


def plot_filter_state(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(
        PLOT_FILTER_BUSES_EXT,
        model.GFL_ext,
        "PLOT_FILTER_BUSES_EXT",
    )
    if not buses:
        return

    traces = {bus: _filter_state_trace(model, active, bus) for bus in buses}
    common_ylim = _filter_state_plot_limits(model, active, buses) if FILTER_STATE_COMMON_YLIM else None

    fig, axes, positions, nrows, _ = _make_bus_grid(len(buses), row_height=1.66)
    legend_handle = None

    for ax, bus in zip(axes, buses):
        z = traces[bus]
        line, = ax.plot(T_GRID, z, color=_bus_color(bus), linewidth=1.62)
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=4, zero_line=True)

        if common_ylim is None:
            _set_data_ylim(ax, [z], include_zero=True, lower_pad=0.14, upper_pad=0.24)
        else:
            ax.set_ylim(*common_ylim)

        _panel_bus_label(ax, bus, [z])
        if legend_handle is None:
            legend_handle = line

    _finish_bus_grid(
        fig,
        axes,
        positions,
        nrows,
        r"$z_i$ [p.u.]",
        [legend_handle],
        [r"$z_i=p_{{\rm s},i}$"],
        1,
        ylabel_fontsize=FONT_SIZE_FILTER_YLABEL,
    )
    save_figure(fig, f"filter_state_by_bus_{_case_tag(filter_enabled)}")


def plot_filter_state_overlay(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(
        PLOT_FILTER_OVERLAY_BUSES_EXT,
        model.GFL_ext,
        "PLOT_FILTER_OVERLAY_BUSES_EXT",
    )
    if not buses:
        return

    traces = {bus: _filter_state_trace(model, active, bus) for bus in buses}
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_DOUBLE, 2.75))

    handles, labels = [], []
    for bus in buses:
        z = traces[bus]
        line, = ax.plot(T_GRID, z, color=_bus_color(bus), linewidth=1.52)
        handles.append(line)
        labels.append(f"Bus {bus}")

    _event_lines(ax, events)
    _boxed_axis(ax, ybins=5, zero_line=True)

    if FILTER_STATE_COMMON_YLIM:
        common_ylim = _filter_state_plot_limits(model, active, buses)
        if common_ylim is not None:
            ax.set_ylim(*common_ylim)
    else:
        _set_data_ylim(
            ax,
            list(traces.values()),
            include_zero=True,
            lower_pad=0.14,
            upper_pad=0.24,
        )

    ax.set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5)
    ax.set_ylabel(r"$z_i$ [p.u.]", fontsize=FONT_SIZE_FILTER_YLABEL, labelpad=5.0)
    _status_label(ax, filter_enabled, list(traces.values()))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.994),
        ncol=min(5, len(labels)),
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONT_SIZE_LEGEND,
    )
    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.175, top=0.865)
    save_figure(fig, f"filter_state_overlay_{_case_tag(filter_enabled)}")



def plot_filter_tracking(model: ReducedModel, active: dict, events: list[dict]) -> None:
    if "target" not in active:
        return
    buses = _resolve_plot_buses(PLOT_FILTER_BUSES_EXT, model.GFL_ext, "PLOT_FILTER_BUSES_EXT")
    if not buses:
        return

    fig, axes, positions, nrows, _ = _make_bus_grid(len(buses), row_height=1.66)
    handles = labels = None
    for ax, bus in zip(axes, buses):
        i = _gfl_index(model, bus)
        z = _filter_state_trace(model, active, bus)
        target = active["target"][:, i]
        l0, = ax.plot(T_GRID, z, color=_bus_color(bus), linewidth=1.58)
        l1, = ax.plot(T_GRID, target, color=COLOR_PURPLE, linestyle="--", linewidth=1.02)
        fill = ax.fill_between(T_GRID, z, target, color=COLOR_TRACKING_FILL, alpha=0.25, linewidth=0.0, zorder=0)
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=4, zero_line=True)
        _set_data_ylim(ax, [z, target], include_zero=True, lower_pad=0.14, upper_pad=0.24)
        _panel_bus_label(ax, bus, [z, target])
        if handles is None:
            handles = [l0, l1, fill]
            labels = [r"$z_i$", r"$T_i(z)$", "Tracking gap"]

    _finish_bus_grid(fig, axes, positions, nrows, r"$z_i$ [p.u.]", handles, labels, 3, ylabel_fontsize=FONT_SIZE_FILTER_YLABEL)
    save_figure(fig, "filter_state_tracking")


def plot_gfl_active_power(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(PLOT_GFL_POWER_BUSES_EXT, model.GFL_ext, "PLOT_GFL_POWER_BUSES_EXT")
    if not buses:
        return

    ref_tol = 1e-10
    show_ref = bool(
        SHOW_GFL_REFERENCE_DEVIATION
        and any(
            np.max(np.abs(_gfl_trace(model, active, "delta_pF_ref", bus))) > ref_tol
            for bus in buses
        )
    )
    fig, axes, positions, nrows, _ = _make_bus_grid(len(buses), row_height=1.70)
    handles = labels = None

    for ax, bus in zip(axes, buses):
        i = _gfl_index(model, bus)
        dp_ref = active["delta_pF_ref"][:, i]
        dp_ffr = active["delta_pF_ffr"][:, i]
        dp_nom = active["delta_pF_nom"][:, i]
        dp_applied = active["delta_pF"][:, i]
        has_ref = bool(np.max(np.abs(dp_ref)) > ref_tol)

        base_curve = dp_nom if has_ref else dp_ffr
        base_label = r"$\Delta p_{F,i}^{\rm nom}$" if has_ref else r"$\Delta p_{F,i}^{\rm FFR}$"
        arrays = [base_curve, dp_applied]

        if filter_enabled:
            l_base, = ax.plot(
                T_GRID, base_curve, color=COLOR_BLUE, linestyle="--", linewidth=1.35
            )
            l_app, = ax.plot(
                T_GRID, dp_applied, color=_bus_color(bus), linewidth=1.65
            )
            correction = ax.fill_between(
                T_GRID,
                base_curve,
                dp_applied,
                color=COLOR_SUPPORT_FILL,
                alpha=0.50,
                linewidth=0.0,
                zorder=0,
            )
            local_handles = [l_base, l_app, correction]
            local_labels = [base_label, r"$\Delta p_{F,i}$", r"$z_i$"]
        else:
            l_app, = ax.plot(
                T_GRID, dp_applied, color=_bus_color(bus), linewidth=1.65
            )
            local_handles = [l_app]
            local_labels = [r"$\Delta p_{F,i}$"]

        if show_ref and has_ref:
            l_ref, = ax.plot(
                T_GRID, dp_ref, color=COLOR_ORANGE, linestyle=":", linewidth=1.10
            )
            local_handles.insert(0, l_ref)
            local_labels.insert(0, r"$\Delta p_{F,i}^{\rm ref}$")
            arrays.append(dp_ref)

        _event_lines(ax, events)
        _boxed_axis(ax, ybins=4, zero_line=True)
        _set_data_ylim(ax, arrays, include_zero=True, lower_pad=0.13, upper_pad=0.24)
        _panel_bus_label(ax, bus, arrays)
        if handles is None:
            handles, labels = local_handles, local_labels

    _finish_bus_grid(
        fig,
        axes,
        positions,
        nrows,
        r"$\Delta p_{F,i}\,[\mathrm{p.u.}]$",
        handles,
        labels,
        len(labels),
    )
    save_figure(fig, f"gfl_power_deviation_{_case_tag(filter_enabled)}")


def plot_gfl_power_decomposition(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    dp_ref = np.sum(active["delta_pF_ref"], axis=1)
    dp_ffr = np.sum(active["delta_pF_ffr"], axis=1)
    sum_z = np.sum(active["z"], axis=1)
    dp_nom = np.sum(active["delta_pF_nom"], axis=1)
    dp_applied = np.sum(active["delta_pF"], axis=1)

    ref_active = bool(np.max(np.abs(dp_ref)) > 1e-10)
    base_curve = dp_nom if ref_active else dp_ffr
    base_label = r"$\Delta P_F^{\rm nom}$" if ref_active else r"$\Delta P_F^{\rm FFR}$"

    fig, axes = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTH_DOUBLE, 4.65), sharex=True
    )

    if filter_enabled:
        p_base, = axes[0].plot(
            T_GRID, base_curve, color=COLOR_BLUE, linestyle="--", linewidth=1.40
        )
        p_applied, = axes[0].plot(
            T_GRID, dp_applied, color=COLOR_TEAL, linewidth=1.70
        )
        p_z = axes[0].fill_between(
            T_GRID,
            base_curve,
            dp_applied,
            color=COLOR_SUPPORT_FILL,
            alpha=0.50,
            linewidth=0.0,
            zorder=0,
        )
        axes[0].legend(
            [p_base, p_applied, p_z],
            [base_label, r"$\Delta P_F$", r"$\sum_i z_i$"],
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND,
        )
    else:
        p_applied, = axes[0].plot(
            T_GRID, dp_applied, color=COLOR_TEAL, linewidth=1.70
        )
        axes[0].legend(
            [p_applied], [r"$\Delta P_F$"], loc="upper right", fontsize=FONT_SIZE_LEGEND
        )

    p_ffr, = axes[1].plot(
        T_GRID, dp_ffr, color=COLOR_SKY, linewidth=1.65
    )
    p_filter, = axes[1].plot(
        T_GRID, sum_z, color=COLOR_VERMILLION, linewidth=1.65
    )
    axes[1].legend(
        [p_ffr, p_filter],
        [r"$\Delta P_F^{\rm FFR}$", r"$\sum_i z_i$"],
        loc="upper right",
        fontsize=FONT_SIZE_LEGEND,
    )

    for ax in axes:
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=5, zero_line=True)
        ax.set_ylabel(r"$\Delta P_F\,[\mathrm{p.u.}]$", fontsize=FONT_SIZE_YLABEL, labelpad=5.0)

    _set_data_ylim(
        axes[0], [base_curve, dp_applied], include_zero=True, upper_pad=0.23
    )
    _set_data_ylim(
        axes[1], [dp_ffr, sum_z], include_zero=True, upper_pad=0.23
    )
    axes[1].set_xlabel(
        r"$t\,[\mathrm{s}]$", fontsize=FONT_SIZE_XLABEL, labelpad=4.5
    )
    fig.subplots_adjust(
        left=0.105, right=0.988, bottom=0.105, top=0.985, hspace=0.075
    )
    save_figure(fig, f"gfl_power_decomposition_{_case_tag(filter_enabled)}")


def plot_sg_mechanical_power(
    model: ReducedModel,
    active: dict,
    events: list[dict],
    filter_enabled: bool,
) -> None:
    buses = _resolve_plot_buses(PLOT_SG_POWER_BUSES_EXT, model.SG_ext, "PLOT_SG_POWER_BUSES_EXT")
    if not buses:
        return

    fig, axes, positions, nrows, _ = _make_bus_grid(len(buses), row_height=1.68)
    legend_handle = None
    for ax, bus in zip(axes, buses):
        i = model.SG_ext.index(bus)
        dp_m = active["delta_pm"][:, i]
        line, = ax.plot(T_GRID, dp_m, color=_bus_color(bus), linewidth=1.60)
        _event_lines(ax, events)
        _boxed_axis(ax, ybins=4, zero_line=True)
        _set_data_ylim(ax, [dp_m], include_zero=True, lower_pad=0.13, upper_pad=0.24)
        _panel_bus_label(ax, bus, [dp_m])
        if legend_handle is None:
            legend_handle = line

    _finish_bus_grid(
        fig,
        axes,
        positions,
        nrows,
        r"$\Delta p_{m,i}\,[\mathrm{p.u.}]$",
        [legend_handle],
        [r"$\Delta p_{m,i}$"],
        1,
    )
    save_figure(fig, f"sg_mechanical_power_deviation_{_case_tag(filter_enabled)}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(CSV_DIR)
    configure_matplotlib()

    mpc_ext = load_case_from_py()
    (
        mpc_base,
        pf_results,
        Ybus,
        id_map,
        inv_map,
        slack_int,
        base_mva,
    ) = run_pf_init(mpc_ext)

    if EXPORT_PF_CSV:
        export_power_flow_csvs(pf_results, inv_map)

    model = build_reduced_model(
        mpc_base=mpc_base,
        Ybus=Ybus,
        id_map=id_map,
        inv_map=inv_map,
        slack_int=slack_int,
        base_mva=base_mva,
    )
    export_shared_cbf_design(model)

    events = build_events(model, id_map)
    print("\nDisturbances")
    if events:
        for event in events:
            print(f"  t={event['time_s']:.3f} s: {event['label']}")
    else:
        print("  none")

    print(f"\n{_status_text(ENABLE_SAFETY_FILTER)}")

    t0 = time.perf_counter()
    baseline = simulate_without_safety(model, events)
    active = simulate_with_dynamic_filter(model, events) if ENABLE_SAFETY_FILTER else baseline
    elapsed = time.perf_counter() - t0

    validate_filter_state_trajectory(model, active, ENABLE_SAFETY_FILTER)
    print_performance_summary(model, baseline, active, ENABLE_SAFETY_FILTER)
    export_results(model, baseline, active, ENABLE_SAFETY_FILTER)
    write_simulation_report(
        model,
        baseline,
        active,
        events,
        ENABLE_SAFETY_FILTER,
        inv_map,
    )

    if MAKE_FREQUENCY_FIGURE:
        plot_frequency_response(model, baseline, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_POWER_OVERLAY_FIGURE:
        plot_power_overlay(model, baseline, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_RESOURCE_FREQUENCY_OVERLAY_FIGURE:
        plot_resource_frequency_overlay(
            model, baseline, active, events, ENABLE_SAFETY_FILTER
        )
    if MAKE_RESOURCE_POWER_OVERLAY_FIGURE:
        plot_resource_power_overlay(
            model, baseline, active, events, ENABLE_SAFETY_FILTER
        )
    if MAKE_CBF_CONSTRAINT_FIGURE:
        plot_cbf_constraints(model, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_BARRIER_FIGURE:
        plot_barrier_functions(model, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_BARRIER_OVERLAY_FIGURE:
        plot_barrier_overlay(model, baseline, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_FILTER_STATE_FIGURE:
        plot_filter_state(model, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_FILTER_STATE_OVERLAY_FIGURE:
        plot_filter_state_overlay(model, active, events, ENABLE_SAFETY_FILTER)
    if ENABLE_SAFETY_FILTER and MAKE_FILTER_TRACKING_FIGURE:
        plot_filter_tracking(model, active, events)
    if MAKE_GFL_POWER_FIGURE:
        plot_gfl_active_power(model, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_GFL_AGGREGATE_FIGURE:
        plot_gfl_power_decomposition(model, active, events, ENABLE_SAFETY_FILTER)
    if MAKE_SG_POWER_FIGURE:
        plot_sg_mechanical_power(model, active, events, ENABLE_SAFETY_FILTER)

    print(f"\nSimulation completed in {elapsed:.3f} s.")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

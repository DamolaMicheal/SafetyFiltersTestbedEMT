#!/usr/bin/env python3
import time
import warnings
import os
import csv
import shutil
import tempfile
import logging
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("fontTools").setLevel(logging.ERROR)

SYS_BASE_MVA = 100.0
F_BASE       = 60.0
OMEGA_BASE   = 2.0 * np.pi * F_BASE

dt        = 0.01
Tfinal    = 6
tgrid     = np.arange(0.0, Tfinal + dt, dt)
Tsteps    = len(tgrid)

SIMULATION_MODE = "DC"

LOAD_CHANGE_ENABLE = True
LOAD_CHANGE_EVENTS = [
    {
        "buses":        [35],
        "event_time_s": 0.25,
        "pu_map":      {35: 300.0 / SYS_BASE_MVA},
    },
    {
        "buses":        [2],
        "event_time_s": 0.25,
        "pu_map":      {2: 150.0 / SYS_BASE_MVA},
    },
    {
        "buses":        [29],
        "event_time_s": 0.25,
        "pu_map":      {29: 75.0 / SYS_BASE_MVA},
    },
]

#Load trip
LOAD_TRIP_ENABLE            = False
LOAD_TRIP_EVENT_TIME_S      = 0.7
LOAD_TRIP_BUSES_EXT         = [20]
LOAD_TRIP_SCALE_FIXED_EXT   = {20: 1.0}

# Generator trip
GEN_TRIP_ENABLE             = False
GEN_TRIP_EVENT_TIME_S       = 0.25
GEN_TRIP_BUSES_EXT          = [33]
GEN_TRIP_SCALE_FIXED_EXT    = {33: 1.0}

PLOT_BUSES_EXT = [30, 31, 34,36,38, 32, 33, 35, 37, 39]

SLACK_BUS_EXT  = 39

PLOT_PANELS = ("freq", "pg", "angle")

CSV_EXPORT_DIR = os.path.join(os.getcwd(), "csv_exports_normal")
EXPORT_TRAJ_CSV = True
EXPORT_PF_CSV   = True

USER_SG_BUSES_EXT  = [32, 33, 35, 37, 39]
USER_GFM_BUSES_EXT =[]
USER_GFL_BUSES_EXT =  [30, 34, 36, 31, 38]


K_FFR_GFL_DEFAULT = 20.0
K_FFR_GFL_MAP_EXT = {
    30: 20.0,
}

ENABLE_SAFETY_FILTER = True
RUN_NOMINAL_COMPARISON = True

FREQ_LOWER_HZ = -0.07
FREQ_UPPER_HZ = +0.07
ALPHA_CBF = 5.0
FILTER_EPSILON_S = 0.01
ROCOF_MODE = "dirty"
ROCOF_TAU_S = 0.0085
PROX_STEP_FRACTION = 0.4
PROX_K_MAP_EXT = {}
ACTIVE_SAFETY_BOUND = "auto"

FIGURE_EXPORT_DIR = os.path.join(os.getcwd(), "publication_figures_dynamic_filter")
SHOW_PLOTS = True
SAVE_PDF = True
SAVE_PNG = True
USE_EXTERNAL_LATEX = False
MAKE_FREQUENCY_SAFETY_FIGURE = False
MAKE_BARRIER_FIGURE = False
MAKE_FILTER_STATE_FIGURE = False
MAKE_GFL_POWER_DECOMPOSITION_FIGURE = False

# dq0-EMT validation overlays. The analytical trajectories are used directly
# from this run; only the Julia EMT trajectories are read from CSV.
MAKE_EMT_FREQUENCY_OVERLAY = False
MAKE_EMT_FILTER_STATE_OVERLAY = False
MAKE_EMT_GFL_POWER_OVERLAY = False
MAKE_EMT_FREQUENCY_ERROR_FIGURE = False
MAKE_EMT_SUMMARY_PANEL = True

# All resources are available; use a compact subset for the main validation
# overlay to keep the figure readable. Set to PLOT_BUSES_EXT for all 10 buses.
EMT_OVERLAY_BUSES_EXT = [30, 31, 34, 36, 38, 39]

# Optional explicit root. Leave None for automatic discovery.
# It must point to .../plots/python_style_cbf/csv.
EMT_CSV_ROOT_OVERRIDE = None

# EMT line style in validation plots.
EMT_LINESTYLE = (0, (5.5, 2.8))
ANALYTICAL_LINESTYLE = "-"
ANALYTICAL_OVERLAY_LW = 1.55
EMT_OVERLAY_LW = 1.45

# MATPOWER column indices
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


def ensure_dir(path):
    if path is None or str(path).strip() == "":
        return
    os.makedirs(path, exist_ok=True)

def enforce_laplacian(B):
    B = 0.5 * (B + B.T)
    np.fill_diagonal(B, 0.0)
    np.fill_diagonal(B, -np.sum(B, axis=1))
    return B

def elimination_mats(N, slack_idx0):
    others = [k for k in range(N) if k != slack_idx0]
    I_minus = np.zeros((N - 1, N))
    for r, i in enumerate(others):
        I_minus[r, i] = 1.0
        I_minus[r, slack_idx0] = -1.0
    I0 = np.zeros((N, N - 1))
    for c, i in enumerate(others):
        I0[i, c] = 1.0
    return I_minus, I0

def selector(N, idxs_1based):
    Z = np.zeros((N, len(idxs_1based)))
    for c, i0 in enumerate([j - 1 for j in idxs_1based]):
        Z[i0, c] = 1.0
    return Z

def row_picker(mask01):
    idx = [i for i, v in enumerate(mask01) if int(v) == 1]
    P = np.zeros((len(idx), len(mask01)))
    for r, i in enumerate(idx):
        P[r, i] = 1.0
    return P

def df_hz_from_omega_dev_pu(omega_dev_pu):
    return float(omega_dev_pu) * F_BASE

def event_time_to_idx(t_event):
    return int(np.clip(np.floor(t_event / dt + 1e-12), 0, Tsteps - 1))
def write_freq_traj_csv(path, t, df):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t_s", "df_hz"])
        for i in range(len(t)):
            w.writerow([float(t[i]), float(df[i])])

def write_angle_traj_csv(path, t, th):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t_s", "th_rad"])
        for i in range(len(t)):
            w.writerow([float(t[i]), float(th[i])])

def write_pg_traj_csv(path, t, p):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t_s", "pG_pu"])
        for i in range(len(t)):
            w.writerow([float(t[i]), float(p[i])])

def write_dpf_traj_csv(path, t, p):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t_s", "dpF_pu"])
        for i in range(len(t)):
            w.writerow([float(t[i]), float(p[i])])

def write_matrix_csv(path, header, M):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(list(header))
        for r in range(M.shape[0]):
            w.writerow([float(x) for x in M[r, :]])

def default_headers(prefix, ncols):
    return [f"{prefix}{i}" for i in range(ncols)]

def build_partitions(user_sg_ext, user_gfm_ext, user_gfl_ext, slack_ext=39):
    valid_30_38 = set(range(30, 39))
    allowed_all = set(range(30, 40))

    sg  = sorted(set(int(b) for b in user_sg_ext  if int(b) in allowed_all))
    gfm = sorted(set(int(b) for b in user_gfm_ext if int(b) in valid_30_38))
    gfl = sorted(set(int(b) for b in user_gfl_ext if int(b) in valid_30_38))

    if int(slack_ext) not in sg:
        sg = sorted(set(sg + [int(slack_ext)]))

    for grp, name in [(gfm, "GFM"), (gfl, "GFL")]:
        if int(slack_ext) in grp:
            print(f"[warn] Slack bus {slack_ext} cannot be {name}; removing.")
            grp = sorted(set(grp) - {int(slack_ext)})

    for ov, src, dst, dname in [
        (sorted(set(sg) & set(gfm)), sg, gfm, "GFM"),
        (sorted(set(sg) & set(gfl)), sg, gfl, "GFL"),
        (sorted(set(gfm) & set(gfl)), gfm, gfl, "GFL (priority GFM)"),
    ]:
        if ov:
            print(f"[warn] Overlap SG/{dname}; removing from {dname}: {ov}")
            dst[:] = sorted(set(dst) - set(ov))

    remainder = sorted(valid_30_38 - ((set(sg) | set(gfm) | set(gfl)) & valid_30_38))
    if remainder:
        print(f"[warn] Unassigned buses 30..38; assigning to GFM: {remainder}")
        gfm = sorted(set(gfm) | set(remainder))

    cover = set(sg) | set(gfm) | set(gfl)
    if cover != allowed_all:
        raise ValueError(f"Partition must cover 30..39. Missing={sorted(allowed_all-cover)}, "
                         f"Extra={sorted(cover-allowed_all)}")
    if set(sg) & set(gfm):
        raise ValueError(f"SG∩GFM: {sorted(set(sg)&set(gfm))}")
    if set(sg) & set(gfl):
        raise ValueError(f"SG∩GFL: {sorted(set(sg)&set(gfl))}")
    if set(gfm) & set(gfl):
        raise ValueError(f"GFM∩GFL: {sorted(set(gfm)&set(gfl))}")

    return sg, gfm, gfl

def deepcopy_case(mpc):
    return {
        k: (np.array(v, dtype=float, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in mpc.items()
    }

def sanitize_case_for_runpf(mpc):
    mpc_s = deepcopy_case(mpc)
    bus = np.asarray(mpc_s["bus"], dtype=float)
    gen = np.asarray(mpc_s["gen"], dtype=float)
    br  = np.asarray(mpc_s["branch"], dtype=float)

    bus_ids = bus[:, BUS_I].astype(int)
    if np.unique(bus_ids).size != bus.shape[0]:
        raise ValueError("Bus IDs not unique.")

    order = np.argsort(bus_ids)
    bus   = bus[order, :]
    bus_ids_sorted = bus[:, BUS_I].astype(int)
    N = bus.shape[0]

    id_map = {int(old): int(new) for new, old in enumerate(bus_ids_sorted, start=1)}

    bus[:, BUS_I]   = np.array([id_map[int(x)] for x in bus_ids_sorted], dtype=float)
    gen[:, GEN_BUS] = np.array([id_map[int(x)] for x in gen[:, GEN_BUS].astype(int)], dtype=float)
    br[:, F_BUS]    = np.array([id_map[int(x)] for x in br[:, F_BUS].astype(int)], dtype=float)
    br[:, T_BUS]    = np.array([id_map[int(x)] for x in br[:, T_BUS].astype(int)], dtype=float)

    if not np.array_equal(bus[:, BUS_I].astype(int), np.arange(1, N + 1, dtype=int)):
        raise ValueError("Sanitization failed.")

    mpc_s["bus"] = bus
    mpc_s["gen"] = gen
    mpc_s["branch"] = br
    return mpc_s, id_map

def build_ybus_dense_matpower(baseMVA, bus, branch):
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)
    nb = bus.shape[0]

    busnums = bus[:, BUS_I].astype(int)
    bus_to_idx = {int(bn): i for i, bn in enumerate(busnums)}
    if len(bus_to_idx) != nb:
        raise ValueError("Duplicate bus numbers.")

    Y = np.zeros((nb, nb), dtype=complex)
    gs = bus[:, GS] / float(baseMVA)
    bs = bus[:, BS] / float(baseMVA)
    Y[np.arange(nb), np.arange(nb)] += (gs + 1j * bs)

    for row in branch:
        if int(row[BR_STATUS]) == 0:
            continue
        fb = int(row[F_BUS])
        tb = int(row[T_BUS])
        if fb not in bus_to_idx or tb not in bus_to_idx:
            raise ValueError(f"Branch references missing bus ({fb}->{tb}).")
        i = bus_to_idx[fb]
        j = bus_to_idx[tb]
        z = float(row[BR_R]) + 1j * float(row[BR_X])
        if abs(z) < 1e-16:
            raise ValueError(f"Zero impedance ({fb}->{tb}).")
        y   = 1.0 / z
        ysh = 1j * float(row[BR_B]) / 2.0
        tap = float(row[TAP]) or 1.0
        t   = tap * np.exp(1j * np.deg2rad(float(row[SHIFT])))
        Y[i, i] += (y + ysh) / (t * np.conj(t))
        Y[i, j] += -y / np.conj(t)
        Y[j, i] += -y / t
        Y[j, j] += (y + ysh)

    return Y

def ac_injections_PQ(Ybus_dense, Vmag, theta_rad):
    V = np.asarray(Vmag).reshape(-1) * np.exp(1j * np.asarray(theta_rad).reshape(-1))
    S = V * np.conj(Ybus_dense.dot(V))
    return np.real(S).reshape(-1, 1), np.imag(S).reshape(-1, 1)

def export_pf_solution_csvs(export_dir, pf_results, inv_map):
    ensure_dir(export_dir)
    bus = np.asarray(pf_results["bus"], dtype=float).copy()
    gen = np.asarray(pf_results["gen"], dtype=float).copy()
    br  = np.asarray(pf_results["branch"], dtype=float).copy()

    bus[:, BUS_I]   = np.array([float(inv_map.get(int(x), int(x))) for x in bus[:, BUS_I]], dtype=float)
    gen[:, GEN_BUS] = np.array([float(inv_map.get(int(x), int(x))) for x in gen[:, GEN_BUS]], dtype=float)
    br[:, F_BUS]    = np.array([float(inv_map.get(int(x), int(x))) for x in br[:, F_BUS]], dtype=float)
    br[:, T_BUS]    = np.array([float(inv_map.get(int(x), int(x))) for x in br[:, T_BUS]], dtype=float)

    bus = bus[np.argsort(bus[:, BUS_I].astype(int)), :]

    bus_hdr = ["BUS_I","BUS_TYPE","PD","QD","GS","BS","BUS_AREA","VM","VA","BASE_KV","ZONE","VMAX","VMIN"]
    gen_hdr = ["GEN_BUS","PG","QG","QMAX","QMIN","VG","MBASE","GEN_STATUS","PMAX","PMIN"]
    br_hdr  = ["F_BUS","T_BUS","BR_R","BR_X","BR_B","RATE_A","RATE_B","RATE_C","TAP","SHIFT","BR_STATUS","ANGMIN","ANGMAX"]

    if bus.shape[1] > len(bus_hdr):
        bus_hdr += default_headers("BUS_COL_", bus.shape[1] - len(bus_hdr))
    if gen.shape[1] > len(gen_hdr):
        gen_hdr += default_headers("GEN_COL_", gen.shape[1] - len(gen_hdr))
    if br.shape[1] > len(br_hdr):
        br_hdr += default_headers("BR_COL_", br.shape[1] - len(br_hdr))

    write_matrix_csv(os.path.join(export_dir, "pf_bus_solution.csv"), bus_hdr, bus)
    write_matrix_csv(os.path.join(export_dir, "pf_gen_solution.csv"), gen_hdr, gen)
    write_matrix_csv(os.path.join(export_dir, "pf_branch_solution.csv"), br_hdr, br)
    for n in ["pf_bus_solution.csv", "pf_gen_solution.csv", "pf_branch_solution.csv"]:
        print(f"[CSV] Wrote PF solution: {os.path.join(export_dir, n)}")

def load_case_from_py():
    import importlib
    mod = importlib.import_module("case39_modified_slack39")
    if hasattr(mod, "get_mpc"):
        return mod.get_mpc()
    if hasattr(mod, "load_case"):
        return mod.load_case()
    if hasattr(mod, "mpc"):
        return mod.mpc
    raise RuntimeError("case39_modified_slack39.py must export get_mpc(), load_case(), or mpc.")

def run_pf_init(mpc_ext):
    mpc_base, id_map = sanitize_case_for_runpf(mpc_ext)
    inv_map = {v: k for k, v in id_map.items()}
    baseMVA = float(mpc_base.get("baseMVA", SYS_BASE_MVA))
    bus    = np.asarray(mpc_base["bus"],    dtype=float)
    branch = np.asarray(mpc_base["branch"], dtype=float)
    N = bus.shape[0]

    if SLACK_BUS_EXT not in id_map:
        raise ValueError(f"Slack bus {SLACK_BUS_EXT} not in case.")

    slack_bus  = id_map[int(SLACK_BUS_EXT)]
    slack_idx0 = slack_bus - 1

    from pypower.api import runpf, ppoption
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0, PF_ALG=1, PF_MAX_IT=80, PF_TOL=1e-10)
    mpc_pf = deepcopy_case(mpc_base)

    t0 = time.perf_counter()
    pf_results, success = runpf(mpc_pf, ppopt)
    dt_pf = time.perf_counter() - t0
    if not success:
        raise RuntimeError("Power flow did not converge.")

    bus_pf = np.asarray(pf_results["bus"], dtype=float)
    bus_pf = bus_pf[np.argsort(bus_pf[:, BUS_I].astype(int)), :]

    Vmag_ref  = bus_pf[:, VM].astype(float).copy()
    theta_ref = (bus_pf[:, VA].astype(float) * np.pi / 180.0).copy()
    theta_ref -= float(theta_ref[slack_idx0])

    Ybus_dense = build_ybus_dense_matpower(baseMVA, bus, branch)
    P_ref, Q_ref = ac_injections_PQ(Ybus_dense, Vmag_ref, theta_ref)

    if abs(baseMVA - SYS_BASE_MVA) > 1e-12:
        P_ref *= baseMVA / SYS_BASE_MVA
        Q_ref *= baseMVA / SYS_BASE_MVA

    u0_base = -P_ref.copy()

    print(f"\n=== PF Init: {dt_pf:.3f}s, baseMVA={baseMVA:.1f}, N={N}, "
          f"slack_int={slack_bus} (ext={inv_map.get(slack_bus, slack_bus)}) ===")

    return (mpc_base, id_map, inv_map, N, slack_bus,
            Vmag_ref, theta_ref, Ybus_dense, P_ref, Q_ref, u0_base, pf_results)

def compute_post_trip_dc_matrices(dc_model, tripped_G_positions):
    A_red   = dc_model["A_red_cont"].copy()
    B_uni   = dc_model["B_unified_cont"].copy()
    Umap    = dc_model["Umap"]
    nθ_GI   = dc_model["nθ_GI"]
    nGI     = dc_model["nGI"]
    nx      = A_red.shape[0]
    m_inter = Umap.shape[1]
    for j in tripped_G_positions:
        p_idx = nθ_GI + nGI + j
        A_red[p_idx, :]     = 0.0
        A_red[p_idx, p_idx] = -1.0
        A_red[nθ_GI:nθ_GI+nGI, p_idx] = 0.0
        B_uni[p_idx, :] = 0.0
    B_red_trip = B_uni @ Umap
    Aug = np.block([[A_red, B_red_trip],
                    [np.zeros((m_inter, nx)), np.zeros((m_inter, m_inter))]])
    Phi     = sl.expm(Aug * dt)
    Ad_trip = Phi[:nx, :nx]
    Bd_trip = Phi[:nx, nx:nx+m_inter]
    return Ad_trip, Bd_trip

def build_frequency_divider_H(B_laplacian, N, slack_idx0, GI_buses, LF_buses, F_buses):
    """
    Returns (Hpll_raw, F_order_for_H) where Hpll_raw columns are in
    sorted(GI_buses) order.  The caller must reorder columns to match
    GI_bus_order (= G_buses + I_buses) before use.
    """
    if len(F_buses) == 0:
        return np.zeros((0, len(GI_buses))), []

    non_slack = [k for k in range(N) if k != slack_idx0]
    B_ground  = B_laplacian[np.ix_(non_slack, non_slack)]
    I_minus, _ = elimination_mats(N, slack_idx0)

    GI_sorted = sorted([int(b) for b in GI_buses])
    F_sorted  = sorted([int(b) for b in F_buses])
    LF_sorted = sorted([int(b) for b in LF_buses if int(b) not in set(F_sorted)])

    S_GI = selector(N, GI_sorted)
    S_F  = selector(N, F_sorted)
    S_L  = selector(N, LF_sorted)
    S_LF = np.concatenate([S_F, S_L], axis=1)

    nF  = S_F.shape[1]
    nLF = S_LF.shape[1]
    if nLF == 0:
        return np.zeros((nF, len(GI_sorted))), F_sorted

    Sbar_GI = I_minus @ S_GI
    Sbar_LF = I_minus @ S_LF
    A_LF    = Sbar_LF.T @ B_ground @ Sbar_LF
    B_LF_GI = Sbar_LF.T @ B_ground @ Sbar_GI
    C_F     = np.hstack([np.eye(nF), np.zeros((nF, nLF - nF))])

    try:
        X = sl.solve(A_LF, B_LF_GI, assume_a="gen")
    except Exception:
        X = sl.lstsq(A_LF, B_LF_GI)[0]

    return -C_F @ X, F_sorted

def print_gfl_law_summary(dc_model, inv_map):
    F_buses = dc_model["GFL_buses"]
    if len(F_buses) == 0:
        print("\n[GFL] No GFL buses in this partition.")
        return
    pF_ref_vec = dc_model["pF_ref_vec"]
    Kf_vec     = dc_model["Kf_vec"]
    print("\n======GFL ACTIVE-POWER LAW ================")
    print("   0 = -S_L^T B̄θ - p_{ℓ,L}           (pure load buses)")
    print("    0 = p_F^r - K_F ω_F - S_F^T B̄θ - p_{ℓ,F}  (GFL buses)")
    print("    ω_F = H_I ω_I + H_G ω_G             (frequency divider)")
    print("  Δp_F = p_F - p_F_ref = -K_F * ω_F")
    print("  p_F_ref from PF operating point.")
    for i, b_int in enumerate(F_buses):
        b_ext = inv_map.get(int(b_int), int(b_int))
        p_ref_pu = float(pF_ref_vec[i])
        kf = float(Kf_vec[i])
        print(f"  Bus {b_ext:>2}: p_F_ref={p_ref_pu:+.6f} pu "
              f"({p_ref_pu*SYS_BASE_MVA:+.3f} MW), K_F={kf:.6f}")
    print("===============================================================\n")

def build_dc_model_with_internals(N, slack_bus, Ybus_dense, mpc_base, id_map, inv_map,
                                  SG_ext, GFM_ext, GFL_ext, P_ref):
    G_buses = [id_map[b] for b in SG_ext  if b in id_map]
    I_buses = [id_map[b] for b in GFM_ext if b in id_map]
    F_buses = [id_map[b] for b in GFL_ext if b in id_map]
    L_buses = sorted(list(set(range(1, N+1)) - set(G_buses) - set(I_buses)))
    slack_idx0 = slack_bus - 1

    B = enforce_laplacian(-np.imag(Ybus_dense).astype(float))
    I_minus, I0 = elimination_mats(N, slack_idx0)

    S_G  = selector(N, G_buses)
    S_I  = selector(N, I_buses)
    S_F  = selector(N, F_buses)
    S_L  = selector(N, L_buses)
    S_GI = np.concatenate([S_G, S_I], axis=1)
    PiL  = S_L @ S_L.T

    nG = S_G.shape[1]
    nI = S_I.shape[1]
    nGI = nG + nI

    non_slack = [k for k in range(N) if k != slack_idx0]
    mask_gi_rows = np.zeros(N-1, dtype=int)
    thetaGI_bus_order = []
    thetaA_bus_order = []

    for r, i0 in enumerate(non_slack):
        bus_id = i0 + 1
        if bus_id in G_buses or bus_id in I_buses:
            mask_gi_rows[r] = 1
            thetaGI_bus_order.append(bus_id)
        else:
            thetaA_bus_order.append(bus_id)

    mask_l_rows = 1 - mask_gi_rows
    P_L  = row_picker(mask_l_rows)
    P_GI = row_picker(mask_gi_rows)
    nθ_GI = int(mask_gi_rows.sum())
    nθ_A  = int(mask_l_rows.sum())

    gen = np.asarray(mpc_base["gen"], dtype=float)
    mbase_map = {int(r[GEN_BUS]): float(r[MBASE_COL]) for r in gen if int(r[GEN_STATUS])==1}

    def S_pu(bus_id):
        return float(mbase_map.get(bus_id, SYS_BASE_MVA)) / SYS_BASE_MVA

    H_map_ext = {30:4.200,31:3.030,32:3.580,33:2.860,34:4.333,
                 35:3.580,36:3.480,37:2.430,38:3.450,39:5.000}
    D_scale = 1.0
    D_map_ext = {30:40.0/D_scale,32:40.0/D_scale,33:40.0/D_scale,
                 36:40.0/D_scale,38:40.0/D_scale,39:48.0/D_scale}

    R_SG_PU  = 0.05
    invR_SG  = 1.0 / R_SG_PU
    R_GFM_PU = 0.08
    invR_GFM = 1.0 / R_GFM_PU

    M_bus = np.zeros(N, dtype=float)
    D_bus = np.zeros(N, dtype=float)
    for b in G_buses:
        Spu = S_pu(b)
        ext = int(inv_map.get(b, b))
        M_bus[b-1] = 2.0 * float(H_map_ext.get(ext, 4.0)) * Spu
        D_bus[b-1] = float(D_map_ext.get(ext, 40.0)) * Spu
    for b in I_buses:
        Spu = S_pu(b)
        ext = int(inv_map.get(b, b))
        M_bus[b-1] = 2.0 * float(H_map_ext.get(ext, 4.0)) * Spu
        D_bus[b-1] = invR_GFM * Spu

    GI_bus_order = G_buses + I_buses
    GI_order0    = [b-1 for b in GI_bus_order]
    M_GI = np.diag([M_bus[i0] for i0 in GI_order0])
    D_GI = np.diag([D_bus[i0] for i0 in GI_order0])
    if np.min(np.diag(M_GI)) <= 0:
        raise ValueError("M_GI non-positive diagonal.")
    Minv_GI = sl.inv(M_GI)

    tauG    = 0.01 * np.ones(nG, dtype=float)
    GammaG  = np.diag(tauG)
    GamGinv = sl.inv(GammaG)

    S_pu_G_vec = np.array([S_pu(b) for b in G_buses], dtype=float)
    Rmat = -np.diag(invR_SG * S_pu_G_vec)

    Ltheta_full = I_minus @ PiL @ (-B) @ I0
    AθGθG = P_GI @ Ltheta_full @ P_GI.T
    AθGθA = P_GI @ Ltheta_full @ P_L.T
    AθAθG = P_L  @ Ltheta_full @ P_GI.T
    AθAθA = P_L  @ Ltheta_full @ P_L.T

    # Angle dynamics: θ̇_GI = ω_base ω_GI.
    AθGw = OMEGA_BASE * (P_GI @ (I_minus @ S_GI))
    AθGp = np.zeros((nθ_GI, nG))

    # SG/GFM frequency dynamics:
    AwθG = -Minv_GI @ (S_GI.T @ B @ I0 @ P_GI.T)
    AwθA = -Minv_GI @ (S_GI.T @ B @ I0 @ P_L.T)
    Aw_w = -Minv_GI @ D_GI
    Aw_p =  Minv_GI @ (S_GI.T @ S_G)

    ApθG = np.zeros((nG, nθ_GI))
    ApθA = np.zeros((nG, nθ_A))
    Ap_w = GamGinv @ Rmat @ (S_G.T @ S_GI)
    Ap_p = -GamGinv

    m_inter = nI + nG + N
    Umap = np.block([
        [S_I,                    S_G,               np.zeros((N, N))],
        [np.zeros((N, nI + nG)), np.eye(N)]
    ])

    Btheta_full = np.hstack([np.zeros((N-1, N)), I_minus @ PiL @ (-np.eye(N))])
    BθG = P_GI @ Btheta_full
    BθA = P_L  @ Btheta_full
    Bw  = np.hstack([Minv_GI @ (S_GI.T @ (np.eye(N) - S_G @ S_G.T)), -Minv_GI @ (S_GI.T)])
    Bp  = np.hstack([GamGinv @ (S_G.T), np.zeros((nG, N))])


    def Kffr_pu(bus_id):
        ext = int(inv_map.get(bus_id, bus_id))
        return float(K_FFR_GFL_MAP_EXT.get(ext, K_FFR_GFL_DEFAULT)) * S_pu(bus_id)

    Kf_vec  = np.array([Kffr_pu(b) for b in F_buses], dtype=float)
    Kf_diag = np.diag(Kf_vec) if len(F_buses) > 0 else np.zeros((0, 0))
    pF_ref_vec = np.array([float(P_ref[b-1, 0]) for b in F_buses], dtype=float)

    LF_buses = sorted(list(set(L_buses)))
    Hpll_raw, F_order_for_H = build_frequency_divider_H(
        B, N, slack_idx0, GI_bus_order, LF_buses, F_buses
    )


    if Hpll_raw.size > 0:
        GI_sorted_local = sorted(GI_bus_order)
        col_perm = [GI_sorted_local.index(b) for b in GI_bus_order]
        Hpll = Hpll_raw[:, col_perm]
    else:
        Hpll = Hpll_raw

    #Algebraic power balance:
    #pure load: 0 = -B_L θ - p_{ℓ,L}
    AθAw = np.zeros((nθ_A, nGI))
    AθAp = np.zeros((nθ_A, nG))

    if len(F_buses) > 0 and Hpll.shape[0] > 0:
        for fi, b_int in enumerate(F_order_for_H):
            if b_int not in thetaA_bus_order:
                continue
            r_a = thetaA_bus_order.index(b_int)
            kf_fi = Kf_vec[F_buses.index(b_int)] if b_int in F_buses else 0.0
            AθAw[r_a, :] = -kf_fi * Hpll[fi, :]


    try:
        A22inv = sl.inv(AθAθA)
    except Exception:
        A22inv = sl.pinv(AθAθA)

    def schur_row(ArowθG, ArowθA, ArowW, ArowP, Brow):
        corr = ArowθA @ A22inv
        return (ArowθG - corr @ AθAθG,
                ArowW  - corr @ AθAw,
                ArowP  - corr @ AθAp,
                Brow   - corr @ BθA)

    AθG_θG, AθG_W, AθG_P, BθG_red = schur_row(AθGθG, AθGθA, AθGw, AθGp, BθG)
    Aw_θG,  Aw_W,  Aw_P,  Bw_red  = schur_row(AwθG,  AwθA,  Aw_w, Aw_p, Bw)
    Ap_θG,  Ap_W,  Ap_P,  Bp_red  = schur_row(ApθG,  ApθA,  Ap_w, Ap_p, Bp)

    A_red = np.block([
        [AθG_θG, AθG_W, AθG_P],
        [Aw_θG,  Aw_W,  Aw_P ],
        [Ap_θG,  Ap_W,  Ap_P ],
    ])

    B_unified_red = np.vstack([BθG_red, Bw_red, Bp_red])
    B_red = B_unified_red @ Umap

    nx = A_red.shape[0]
    Aug = np.block([[A_red, B_red],
                    [np.zeros((m_inter, nx)), np.zeros((m_inter, m_inter))]])
    Phi = sl.expm(Aug * dt)
    Ad  = Phi[:nx, :nx]
    Bd  = Phi[:nx, nx:nx+m_inter]

    Kx     = -A22inv @ np.block([AθAθG, AθAw, AθAp])
    Ku_int = (-A22inv @ BθA) @ Umap
    T_GI = I0 @ P_GI.T
    T_A  = I0 @ P_L.T

    def pack_full_theta(tGI, tA):
        return T_GI @ tGI + T_A @ tA

    return {
        "G_buses": G_buses, "I_buses": I_buses, "GFL_buses": F_buses,
        "L_buses": L_buses,
        "GI_bus_order": GI_bus_order,
        "thetaGI_bus_order": thetaGI_bus_order,
        "thetaA_bus_order":  thetaA_bus_order,
        "slack_bus": slack_bus,
        "B": B, "Ad": Ad, "Bd": Bd,
        "A_red_cont": A_red, "B_unified_cont": B_unified_red,
        "Umap": Umap,
        "Ad_trip": None, "Bd_trip": None,
        "tripped_G_positions": [], "gen_trip_event_idx": None,
        "Kx": Kx, "Ku_int": Ku_int,
        "T_GI": T_GI, "T_A": T_A,
        "pack_full_theta": pack_full_theta,
        "nθ_GI": nθ_GI, "nθ_A": nθ_A,
        "nGI": nGI, "nG": nG, "nI": nI,
        "m_inter": m_inter,
        "Minv_GI": Minv_GI, "D_GI": D_GI,
        "GamGinv": GamGinv, "Ap_w": Ap_w, "Ap_p": Ap_p,
        "AθGw": AθGw,
        "Hpll": Hpll,
        "F_order_for_H": F_order_for_H,
        "S_F": S_F,
        "Kf_diag": Kf_diag, "Kf_vec": Kf_vec,
        "pF_ref_vec": pF_ref_vec,
    }

def build_ac_dae_helpers(N, Ybus_dense, Vmag_ref, theta_ref, u0_base, dc_model):
    G_buses = dc_model["G_buses"]
    I_buses = dc_model["I_buses"]
    F_buses = dc_model["GFL_buses"]
    L_buses = dc_model["L_buses"]

    nθ_GI = dc_model["nθ_GI"]
    nGI = dc_model["nGI"]
    nG    = dc_model["nG"]
    nI  = dc_model["nI"]
    nθ_A  = dc_model["nθ_A"]
    m_inter = dc_model["m_inter"]

    pack_full_theta = dc_model["pack_full_theta"]
    Minv_GI = dc_model["Minv_GI"]
    D_GI    = dc_model["D_GI"]
    GamGinv = dc_model["GamGinv"]
    Ap_w    = dc_model["Ap_w"]
    Ap_p    = dc_model["Ap_p"]
    AθGw    = dc_model["AθGw"]

    S_G  = selector(N, G_buses)
    S_I  = selector(N, I_buses)
    S_GI = np.concatenate([S_G, S_I], axis=1)

    Kf_vec         = dc_model["Kf_vec"]
    pF_ref_vec     = dc_model["pF_ref_vec"]
    Hpll           = dc_model["Hpll"]
    F_order_for_H  = dc_model["F_order_for_H"]

    TOL_AC = 1e-10
    MAX_IT = 60

    Gmat = np.real(Ybus_dense).astype(float)
    Bmat = np.imag(Ybus_dense).astype(float)
    Vm   = Vmag_ref.reshape(-1,1).astype(float)
    VmVm = Vm @ Vm.T

    Lidx0 = np.array([b-1 for b in L_buses], dtype=int)
    ny    = int(Lidx0.size)
    I_ny  = np.eye(ny)

    F_set = set(int(b) for b in F_buses)
    F_in_L_info = []
    for i_L, b_int in enumerate(L_buses):
        if int(b_int) in F_set:
            if b_int in F_order_for_H:
                fi    = F_order_for_H.index(b_int)
                kf_fi = Kf_vec[F_buses.index(b_int)] if b_int in F_buses else 0.0
                F_in_L_info.append((i_L, fi, kf_fi))

    def P_from_theta(theta_abs):
        th = theta_abs.reshape(-1,1)
        d = th - th.T
        return (VmVm*(Gmat*np.cos(d) + Bmat*np.sin(d))).sum(axis=1, keepdims=True)

    def dP_dtheta(theta_abs):
        th = theta_abs.reshape(-1,1)
        d = th - th.T
        Hm = VmVm*(Gmat*np.sin(d) - Bmat*np.cos(d))
        np.fill_diagonal(Hm, 0.0)
        np.fill_diagonal(Hm, -Hm.sum(axis=1))
        return Hm

    def split_u(u):
        return u[0:nI,:], u[nI:nI+nG,:], u[nI+nG:nI+nG+N,:]

    GEN_TRIP_MASK = np.ones(nG, dtype=float)
    def reset_gen_trip_mask():
        GEN_TRIP_MASK[:] = 1.0
    def apply_gen_trip(tripped_positions):
        for j in tripped_positions:
            GEN_TRIP_MASK[j] = 0.0
    def Mgen_diag():
        return np.diag(GEN_TRIP_MASK)

    def f_rhs(x, y, u_inter):
        """Differential equations for θ_GI, ω_GI, and p_G."""
        tGI = x[0:nθ_GI,:]
        wGI = x[nθ_GI:nθ_GI+nGI,:]
        pG = x[nθ_GI+nGI:,:]
        u_IBR, u_Gref, u_load = split_u(u_inter)
        Mgen = Mgen_diag()

        dth   = pack_full_theta(tGI, y)
        P_abs = P_from_theta(theta_ref.reshape(-1,1) + dth)

        # θ̇_GI = ω_base ω_GI.
        tGI_dot = AθGw @ wGI
        wGI_dot = Minv_GI @ (
            -D_GI @ wGI
            + (S_GI.T @ S_G @ Mgen) @ pG
            + (S_GI.T @ S_I) @ u_IBR
            - S_GI.T @ (u_load + u0_base + P_abs)
        )
        pG_dot = (Mgen @ (Ap_w @ wGI + Ap_p @ pG) + Mgen @ (GamGinv @ u_Gref)) \
                 - ((np.eye(nG) - Mgen) @ pG)

        return np.vstack([tGI_dot, wGI_dot, pG_dot])

    def g_and_J(x, y, u_inter):
        """Algebraic power-balance equations for load and GFL buses."""
        tGI = x[0:nθ_GI,:]
        wGI = x[nθ_GI:nθ_GI+nGI,:]
        _, _, u_load = split_u(u_inter)

        dth   = pack_full_theta(tGI, y)
        P_abs = P_from_theta(theta_ref.reshape(-1,1) + dth)

        # Pure-load balance: 0 = P + p_ℓ.
        g = P_abs[Lidx0,:] + (u_load + u0_base)[Lidx0,:]

        if Hpll.shape[0] > 0 and F_in_L_info:
            omegaF = Hpll @ wGI
            for (i_L, fi, kf_fi) in F_in_L_info:
                g[i_L, :] += kf_fi * float(omegaF[fi, 0])

        J = dP_dtheta(theta_ref.reshape(-1,1) + dth)[np.ix_(Lidx0, Lidx0)]
        return g, J

    def solve_algebraic(x, u_inter, y_guess):
        y = y_guess.copy()
        for _ in range(MAX_IT):
            g, J = g_and_J(x, y, u_inter)
            if np.linalg.norm(g, np.inf) < TOL_AC:
                return y
            condJ = np.linalg.cond(J)
            J_eff = J + 1e-8*I_ny if (not np.isfinite(condJ)) or condJ > 1e12 else J
            try:
                dy = sl.solve(J_eff, -g, assume_a="gen")
            except Exception:
                dy = sl.lstsq(J_eff, -g)[0]
            sn = np.linalg.norm(dy, np.inf)
            if sn > 0.5:
                dy *= 0.5 / sn
            alpha = 1.0
            y_new  = y + alpha*dy
            g_new, _ = g_and_J(x, y_new, u_inter)
            ninf = np.linalg.norm(g, np.inf)
            while np.linalg.norm(g_new, np.inf) > (1.0 - 1e-4*alpha)*ninf and alpha > 1e-4:
                alpha *= 0.5
                y_new = y + alpha*dy
                g_new, _ = g_and_J(x, y_new, u_inter)
            y = y_new
        warnings.warn(f"[AC] Newton did not converge: ||g||={np.linalg.norm(g, np.inf):.3e}")
        return y

    def rk4_step(x, y, u_inter, h):
        y1 = solve_algebraic(x, u_inter, y)
        k1 = f_rhs(x, y1, u_inter)
        x2 = x + 0.5*h*k1
        y2 = solve_algebraic(x2, u_inter, y1)
        k2 = f_rhs(x2, y2, u_inter)
        x3 = x + 0.5*h*k2
        y3 = solve_algebraic(x3, u_inter, y2)
        k3 = f_rhs(x3, y3, u_inter)
        x4 = x + h*k3
        y4 = solve_algebraic(x4, u_inter, y3)
        k4 = f_rhs(x4, y4, u_inter)
        xn = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return xn, solve_algebraic(xn, u_inter, y4)

    return {
        "nθ_GI": nθ_GI, "nGI": nGI, "nG": nG, "nI": nI,
        "nθ_A": nθ_A, "m_inter": m_inter,
        "solve_algebraic": solve_algebraic, "rk4_step": rk4_step, "f_rhs": f_rhs,
        "pack_full_theta": pack_full_theta,
        "GI_bus_order": dc_model["GI_bus_order"],
        "G_buses": dc_model["G_buses"], "I_buses": dc_model["I_buses"],
        "GFL_buses": dc_model["GFL_buses"],
        "Hpll": dc_model["Hpll"], "F_order_for_H": dc_model["F_order_for_H"],
        "reset_gen_trip_mask": reset_gen_trip_mask,
        "apply_gen_trip": apply_gen_trip,
        "tripped_G_positions": dc_model["tripped_G_positions"],
        "gen_trip_event_idx":  dc_model["gen_trip_event_idx"],
        "pF_ref_vec": pF_ref_vec,
        "Kf_vec": Kf_vec,
        "Kf_diag": dc_model["Kf_diag"],
    }

def build_fixed_disturbances(id_map, pf_results, baseMVA, m_inter, nI, nG):
    """Build the enabled disturbance schedule."""
    disturbances = []

    if LOAD_CHANGE_ENABLE:
        for lc_entry in sorted(LOAD_CHANGE_EVENTS, key=lambda e: float(e["event_time_s"])):
            lc_buses = lc_entry.get("buses", [])
            lc_time_s = float(lc_entry["event_time_s"])
            ev_idx = event_time_to_idx(lc_time_s)
            u = np.zeros((m_inter, 1))
            affected = []

            for b_ext in lc_buses:
                b_ext = int(b_ext)
                if b_ext not in id_map:
                    print(f"[warn] LOAD_CHANGE bus {b_ext} not in id_map; skipping.")
                    continue
                li = nI + nG + (id_map[b_ext] - 1)

                if b_ext in lc_entry.get("pu_map", {}):
                    val = float(lc_entry["pu_map"][b_ext])
                else:
                    val = 0.0

                u[li, 0] += val
                affected.append((b_ext, val))

            if affected:
                label = f"LoadChange@t={lc_time_s:.2f}s " + ", ".join(
                    f"bus={b} Δp={v:+.4f}pu" for b, v in affected
                )
                disturbances.append({
                    "label": label,
                    "event_idx": ev_idx,
                    "event_time_s": lc_time_s,
                    "u": u,
                })

    if LOAD_TRIP_ENABLE:
        ev_idx = event_time_to_idx(LOAD_TRIP_EVENT_TIME_S)
        u = np.zeros((m_inter, 1))
        bus_pf = np.asarray(pf_results["bus"], dtype=float)
        bus_pf = bus_pf[np.argsort(bus_pf[:, BUS_I].astype(int)), :]
        affected = []

        for b_ext in LOAD_TRIP_BUSES_EXT:
            b_ext = int(b_ext)
            if b_ext not in id_map:
                print(f"[warn] LOAD_TRIP bus {b_ext} not in id_map; skipping.")
                continue
            b_int = id_map[b_ext]
            li = nI + nG + (b_int - 1)
            pd_pu = float(bus_pf[b_int - 1, PD]) / baseMVA
            scale = float(LOAD_TRIP_SCALE_FIXED_EXT.get(b_ext, 1.0))
            val = -pd_pu * scale
            u[li, 0] += val
            affected.append((b_ext, val))

        if affected:
            label = f"LoadTrip@t={LOAD_TRIP_EVENT_TIME_S:.2f}s " + ", ".join(
                f"bus={b} Δp={v:+.4f}pu" for b, v in affected
            )
            disturbances.append({
                "label": label,
                "event_idx": ev_idx,
                "event_time_s": float(LOAD_TRIP_EVENT_TIME_S),
                "u": u,
            })

    if GEN_TRIP_ENABLE:
        ev_idx = event_time_to_idx(GEN_TRIP_EVENT_TIME_S)
        u = np.zeros((m_inter, 1))
        gen_pf = np.asarray(pf_results["gen"], dtype=float)
        affected = []

        for b_ext in GEN_TRIP_BUSES_EXT:
            b_ext = int(b_ext)
            if b_ext not in id_map:
                print(f"[warn] GEN_TRIP bus {b_ext} not in id_map; skipping.")
                continue
            b_int = id_map[b_ext]
            li = nI + nG + (b_int - 1)
            pg_pu = sum(float(r[PG]) / baseMVA for r in gen_pf
                        if int(r[GEN_BUS]) == b_int and int(r[GEN_STATUS]) == 1)
            scale = float(GEN_TRIP_SCALE_FIXED_EXT.get(b_ext, 1.0))
            val = pg_pu * scale
            u[li, 0] += val
            affected.append((b_ext, val))

        if affected:
            label = f"GenTrip@t={GEN_TRIP_EVENT_TIME_S:.2f}s " + ", ".join(
                f"bus={b} Δp={v:+.4f}pu" for b, v in affected
            )
            disturbances.append({
                "label": label,
                "event_idx": ev_idx,
                "event_time_s": float(GEN_TRIP_EVENT_TIME_S),
                "u": u,
            })

    return disturbances

# Dynamic local safety filter

def _filter_design(dc_model, inv_map):
    F_buses = list(dc_model["GFL_buses"])
    nF = len(F_buses)
    ntheta = int(dc_model["nθ_GI"])
    nGI = int(dc_model["nGI"])
    nx = int(dc_model["A_red_cont"].shape[0])
    m_inter = int(dc_model["m_inter"])
    nI = int(dc_model["nI"])
    nG = int(dc_model["nG"])

    Hraw = np.asarray(dc_model["Hpll"], dtype=float)
    F_order = list(dc_model["F_order_for_H"])
    Hf = np.vstack([Hraw[F_order.index(b), :] for b in F_buses]) if nF else np.zeros((0, nGI))

    Bred = np.asarray(dc_model["B_unified_cont"], dtype=float) @ np.asarray(dc_model["Umap"], dtype=float)
    Uz = np.zeros((m_inter, nF), dtype=float)
    load_offset = nI + nG
    for j, b in enumerate(F_buses):
        Uz[load_offset + b - 1, j] = -1.0

    Cw = np.zeros((nF, nx), dtype=float)
    Cw[:, ntheta:ntheta + nGI] = Hf
    Gs = Bred @ Uz
    Gamma = Cw @ Gs
    Gamma_sym = 0.5 * (Gamma + Gamma.T)

    if nF:
        scale = max(float(np.max(np.abs(Gamma_sym))), 1.0)
        tol = max(1e-12, 1e-10 * scale)
        eig = sl.eigvalsh(Gamma_sym)
        if eig[0] <= tol:
            raise ValueError(
                f"Dynamic-filter design requires Gamma >> 0; lambda_min={eig[0]:.3e}."
            )

        gamma_diag = np.diag(Gamma_sym).copy()
        Dinvhalf = np.diag(1.0 / np.sqrt(gamma_diag))
        normalized = Dinvhalf @ Gamma_sym @ Dinvhalf
        normalized_lmax = float(sl.eigvalsh(normalized)[-1])
        rho = float(PROX_STEP_FRACTION) / normalized_lmax
        prox_k = rho / gamma_diag

        for j, b_int in enumerate(F_buses):
            b_ext = int(inv_map.get(int(b_int), int(b_int)))
            if b_ext in PROX_K_MAP_EXT:
                prox_k[j] = float(PROX_K_MAP_EXT[b_ext])

        Khalf = np.diag(np.sqrt(prox_k))
        weighted_lmax = float(sl.eigvalsh(Khalf @ Gamma_sym @ Khalf)[-1])
        if weighted_lmax >= 2.0:
            warnings.warn(
                "Proximal gains violate lambda_max(K^(1/2) Gamma K^(1/2)) < 2."
            )
    else:
        eig = np.array([])
        gamma_diag = np.array([])
        normalized_lmax = np.nan
        rho = np.nan
        prox_k = np.array([])
        weighted_lmax = np.nan

    asym = np.linalg.norm(Gamma - Gamma.T, ord="fro") / max(np.linalg.norm(Gamma, ord="fro"), 1e-14)
    if asym > 1e-7:
        warnings.warn(f"Gamma relative asymmetry is {asym:.3e}; symmetric part is used for proximal design.")

    return {
        "F_buses": F_buses,
        "F_ext": [int(inv_map.get(int(b), int(b))) for b in F_buses],
        "Hf": Hf,
        "Cw": Cw,
        "Gs": Gs,
        "Gamma": Gamma,
        "Gamma_sym": Gamma_sym,
        "Uz": Uz,
        "prox_k": prox_k,
        "lower_pu": np.full(nF, FREQ_LOWER_HZ / F_BASE, dtype=float),
        "upper_pu": np.full(nF, FREQ_UPPER_HZ / F_BASE, dtype=float),
        "alpha_low": np.full(nF, ALPHA_CBF, dtype=float),
        "alpha_up": np.full(nF, ALPHA_CBF, dtype=float),
        "eig": eig,
        "rho": rho,
        "normalized_lmax": normalized_lmax,
        "weighted_lmax": weighted_lmax,
    }


def _project_zero_interval(a, b):
    if np.any(b < a - 1e-11):
        raise RuntimeError("Local proximal interval is not ordered.")
    return np.where(a > 0.0, a, np.where(b < 0.0, b, 0.0))


def _filter_target(omega_f, rocof_hat, z, design):
    k = design["prox_k"].reshape(-1, 1)
    lower = design["lower_pu"].reshape(-1, 1)
    upper = design["upper_pu"].reshape(-1, 1)
    alpha_l = design["alpha_low"].reshape(-1, 1)
    alpha_u = design["alpha_up"].reshape(-1, 1)
    a = z + k * (-alpha_l * (omega_f - lower) - rocof_hat)
    b = z + k * (+alpha_u * (upper - omega_f) - rocof_hat)
    return _project_zero_interval(a, b), a, b


def _filter_derivatives(omega_f, exact_rocof, z, chi, design, enabled):
    chi_dot = (-chi + omega_f) / ROCOF_TAU_S
    mode = str(ROCOF_MODE).strip().lower()
    if mode == "exact":
        rocof_hat = exact_rocof
    elif mode == "dirty":
        rocof_hat = (omega_f - chi) / ROCOF_TAU_S
    else:
        raise ValueError('ROCOF_MODE must be "dirty" or "exact".')

    if enabled:
        target, a, b = _filter_target(omega_f, rocof_hat, z, design)
        z_dot = (-z + target) / FILTER_EPSILON_S
    else:
        target = np.zeros_like(z)
        a = np.zeros_like(z)
        b = np.zeros_like(z)
        z_dot = np.zeros_like(z)
    return z_dot, chi_dot, rocof_hat, target, a, b


def _input_at_index(k, disturbances, m_inter):
    u = np.zeros((m_inter, 1), dtype=float)
    for d in disturbances:
        if k >= int(d["event_idx"]):
            u += d["u"]
    return u


def _dc_continuous_pair(dc_model, post_trip=False):
    A = np.asarray(dc_model["A_red_cont"], dtype=float).copy()
    Buni = np.asarray(dc_model["B_unified_cont"], dtype=float).copy()
    if post_trip:
        ntheta = int(dc_model["nθ_GI"])
        nGI = int(dc_model["nGI"])
        for j in dc_model.get("tripped_G_positions", []):
            p_idx = ntheta + nGI + int(j)
            A[p_idx, :] = 0.0
            A[p_idx, p_idx] = -1.0
            A[ntheta:ntheta + nGI, p_idx] = 0.0
            Buni[p_idx, :] = 0.0
    return A, Buni @ np.asarray(dc_model["Umap"], dtype=float)


def _empty_trajectory(T, N, nGI, nG, nF):
    return {
        "dtheta_full": np.zeros((T, N)),
        "omega_GI": np.zeros((T, nGI)),
        "omega_F": np.zeros((T, nF)),
        "p_F": np.zeros((T, nF)),
        "dp_F": np.zeros((T, nF)),
        "dp_F_ffr": np.zeros((T, nF)),
        "p_G": np.zeros((T, nG)),
        "z": np.zeros((T, nF)),
        "chi": np.zeros((T, nF)),
        "target": np.zeros((T, nF)),
        "rocof_hat": np.zeros((T, nF)),
        "rocof_exact": np.zeros((T, nF)),
    }


def simulate_dc(dc_model, disturbances, design, filter_enabled=False):
    A0, B0 = _dc_continuous_pair(dc_model, post_trip=False)
    A1, B1 = _dc_continuous_pair(dc_model, post_trip=True)
    nx = A0.shape[0]
    ntheta = int(dc_model["nθ_GI"])
    nGI = int(dc_model["nGI"])
    nG = int(dc_model["nG"])
    nF = len(design["F_buses"])
    N = dc_model["T_GI"].shape[0]
    m_inter = int(dc_model["m_inter"])
    trip_idx = dc_model.get("gen_trip_event_idx")

    q = np.zeros((nx + 2 * nF, 1), dtype=float)
    out = _empty_trajectory(Tsteps, N, nGI, nG, nF)
    Kf = np.asarray(dc_model["Kf_diag"], dtype=float)
    pref = np.asarray(dc_model["pF_ref_vec"], dtype=float).reshape(-1, 1)

    def stage(qs, u_base, A, B):
        x = qs[:nx, :]
        z = qs[nx:nx + nF, :]
        chi = qs[nx + nF:, :]
        u_eff = u_base + design["Uz"] @ z
        xdot = A @ x + B @ u_eff
        omega = design["Cw"] @ x
        exact = design["Cw"] @ xdot
        zdot, chidot, _, _, _, _ = _filter_derivatives(
            omega, exact, z, chi, design, filter_enabled
        )
        return np.vstack([xdot, zdot, chidot])

    for k in range(Tsteps):
        post = trip_idx is not None and k >= int(trip_idx)
        A, B = (A1, B1) if post else (A0, B0)
        u_base = _input_at_index(k, disturbances, m_inter)

        x = q[:nx, :]
        z = q[nx:nx + nF, :]
        chi = q[nx + nF:, :]
        u_eff = u_base + design["Uz"] @ z
        xdot = A @ x + B @ u_eff
        omega = design["Cw"] @ x
        exact = design["Cw"] @ xdot
        _, _, rocof_hat, target, _, _ = _filter_derivatives(
            omega, exact, z, chi, design, filter_enabled
        )

        tGI = x[:ntheta, :]
        tA = dc_model["Kx"] @ x + dc_model["Ku_int"] @ u_eff
        theta = dc_model["pack_full_theta"](tGI, tA)
        pG = x[ntheta + nGI:ntheta + nGI + nG, :]
        dffr = -Kf @ omega
        dpf = dffr + z

        out["dtheta_full"][k, :] = theta[:, 0]
        out["omega_GI"][k, :] = x[ntheta:ntheta + nGI, 0]
        out["omega_F"][k, :] = omega[:, 0]
        out["p_G"][k, :] = pG[:, 0]
        out["dp_F_ffr"][k, :] = dffr[:, 0]
        out["dp_F"][k, :] = dpf[:, 0]
        out["p_F"][k, :] = (pref + dpf)[:, 0]
        out["z"][k, :] = z[:, 0]
        out["chi"][k, :] = chi[:, 0]
        out["target"][k, :] = target[:, 0]
        out["rocof_hat"][k, :] = rocof_hat[:, 0]
        out["rocof_exact"][k, :] = exact[:, 0]

        if k == Tsteps - 1:
            break
        k1 = stage(q, u_base, A, B)
        k2 = stage(q + 0.5 * dt * k1, u_base, A, B)
        k3 = stage(q + 0.5 * dt * k2, u_base, A, B)
        k4 = stage(q + dt * k3, u_base, A, B)
        q = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return out


def simulate_ac(ac_help, dc_model, disturbances, design, filter_enabled=False):
    ntheta = int(ac_help["nθ_GI"])
    nGI = int(ac_help["nGI"])
    nG = int(ac_help["nG"])
    nF = len(design["F_buses"])
    nx = ntheta + nGI + nG
    N = dc_model["T_GI"].shape[0]
    m_inter = int(ac_help["m_inter"])
    trip_idx = ac_help.get("gen_trip_event_idx")
    tripped = ac_help.get("tripped_G_positions", [])
    solve_alg = ac_help["solve_algebraic"]
    f_rhs = ac_help["f_rhs"]
    Kf = np.asarray(ac_help["Kf_diag"], dtype=float)
    pref = np.asarray(ac_help["pF_ref_vec"], dtype=float).reshape(-1, 1)

    ac_help["reset_gen_trip_mask"]()
    q = np.zeros((nx + 2 * nF, 1), dtype=float)
    y = np.zeros((int(ac_help["nθ_A"]), 1), dtype=float)
    out = _empty_trajectory(Tsteps, N, nGI, nG, nF)

    def stage(qs, u_base, y_guess):
        x = qs[:nx, :]
        z = qs[nx:nx + nF, :]
        chi = qs[nx + nF:, :]
        u_eff = u_base + design["Uz"] @ z
        ys = solve_alg(x, u_eff, y_guess)
        xdot = f_rhs(x, ys, u_eff)
        omega = design["Hf"] @ x[ntheta:ntheta + nGI, :]
        exact = design["Hf"] @ xdot[ntheta:ntheta + nGI, :]
        zdot, chidot, _, _, _, _ = _filter_derivatives(
            omega, exact, z, chi, design, filter_enabled
        )
        return np.vstack([xdot, zdot, chidot]), ys

    for k in range(Tsteps):
        if trip_idx is not None and k == int(trip_idx) and tripped:
            ac_help["apply_gen_trip"](tripped)

        u_base = _input_at_index(k, disturbances, m_inter)
        x = q[:nx, :]
        z = q[nx:nx + nF, :]
        chi = q[nx + nF:, :]
        u_eff = u_base + design["Uz"] @ z
        y = solve_alg(x, u_eff, y)
        xdot = f_rhs(x, y, u_eff)
        omega = design["Hf"] @ x[ntheta:ntheta + nGI, :]
        exact = design["Hf"] @ xdot[ntheta:ntheta + nGI, :]
        _, _, rocof_hat, target, _, _ = _filter_derivatives(
            omega, exact, z, chi, design, filter_enabled
        )

        theta = ac_help["pack_full_theta"](x[:ntheta, :], y)
        pG = x[ntheta + nGI:ntheta + nGI + nG, :]
        dffr = -Kf @ omega
        dpf = dffr + z

        out["dtheta_full"][k, :] = theta[:, 0]
        out["omega_GI"][k, :] = x[ntheta:ntheta + nGI, 0]
        out["omega_F"][k, :] = omega[:, 0]
        out["p_G"][k, :] = pG[:, 0]
        out["dp_F_ffr"][k, :] = dffr[:, 0]
        out["dp_F"][k, :] = dpf[:, 0]
        out["p_F"][k, :] = (pref + dpf)[:, 0]
        out["z"][k, :] = z[:, 0]
        out["chi"][k, :] = chi[:, 0]
        out["target"][k, :] = target[:, 0]
        out["rocof_hat"][k, :] = rocof_hat[:, 0]
        out["rocof_exact"][k, :] = exact[:, 0]

        if k == Tsteps - 1:
            break

        k1, y1 = stage(q, u_base, y)
        k2, y2 = stage(q + 0.5 * dt * k1, u_base, y1)
        k3, y3 = stage(q + 0.5 * dt * k2, u_base, y2)
        k4, y4 = stage(q + dt * k3, u_base, y3)
        q = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        znew = q[nx:nx + nF, :]
        y = solve_alg(q[:nx, :], u_base + design["Uz"] @ znew, y4)

    ac_help["reset_gen_trip_mask"]()
    return out


# Trajectory access and export

def freq_trace_from_traj(traj, dc_model, bus_int):
    GI = list(dc_model["GI_bus_order"])
    F = list(dc_model["GFL_buses"])
    if bus_int in GI:
        return F_BASE * traj["omega_GI"][:, GI.index(bus_int)]
    if bus_int in F:
        return F_BASE * traj["omega_F"][:, F.index(bus_int)]
    raise RuntimeError("Frequency trace requested at a non-resource bus.")


def angle_trace_from_traj(traj, bus_int):
    return traj["dtheta_full"][:, int(bus_int) - 1]


def pg_trace_from_traj(traj, dc_model, bus_int):
    G = list(dc_model["G_buses"])
    if bus_int not in G:
        raise RuntimeError("Mechanical-power trace requested at a non-SG bus.")
    return traj["p_G"][:, G.index(bus_int)]


def dpf_trace_from_traj(traj, dc_model, bus_int):
    F = list(dc_model["GFL_buses"])
    if bus_int not in F:
        raise RuntimeError("GFL active-power trace requested at a non-GFL bus.")
    return traj["dp_F"][:, F.index(bus_int)]


def export_trajectory_csvs(traj, dc_model, plot_buses_int, inv_map, mode_tag,
                           want_freq, want_angle, want_pg, want_dpf):
    ensure_dir(CSV_EXPORT_DIR)
    GI = list(dc_model["GI_bus_order"])
    G = list(dc_model["G_buses"])
    F = list(dc_model["GFL_buses"])
    for b_int in plot_buses_int:
        b_ext = int(inv_map.get(int(b_int), int(b_int)))
        if want_freq and (b_int in GI or b_int in F):
            write_freq_traj_csv(
                os.path.join(CSV_EXPORT_DIR, f"freq_bus{b_ext}_{mode_tag}_python.csv"),
                tgrid, freq_trace_from_traj(traj, dc_model, b_int))
        if want_angle:
            write_angle_traj_csv(
                os.path.join(CSV_EXPORT_DIR, f"angle_bus{b_ext}_{mode_tag}_python.csv"),
                tgrid, angle_trace_from_traj(traj, b_int))
        if want_pg and b_int in G:
            write_pg_traj_csv(
                os.path.join(CSV_EXPORT_DIR, f"pg_bus{b_ext}_{mode_tag}_python.csv"),
                tgrid, pg_trace_from_traj(traj, dc_model, b_int))
        if want_dpf and b_int in F:
            write_dpf_traj_csv(
                os.path.join(CSV_EXPORT_DIR, f"dpf_bus{b_ext}_{mode_tag}_python.csv"),
                tgrid, dpf_trace_from_traj(traj, dc_model, b_int))

    for j, b_int in enumerate(F):
        b_ext = int(inv_map.get(int(b_int), int(b_int)))
        path = os.path.join(CSV_EXPORT_DIR, f"gfl_filter_bus{b_ext}_{mode_tag}_python.csv")
        with open(path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([
                "t_s", "frequency_deviation_hz", "z_command_system_pu",
                "delta_pF_ffr_system_pu", "delta_pF_system_pu",
                "rocof_hat_pu_s", "rocof_exact_pu_s"
            ])
            for k, t in enumerate(tgrid):
                w.writerow([
                    float(t), float(F_BASE * traj["omega_F"][k, j]),
                    float(traj["z"][k, j]), float(traj["dp_F_ffr"][k, j]),
                    float(traj["dp_F"][k, j]), float(traj["rocof_hat"][k, j]),
                    float(traj["rocof_exact"][k, j])
                ])


# Publication plotting

FIGURE_WIDTH = 7.10
AXIS_LW = 0.82
TICK_LW = 0.74
FONT_BASE = 10.5
FONT_XLABEL = 14.0
FONT_YLABEL = 13.2
FONT_LEGEND = 9.8
FONT_STATUS = 8.1
PNG_DPI = 600
PDF_DPI = 300

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
COLOR_LIGHT = COLOR_ZERO
COLOR_Z = COLOR_FILTER

BUS_COLORS = {
    30: "#0072B2", 31: "#009E73", 32: "#E69F00", 33: "#3A923A",
    34: "#56B4E9", 35: "#CC6677", 36: "#D55E00", 37: "#AA4499",
    38: "#7E57C2", 39: "#008B8B",
}


def configure_matplotlib():
    use_tex = bool(USE_EXTERNAL_LATEX and shutil.which("latex"))
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_YLABEL,
        "xtick.labelsize": 10.2,
        "ytick.labelsize": 10.2,
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


def _style_axis(ax, ybins=5, zero=False):
    ax.set_xlim(float(tgrid[0]), float(tgrid[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LW)
        spine.set_color(COLOR_AXIS)
    ax.tick_params(direction="in", top=True, right=True, labeltop=False, labelright=False,
                   length=3.7, width=TICK_LW, pad=3.0)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.42, alpha=0.40, zorder=-5)
    ax.margins(x=0.0)
    if zero:
        ax.axhline(0.0, color=COLOR_ZERO, linewidth=0.72, zorder=0)


def _event_lines(ax, disturbances):
    for t_ev in sorted(set(float(d["event_time_s"]) for d in disturbances)):
        ax.axvline(t_ev, color=COLOR_EVENT, linestyle=(0, (1.0, 1.7)),
                   linewidth=0.78, alpha=0.88, zorder=0)


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
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    if span < 1e-8:
        span = max(abs(lo), abs(hi), 1.0) * 0.05
        lo -= span
        hi += span
        span = hi - lo
    ax.set_ylim(lo - lower_pad * span, hi + upper_pad * span)


def _safe_save(fig, path, **kwargs):
    try:
        fig.savefig(path, **kwargs)
        return
    except OSError:
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        fig.savefig(tmp, **kwargs)
        shutil.copyfile(tmp, path)
        try:
            os.remove(tmp)
        except OSError:
            pass


def save_figure(fig, stem):
    ensure_dir(FIGURE_EXPORT_DIR)
    common = dict(bbox_inches="tight", pad_inches=0.035, facecolor="white")

    if SAVE_PDF:
        _safe_save(
            fig,
            os.path.join(FIGURE_EXPORT_DIR, stem + ".pdf"),
            format="pdf",
            dpi=PDF_DPI,
            **common,
        )

    if SAVE_PNG:
        _safe_save(
            fig,
            os.path.join(FIGURE_EXPORT_DIR, stem + ".png"),
            format="png",
            dpi=PNG_DPI,
            **common,
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _role(dc_model, b_int):
    if b_int in dc_model["GFL_buses"]:
        return "GFL"
    if b_int in dc_model["G_buses"]:
        return "SG"
    return "GFM"


def _resource_style(dc_model, b_int):
    role = _role(dc_model, b_int)
    if role == "GFL":
        return "-", None, 1.50
    if role == "SG":
        return "--", (5.5, 2.8), 1.45
    return "-.", None, 1.45


def _active_side(baseline, active):
    data = np.concatenate([F_BASE * baseline["omega_F"].reshape(-1),
                           F_BASE * active["omega_F"].reshape(-1)])
    mode = str(ACTIVE_SAFETY_BOUND).strip().lower()
    if mode in {"lower", "upper"}:
        return mode
    if mode != "auto":
        raise ValueError('ACTIVE_SAFETY_BOUND must be "auto", "lower", or "upper".')
    fmin, fmax = float(np.min(data)), float(np.max(data))
    lower = max(0.0, -fmin) / max(abs(FREQ_LOWER_HZ), 1e-12)
    upper = max(0.0, fmax) / max(abs(FREQ_UPPER_HZ), 1e-12)
    return "lower" if lower >= upper else "upper"


def _frequency_bound(ax, side):
    value = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    label = r"$\Delta f_{\min}$" if side == "lower" else r"$\Delta f_{\max}$"
    ax.axhline(value, color=COLOR_LIMIT, linestyle=(0, (4.0, 2.2)), linewidth=1.0, zorder=1)
    ax.text(0.986, value, label, transform=ax.get_yaxis_transform(), ha="right",
            va="bottom" if side == "lower" else "top", fontsize=8.2, color=COLOR_LIMIT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.6})


def _status(ax, enabled):
    ax.text(0.985, 0.955, "Filter ON" if enabled else "Filter OFF",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT_STATUS,
            color=COLOR_APPLIED if enabled else COLOR_FILTER,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.6})


def plot_frequency_safety(dc_model, baseline, active, disturbances, plot_buses_int, inv_map):
    side = _active_side(baseline, active)
    compare = bool(ENABLE_SAFETY_FILTER and RUN_NOMINAL_COMPARISON)
    rows = 2 if compare else 1
    fig, axes = plt.subplots(rows, 1, figsize=(FIGURE_WIDTH, 4.85 if compare else 2.95),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    handles, labels, all_arrays = [], [], []

    valid = [b for b in plot_buses_int if b in dc_model["GI_bus_order"] or b in dc_model["GFL_buses"]]
    for b in valid:
        ext = int(inv_map.get(int(b), int(b)))
        off = freq_trace_from_traj(baseline, dc_model, b)
        on = freq_trace_from_traj(active, dc_model, b)
        ls, dashes, lw = _resource_style(dc_model, b)
        kw = dict(color=BUS_COLORS.get(ext, COLOR_INK), linestyle=ls, linewidth=lw)
        if dashes is not None:
            kw["dashes"] = dashes
        line, = axes[0].plot(tgrid, off if compare else on, **kw)
        if compare:
            axes[1].plot(tgrid, on, **kw)
        handles.append(line)
        labels.append(f"Bus {ext} ({_role(dc_model, b)})")
        all_arrays.extend([off, on])

    bound = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    vals = np.concatenate([np.asarray(a) for a in all_arrays] + [np.array([0.0, bound])])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 1e-3)
    ylim = (lo - 0.06 * span, hi + 0.08 * span)

    for i, ax in enumerate(axes):
        _frequency_bound(ax, side)
        _event_lines(ax, disturbances)
        _style_axis(ax)
        ax.set_ylim(*ylim)
        ax.set_ylabel(r"$\Delta f_i\,[\mathrm{Hz}]$", fontsize=FONT_YLABEL, labelpad=5.0)
        _status(ax, False if compare and i == 0 else True)

    axes[-1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_XLABEL, labelpad=4.5)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=min(5, max(1, len(labels))), frameon=False,
               fontsize=FONT_LEGEND, handlelength=2.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.105 if compare else 0.17,
                        top=0.80 if compare else 0.76, hspace=0.065)
    save_figure(fig, "frequency_safety_filter_off_on" if compare else "frequency_safety_filter_on")


def plot_barrier_safety(dc_model, baseline, active, disturbances, inv_map):
    side = _active_side(baseline, active)
    compare = bool(ENABLE_SAFETY_FILTER and RUN_NOMINAL_COMPARISON)
    rows = 2 if compare else 1
    fig, axes = plt.subplots(rows, 1, figsize=(FIGURE_WIDTH, 4.55 if compare else 2.80),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    handles, labels, arrays = [], [], []

    for j, b in enumerate(dc_model["GFL_buses"]):
        ext = int(inv_map.get(int(b), int(b)))
        f0 = F_BASE * baseline["omega_F"][:, j]
        f1 = F_BASE * active["omega_F"][:, j]
        h0 = f0 - FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ - f0
        h1 = f1 - FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ - f1
        color = BUS_COLORS.get(ext, COLOR_INK)
        line, = axes[0].plot(tgrid, h0 if compare else h1, color=color, linewidth=1.50)
        if compare:
            axes[1].plot(tgrid, h1, color=color, linewidth=1.50)
        handles.append(line)
        labels.append(f"Bus {ext}")
        arrays.extend([h0, h1])

    tmp = np.concatenate(arrays + [np.array([0.0])])
    lo, hi = float(np.min(tmp)), float(np.max(tmp))
    span = max(hi - lo, 1e-3)
    ylim = (lo - 0.10 * span, hi + 0.12 * span)
    ylabel = r"$h_i^{\min}\,[\mathrm{Hz}]$" if side == "lower" else r"$h_i^{\max}\,[\mathrm{Hz}]$"

    for i, ax in enumerate(axes):
        ax.axhspan(ylim[0], 0.0, color=COLOR_UNSAFE_FILL, alpha=0.45, linewidth=0.0, zorder=-4)
        ax.axhline(0.0, color=COLOR_LIMIT, linestyle=(0, (3.0, 2.0)), linewidth=0.95)
        _event_lines(ax, disturbances)
        _style_axis(ax)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=FONT_YLABEL, labelpad=5.0)
        _status(ax, False if compare and i == 0 else True)

    axes[-1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_XLABEL, labelpad=4.5)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=min(5, len(labels)), frameon=False, fontsize=FONT_LEGEND)
    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.105 if compare else 0.17,
                        top=0.84 if compare else 0.80, hspace=0.065)
    save_figure(fig, "gfl_barrier_safety_filter_off_on" if compare else "gfl_barrier_safety_filter_on")


def plot_filter_state(active, dc_model, disturbances, inv_map):
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.80))
    handles, labels = [], []
    arrays = []
    for j, b in enumerate(dc_model["GFL_buses"]):
        ext = int(inv_map.get(int(b), int(b)))
        z = active["z"][:, j]
        line, = ax.plot(tgrid, z, color=BUS_COLORS.get(ext, COLOR_INK), linewidth=1.52)
        handles.append(line)
        labels.append(f"Bus {ext}")
        arrays.append(z)
    _event_lines(ax, disturbances)
    _style_axis(ax, zero=True)
    _set_ylim(ax, arrays, include_zero=True, lower_pad=0.12, upper_pad=0.18)
    ax.set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_XLABEL, labelpad=4.5)
    ax.set_ylabel(r"$p_{{\mathrm{s}},i}=z_i\,[\mathrm{p.u.}]$", fontsize=FONT_YLABEL, labelpad=5.0)
    _status(ax, True)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=min(5, len(labels)), frameon=False, fontsize=FONT_LEGEND)
    fig.subplots_adjust(left=0.112, right=0.988, bottom=0.18, top=0.79)
    save_figure(fig, "local_safety_filter_corrections")


def plot_gfl_power_decomposition(active, disturbances):
    dffr = np.sum(active["dp_F_ffr"], axis=1)
    zsum = np.sum(active["z"], axis=1)
    applied = np.sum(active["dp_F"], axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 4.55), sharex=True)

    l0, = axes[0].plot(tgrid, dffr, color=COLOR_FFR, linestyle="--", linewidth=1.40)
    l1, = axes[0].plot(tgrid, applied, color=COLOR_APPLIED, linewidth=1.70)
    fill = axes[0].fill_between(tgrid, dffr, applied, color=COLOR_FILL, alpha=0.30, linewidth=0.0)
    axes[0].legend([l0, l1, fill], [r"$\Delta P_F^{\rm FFR}$", r"$\Delta P_F$", r"$\sum_i z_i$"],
                   loc="upper right", fontsize=FONT_LEGEND)

    l2, = axes[1].plot(tgrid, dffr, color=COLOR_FFR, linewidth=1.55)
    l3, = axes[1].plot(tgrid, zsum, color=COLOR_FILTER, linewidth=1.55)
    axes[1].legend([l2, l3], [r"$\Delta P_F^{\rm FFR}$", r"$\sum_i z_i$"],
                   loc="upper right", fontsize=FONT_LEGEND)

    for ax in axes:
        _event_lines(ax, disturbances)
        _style_axis(ax, zero=True)
        ax.set_ylabel(r"$\Delta P_F\,[\mathrm{p.u.}]$", fontsize=FONT_YLABEL, labelpad=5.0)
    _set_ylim(axes[0], [dffr, applied], include_zero=True, upper_pad=0.20)
    _set_ylim(axes[1], [dffr, zsum], include_zero=True, upper_pad=0.20)
    axes[1].set_xlabel(r"$t\,[\mathrm{s}]$", fontsize=FONT_XLABEL, labelpad=4.5)
    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.105, top=0.985, hspace=0.075)
    save_figure(fig, "aggregate_gfl_power_decomposition")



# dq0-EMT CSV validation

def _script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def _emt_sentinel(root):
    return Path(root) / "filter_off" / "gfl_bus30.csv"


def _discover_emt_csv_root():
    if EMT_CSV_ROOT_OVERRIDE is not None:
        root = Path(EMT_CSV_ROOT_OVERRIDE).expanduser()
        if _emt_sentinel(root).is_file():
            return root
        raise FileNotFoundError(
            "EMT_CSV_ROOT_OVERRIDE does not contain the expected Julia CSVs:\n"
            f"  {root}"
        )

    base = _script_dir()
    candidates = [
        base / "Full_Order" / "plots" / "python_style_cbf" / "csv",
        base / "Full_order" / "plots" / "python_style_cbf" / "csv",
        base / "plots" / "python_style_cbf" / "csv",
        Path.cwd() / "Full_Order" / "plots" / "python_style_cbf" / "csv",
        Path.cwd() / "Full_order" / "plots" / "python_style_cbf" / "csv",
        Path.cwd() / "plots" / "python_style_cbf" / "csv",
        base.parent / "Full_Order" / "plots" / "python_style_cbf" / "csv",
        base.parent / "Full_order" / "plots" / "python_style_cbf" / "csv",
    ]

    seen = set()
    for root in candidates:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        if _emt_sentinel(root).is_file():
            return root

    search_bases = [base, base.parent, Path.cwd()]
    for search_base in search_bases:
        if not search_base.exists():
            continue
        try:
            for hit in search_base.rglob("gfl_bus30.csv"):
                if hit.parent.name == "filter_off":
                    root = hit.parent.parent
                    if _emt_sentinel(root).is_file():
                        return root
        except (OSError, PermissionError):
            pass

    checked = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find the Julia dq0-EMT CSV root automatically.\n"
        "Expected a folder containing filter_off/gfl_bus30.csv.\n\n"
        "Checked:\n"
        f"{checked}\n\n"
        "Set EMT_CSV_ROOT_OVERRIDE to the full "
        ".../plots/python_style_cbf/csv path if needed."
    )


def _read_numeric_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))

    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {path}")

    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            try:
                cols[key].append(float(value))
            except (TypeError, ValueError):
                cols[key].append(np.nan)

    return {key: np.asarray(value, dtype=float) for key, value in cols.items()}


def _emt_resource_path(root, bus_ext, filter_on):
    tag = "filter_on" if filter_on else "filter_off"
    if int(bus_ext) in USER_GFL_BUSES_EXT:
        return Path(root) / tag / f"gfl_bus{int(bus_ext):02d}.csv"
    if int(bus_ext) in USER_SG_BUSES_EXT:
        return Path(root) / tag / f"sg_bus{int(bus_ext):02d}.csv"
    raise ValueError(f"Bus {bus_ext} is not a configured GFL/SG resource.")


def _emt_frequency(root, bus_ext, filter_on):
    data = _read_numeric_csv(_emt_resource_path(root, bus_ext, filter_on))
    return data["t_s"], data["frequency_deviation_hz"]


def _emt_gfl_signal(root, bus_ext, filter_on, column):
    if int(bus_ext) not in USER_GFL_BUSES_EXT:
        raise ValueError(f"Bus {bus_ext} is not a GFL bus.")
    data = _read_numeric_csv(_emt_resource_path(root, bus_ext, filter_on))
    if column not in data:
        raise KeyError(
            f"Column '{column}' is missing in "
            f"{_emt_resource_path(root, bus_ext, filter_on)}"
        )
    return data["t_s"], data[column]


def _analytical_frequency_from_run(traj, dc_model, id_map, bus_ext):
    if int(bus_ext) not in id_map:
        raise KeyError(f"Bus {bus_ext} not present in the analytical case.")
    b_int = id_map[int(bus_ext)]
    return tgrid, freq_trace_from_traj(traj, dc_model, b_int)


def _analytical_gfl_index(dc_model, id_map, bus_ext):
    b_int = id_map[int(bus_ext)]
    F = list(dc_model["GFL_buses"])
    if b_int not in F:
        raise ValueError(f"Bus {bus_ext} is not a GFL bus in the analytical model.")
    return F.index(b_int)


def _emt_available(root):
    required = []
    for tag in ("filter_off", "filter_on"):
        for bus in EMT_OVERLAY_BUSES_EXT:
            required.append(_emt_resource_path(root, bus, tag == "filter_on"))
    missing = [p for p in required if not p.is_file()]
    return len(missing) == 0, missing


def _emt_metadata(root):
    path = Path(root) / "filter_on" / "case_metadata.csv"
    if not path.is_file():
        return {}

    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))

    return {
        str(r.get("quantity", "")).strip(): str(r.get("value", "")).strip()
        for r in rows
        if str(r.get("quantity", "")).strip()
    }


def _report_emt_parameter_match(root):
    meta = _emt_metadata(root)
    if not meta:
        print("[EMT] case_metadata.csv not found; controller parameter matching not checked.")
        return

    checks = [
        ("frequency_lower_hz", FREQ_LOWER_HZ, "frequency lower"),
        ("frequency_upper_hz", FREQ_UPPER_HZ, "frequency upper"),
        ("alpha_lower", ALPHA_CBF, "alpha"),
        ("filter_epsilon_s", FILTER_EPSILON_S, "epsilon"),
        ("rocof_tau_s", ROCOF_TAU_S, "RoCoF tau"),
        ("prox_step_fraction", PROX_STEP_FRACTION, "prox step"),
    ]

    print("\nAnalytical / dq0-EMT controller settings")
    print("-" * 72)
    mismatch = False

    for key, analytical_value, label in checks:
        if key not in meta:
            print(f"{label:20s}: analytical={analytical_value:g}, EMT=not recorded")
            continue

        try:
            emt_value = float(meta[key])
        except ValueError:
            print(f"{label:20s}: analytical={analytical_value:g}, EMT={meta[key]}")
            continue

        ok = np.isclose(analytical_value, emt_value, rtol=1e-9, atol=1e-12)
        mismatch = mismatch or (not ok)
        status = "OK" if ok else "MISMATCH"
        print(
            f"{label:20s}: analytical={analytical_value:g}, "
            f"EMT={emt_value:g}  [{status}]"
        )

    if mismatch:
        warnings.warn(
            "Analytical and dq0-EMT safety-filter settings are not identical. "
            "The overlay is still plotted, but it is not a like-for-like "
            "controller validation until the settings are synchronized."
        )


def _validation_metric(ta, ya, te, ye):
    ta = np.asarray(ta, dtype=float)
    ya = np.asarray(ya, dtype=float)
    te = np.asarray(te, dtype=float)
    ye = np.asarray(ye, dtype=float)

    lo = max(float(np.nanmin(ta)), float(np.nanmin(te)))
    hi = min(float(np.nanmax(ta)), float(np.nanmax(te)))
    mask = np.isfinite(te) & np.isfinite(ye) & (te >= lo) & (te <= hi)
    tg = te[mask]

    if tg.size < 2:
        return np.nan, np.nan, np.nan

    ya_i = np.interp(tg, ta, ya)
    err = ya_i - ye[mask]
    return (
        float(np.sqrt(np.mean(err ** 2))),
        float(np.mean(np.abs(err))),
        float(np.max(np.abs(err))),
    )


def _overlay_model_handles():
    return [
        Line2D(
            [0], [0],
            color=COLOR_INK,
            linestyle=ANALYTICAL_LINESTYLE,
            linewidth=ANALYTICAL_OVERLAY_LW,
            label="Analytical",
        ),
        Line2D(
            [0], [0],
            color=COLOR_INK,
            linestyle=EMT_LINESTYLE,
            linewidth=EMT_OVERLAY_LW,
            label=r"$dq0$-EMT",
        ),
    ]


def _overlay_bus_handles(buses):
    return [
        Line2D(
            [0], [0],
            color=BUS_COLORS.get(int(bus), COLOR_INK),
            linewidth=1.7,
            label=f"Bus {int(bus)}",
        )
        for bus in buses
    ]



def plot_analytical_vs_emt_frequency(
    emt_root,
    dc_model,
    baseline,
    active,
    disturbances,
    id_map,
):
    side = _active_side(baseline, active)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(FIGURE_WIDTH, 4.90),
        sharex=True,
        sharey=True,
    )

    all_arrays = []
    metric_rows = []

    for panel, filter_on in enumerate((False, True)):
        ax = axes[panel]
        atraj = active if filter_on else baseline

        for bus_ext in EMT_OVERLAY_BUSES_EXT:
            ta, fa = _analytical_frequency_from_run(
                atraj, dc_model, id_map, bus_ext
            )
            te, fe = _emt_frequency(emt_root, bus_ext, filter_on)

            color = BUS_COLORS.get(int(bus_ext), COLOR_INK)

            ax.plot(
                ta, fa,
                color=color,
                linestyle=ANALYTICAL_LINESTYLE,
                linewidth=ANALYTICAL_OVERLAY_LW,
                zorder=3,
            )
            ax.plot(
                te, fe,
                color=color,
                linestyle=EMT_LINESTYLE,
                linewidth=EMT_OVERLAY_LW,
                zorder=2,
            )

            all_arrays.extend([fa, fe])

            rmse, mae, emax = _validation_metric(ta, fa, te, fe)
            metric_rows.append(
                (
                    "filter_on" if filter_on else "filter_off",
                    int(bus_ext),
                    rmse,
                    mae,
                    emax,
                    float(np.min(fa)),
                    float(np.min(fe)),
                )
            )

        _frequency_bound(ax, side)
        _event_lines(ax, disturbances)
        _style_axis(ax)
        ax.set_ylabel(
            r"$\Delta f_i\,[\mathrm{Hz}]$",
            fontsize=FONT_YLABEL,
            labelpad=5.0,
        )
        _status(ax, filter_on)

    bound = FREQ_LOWER_HZ if side == "lower" else FREQ_UPPER_HZ
    vals = np.concatenate(
        [np.asarray(a).reshape(-1) for a in all_arrays]
        + [np.array([0.0, bound])]
    )
    vals = vals[np.isfinite(vals)]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 1e-3)
    ylim = (lo - 0.06 * span, hi + 0.08 * span)

    for ax in axes:
        ax.set_ylim(*ylim)

    axes[-1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )

    bus_handles = _overlay_bus_handles(EMT_OVERLAY_BUSES_EXT)
    model_handles = _overlay_model_handles()

    leg1 = fig.legend(
        handles=bus_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(6, len(bus_handles)),
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.8,
        columnspacing=2.0,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.105,
        top=0.79,
        hspace=0.065,
    )

    save_figure(fig, "analytical_vs_dq0_emt_frequency_filter_off_on")

    outdir = os.path.join(FIGURE_EXPORT_DIR, "validation")
    ensure_dir(outdir)
    path = os.path.join(outdir, "analytical_vs_dq0_emt_frequency_metrics.csv")
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "case", "bus", "rmse_hz", "mae_hz", "max_abs_error_hz",
            "analytical_nadir_hz", "emt_nadir_hz",
        ])
        w.writerows(metric_rows)
    print(f"[EMT] Wrote validation metrics: {path}")


def plot_analytical_vs_emt_filter_state(
    emt_root,
    dc_model,
    active,
    disturbances,
    id_map,
):
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.90))
    arrays = []

    for bus_ext in USER_GFL_BUSES_EXT:
        j = _analytical_gfl_index(dc_model, id_map, bus_ext)
        za = np.asarray(active["z"][:, j], dtype=float)
        te, ze = _emt_gfl_signal(
            emt_root, bus_ext, True, "z_command_system_pu"
        )

        color = BUS_COLORS.get(int(bus_ext), COLOR_INK)
        ax.plot(
            tgrid, za,
            color=color,
            linestyle=ANALYTICAL_LINESTYLE,
            linewidth=ANALYTICAL_OVERLAY_LW,
        )
        ax.plot(
            te, ze,
            color=color,
            linestyle=EMT_LINESTYLE,
            linewidth=EMT_OVERLAY_LW,
        )
        arrays.extend([za, ze])

    _event_lines(ax, disturbances)
    _style_axis(ax, zero=True)
    _set_ylim(
        ax, arrays,
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
        r"$p_{{\mathrm{s}},i}=z_i\,[\mathrm{p.u.}]$",
        fontsize=FONT_YLABEL,
        labelpad=5.0,
    )
    _status(ax, True)

    bus_handles = _overlay_bus_handles(USER_GFL_BUSES_EXT)
    model_handles = _overlay_model_handles()

    leg1 = fig.legend(
        handles=bus_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.2,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.8,
    )

    fig.subplots_adjust(
        left=0.112,
        right=0.988,
        bottom=0.18,
        top=0.72,
    )

    save_figure(fig, "analytical_vs_dq0_emt_local_filter_state")


def _sum_emt_gfl(emt_root, column):
    tref = None
    total = None

    for bus_ext in USER_GFL_BUSES_EXT:
        t, y = _emt_gfl_signal(emt_root, bus_ext, True, column)
        if tref is None:
            tref = np.asarray(t, dtype=float)
            total = np.zeros_like(tref)

        if len(t) == len(tref) and np.allclose(t, tref):
            total += y
        else:
            total += np.interp(tref, t, y)

    return tref, total


def plot_analytical_vs_emt_gfl_power(
    emt_root,
    active,
    disturbances,
):
    pa = np.sum(active["dp_F"], axis=1)
    pa_ffr = np.sum(active["dp_F_ffr"], axis=1)

    te, pe = _sum_emt_gfl(emt_root, "delta_pF_system_pu")
    te_ffr, pe_ffr = _sum_emt_gfl(
        emt_root, "delta_pF_ffr_system_pu"
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(FIGURE_WIDTH, 4.45),
        sharex=True,
    )

    axes[0].plot(
        tgrid, pa,
        color=COLOR_APPLIED,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=1.70,
        label="Analytical",
    )
    axes[0].plot(
        te, pe,
        color=COLOR_APPLIED,
        linestyle=EMT_LINESTYLE,
        linewidth=1.50,
        label=r"$dq0$-EMT",
    )

    axes[1].plot(
        tgrid, pa_ffr,
        color=COLOR_FFR,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=1.60,
        label="Analytical",
    )
    axes[1].plot(
        te_ffr, pe_ffr,
        color=COLOR_FFR,
        linestyle=EMT_LINESTYLE,
        linewidth=1.45,
        label=r"$dq0$-EMT",
    )

    for ax in axes:
        _event_lines(ax, disturbances)
        _style_axis(ax, zero=True)
        ax.legend(
            loc="best",
            fontsize=FONT_LEGEND,
        )

    _set_ylim(
        axes[0], [pa, pe],
        include_zero=True,
        upper_pad=0.20,
    )
    _set_ylim(
        axes[1], [pa_ffr, pe_ffr],
        include_zero=True,
        upper_pad=0.20,
    )

    axes[0].set_ylabel(
        r"$\Delta P_F\,[\mathrm{p.u.}]$",
        fontsize=FONT_YLABEL,
        labelpad=5.0,
    )
    axes[1].set_ylabel(
        r"$\Delta P_F^{\rm FFR}\,[\mathrm{p.u.}]$",
        fontsize=FONT_YLABEL,
        labelpad=5.0,
    )
    axes[1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )

    fig.subplots_adjust(
        left=0.112,
        right=0.988,
        bottom=0.11,
        top=0.985,
        hspace=0.075,
    )

    save_figure(fig, "analytical_vs_dq0_emt_aggregate_gfl_power")


def plot_analytical_vs_emt_frequency_error(
    emt_root,
    dc_model,
    baseline,
    active,
    disturbances,
    id_map,
):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(FIGURE_WIDTH, 4.65),
        sharex=True,
        sharey=True,
    )

    arrays = []

    for panel, filter_on in enumerate((False, True)):
        ax = axes[panel]
        atraj = active if filter_on else baseline

        for bus_ext in EMT_OVERLAY_BUSES_EXT:
            ta, fa = _analytical_frequency_from_run(
                atraj, dc_model, id_map, bus_ext
            )
            te, fe = _emt_frequency(emt_root, bus_ext, filter_on)

            mask = (
                np.isfinite(te)
                & np.isfinite(fe)
                & (te >= ta[0])
                & (te <= ta[-1])
            )
            tg = te[mask]
            if tg.size < 2:
                continue

            err = np.interp(tg, ta, fa) - fe[mask]
            arrays.append(err)

            ax.plot(
                tg, err,
                color=BUS_COLORS.get(int(bus_ext), COLOR_INK),
                linewidth=1.45,
            )

        _event_lines(ax, disturbances)
        _style_axis(ax, zero=True)
        ax.set_ylabel(
            r"$e_{f,i}\,[\mathrm{Hz}]$",
            fontsize=FONT_YLABEL,
            labelpad=5.0,
        )
        _status(ax, filter_on)

    if arrays:
        vals = np.concatenate([np.asarray(a) for a in arrays] + [np.array([0.0])])
        lo, hi = float(np.min(vals)), float(np.max(vals))
        span = max(hi - lo, 1e-4)
        ylim = (lo - 0.10 * span, hi + 0.10 * span)
        axes[0].set_ylim(*ylim)
        axes[1].set_ylim(*ylim)

    axes[-1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=4.5,
    )

    fig.legend(
        handles=_overlay_bus_handles(EMT_OVERLAY_BUSES_EXT),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(6, len(EMT_OVERLAY_BUSES_EXT)),
        frameon=False,
        fontsize=FONT_LEGEND,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.105,
        top=0.86,
        hspace=0.065,
    )

    save_figure(fig, "analytical_minus_dq0_emt_frequency_error")




def _analytical_gfl_frequency(traj, dc_model, id_map, bus_ext):
    return _analytical_frequency_from_run(traj, dc_model, id_map, bus_ext)


def _analytical_sg_frequency(traj, dc_model, id_map, bus_ext):
    if int(bus_ext) not in id_map:
        raise KeyError(f"Bus {bus_ext} not present in the analytical case.")
    b_int = id_map[int(bus_ext)]
    G = list(dc_model["G_buses"])
    if b_int not in G:
        raise ValueError(f"Bus {bus_ext} is not an SG bus in the analytical model.")
    j = G.index(b_int)
    if "omega_G" in traj:
        return tgrid, F_BASE * np.asarray(traj["omega_G"][:, j], dtype=float)
    if "omega_GI" in traj:
        return tgrid, F_BASE * np.asarray(traj["omega_GI"][:, j], dtype=float)
    raise KeyError("Analytical trajectory does not contain omega_G or omega_GI.")


def _emt_aggregate_gfl(root, filter_on, column):
    tag = bool(filter_on)
    tref = None
    total = None
    for bus_ext in USER_GFL_BUSES_EXT:
        t, y = _emt_gfl_signal(root, bus_ext, tag, column)
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if tref is None:
            tref = t
            total = np.zeros_like(tref)
        if len(t) == len(tref) and np.allclose(t, tref):
            total += y
        else:
            total += np.interp(tref, t, y)
    return tref, total


def _emt_gfl_frequency(root, bus_ext, filter_on):
    return _emt_frequency(root, bus_ext, filter_on)


def _emt_sg_frequency(root, bus_ext, filter_on):
    return _emt_frequency(root, bus_ext, filter_on)


def _safe_emt_gfl_signal(root, bus_ext, filter_on, column, zeros_if_missing=False):
    path = _emt_resource_path(root, bus_ext, filter_on)
    data = _read_numeric_csv(path)
    t = np.asarray(data["t_s"], dtype=float)
    if column in data:
        return t, np.asarray(data[column], dtype=float)
    if zeros_if_missing:
        return t, np.zeros_like(t)
    raise KeyError(f"Column '{column}' is missing in {path}")


def _panel_status_legend_handles():
    return [
        Line2D([0], [0], color=COLOR_INK, linestyle=(0, (5.5, 2.8)), linewidth=1.45, label="SF off"),
        Line2D([0], [0], color=COLOR_INK, linestyle="-", linewidth=1.55, label="SF on"),
    ]


def _resource_bus_handles():
    handles = []
    for bus in USER_GFL_BUSES_EXT + USER_SG_BUSES_EXT:
        handles.append(
            Line2D([0], [0], color=BUS_COLORS.get(int(bus), COLOR_INK), linewidth=1.8, label=f"Bus {int(bus)}")
        )
    return handles


def _summary_frequency_ylim(side):
    if side == "lower":
        return (min(FREQ_LOWER_HZ - 0.045, -0.115), 0.010)
    return (-0.010, max(FREQ_UPPER_HZ + 0.045, 0.115))


def _summary_barrier(traces, side):
    if side == "lower":
        return [
            np.asarray(y, dtype=float) - FREQ_LOWER_HZ
            for y in traces
        ]
    return [
        FREQ_UPPER_HZ - np.asarray(y, dtype=float)
        for y in traces
    ]


def _summary_barrier_label(side):
    # Fast-rendering manuscript notation.  The combining low line produces
    # an underbar under h without invoking external LaTeX.
    if side == "lower":
        return "h̲ᵢ [Hz]"
    return "h̄ᵢ [Hz]"


def plot_reduced_vs_detailed_summary_panel(
    emt_root,
    dc_model,
    baseline,
    active,
    disturbances,
    id_map,
):
    """
    Final paper-ready 4 x 2 comparison.

    Left column  : Reduced Model
    Right column : dq0-EMT Model

    Row 1: GFL-bus frequency response.
    Row 2: aggregate nominal GFL response and aggregate safety action.
    Row 3: CBF barrier functions.
    Row 4: SG-bus frequency response.

    Dashed curves denote the GFL safety filter OFF.
    Solid curves denote the GFL safety filter ON.
    """
    side = _active_side(baseline, active)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(FIGURE_WIDTH * 2.10, 8.85),
        sharex="col",
    )

    ls_off = (0, (5.0, 2.6))
    ls_on = "-"
    lw_off = 1.30
    lw_on = 2.45

    # ---------------------------------------------------------------
    # Local helpers
    # ---------------------------------------------------------------
    def common_row_ylim(
        ax_left,
        ax_right,
        arrays,
        include_zero=False,
        lower_pad=0.08,
        upper_pad=0.10,
    ):
        vals = []

        for arr in arrays:
            y = np.asarray(arr, dtype=float).reshape(-1)
            y = y[np.isfinite(y)]
            if y.size:
                vals.append(y)

        if include_zero:
            vals.append(np.array([0.0]))

        if not vals:
            return

        y = np.concatenate(vals)
        lo = float(np.min(y))
        hi = float(np.max(y))
        span = max(hi - lo, 1e-8)

        ylim = (
            lo - lower_pad * span,
            hi + upper_pad * span,
        )

        ax_left.set_ylim(*ylim)
        ax_right.set_ylim(*ylim)

    def bus_handles(buses):
        return [
            Line2D(
                [0],
                [0],
                color=BUS_COLORS[int(bus)],
                linewidth=2.05,
                label=f"Bus {int(bus)}",
            )
            for bus in buses
        ]

    def filter_handles(*, sg_context=False):
        if sg_context:
            off_label = "Without GFL Safety Filter"
            on_label = "With GFL Safety Filter"
        else:
            off_label = "Without Safety Filter"
            on_label = "With Safety Filter"

        return [
            Line2D(
                [0],
                [0],
                color=COLOR_INK,
                linestyle=ls_off,
                linewidth=1.40,
                label=off_label,
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_INK,
                linestyle=ls_on,
                linewidth=2.40,
                label=on_label,
            ),
        ]

    def add_bus_and_filter_legends(
        ax,
        buses,
        bus_title,
        *,
        sg_context=False,
    ):
        # Compact bus legend centered in the quiet upper portion.
        leg_bus = ax.legend(
            handles=bus_handles(buses),
            title=bus_title,
            loc="upper center",
            bbox_to_anchor=(0.405, 0.997),
            ncol=len(buses),
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            fontsize=7.35,
            title_fontsize=7.50,
            handlelength=1.55,
            handletextpad=0.34,
            columnspacing=0.54,
            borderpad=0.18,
            labelspacing=0.15,
            borderaxespad=0.18,
        )
        leg_bus.get_frame().set_facecolor("white")
        leg_bus.get_frame().set_edgecolor("none")
        ax.add_artist(leg_bus)

        # Filter-status legend at the upper right.
        leg_sf = ax.legend(
            handles=filter_handles(
                sg_context=sg_context
            ),
            loc="upper right",
            bbox_to_anchor=(0.998, 0.997),
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            fontsize=7.35,
            handlelength=2.55,
            handletextpad=0.44,
            borderpad=0.18,
            labelspacing=0.14,
            borderaxespad=0.18,
        )
        leg_sf.get_frame().set_facecolor("white")
        leg_sf.get_frame().set_edgecolor("none")

    def plot_bus_overlay(
        ax,
        buses,
        off_getter,
        on_getter,
        ylabel,
        *,
        draw_frequency_bound=False,
    ):
        arrays = []

        for bus in buses:
            color = BUS_COLORS.get(
                int(bus),
                COLOR_INK,
            )

            toff, yoff = off_getter(bus)
            ton, yon = on_getter(bus)

            ax.plot(
                toff,
                yoff,
                color=color,
                linestyle=ls_off,
                linewidth=lw_off,
                zorder=2,
            )

            ax.plot(
                ton,
                yon,
                color=color,
                linestyle=ls_on,
                linewidth=lw_on,
                zorder=3,
            )

            arrays.extend(
                [
                    yoff,
                    yon,
                ]
            )

        if draw_frequency_bound:
            _frequency_bound(
                ax,
                side,
            )

        _event_lines(
            ax,
            disturbances,
        )

        _style_axis(
            ax,
            zero=False,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=FONT_YLABEL,
            labelpad=1.0,
        )

        return arrays

    # ---------------------------------------------------------------
    # Column headings
    # ---------------------------------------------------------------
    axes[0, 0].set_title(
        "Reduced Model",
        fontsize=12.5,
        pad=9.0,
        color=COLOR_INK,
    )

    axes[0, 1].set_title(
        r"$dq0$-EMT Model",
        fontsize=12.5,
        pad=9.0,
        color=COLOR_INK,
    )

    # ===============================================================
    # Row 1: GFL-bus frequency
    # ===============================================================
    gfl_red = plot_bus_overlay(
        axes[0, 0],
        USER_GFL_BUSES_EXT,
        lambda bus: _analytical_gfl_frequency(
            baseline,
            dc_model,
            id_map,
            bus,
        ),
        lambda bus: _analytical_gfl_frequency(
            active,
            dc_model,
            id_map,
            bus,
        ),
        r"$\Delta f_i\,[\mathrm{Hz}]$",
        draw_frequency_bound=True,
    )

    gfl_emt = plot_bus_overlay(
        axes[0, 1],
        USER_GFL_BUSES_EXT,
        lambda bus: _emt_gfl_frequency(
            emt_root,
            bus,
            False,
        ),
        lambda bus: _emt_gfl_frequency(
            emt_root,
            bus,
            True,
        ),
        r"$\Delta f_i\,[\mathrm{Hz}]$",
        draw_frequency_bound=True,
    )

    common_row_ylim(
        axes[0, 0],
        axes[0, 1],
        gfl_red
        + gfl_emt
        + [np.array([FREQ_LOWER_HZ, 0.0])],
        include_zero=True,
        lower_pad=0.060,
        upper_pad=0.050,
    )

    # Frequency-unsafe region.
    for ax in axes[0, :]:
        ylo, yhi = ax.get_ylim()

        if side == "lower":
            ax.axhspan(
                ylo,
                FREQ_LOWER_HZ,
                color=COLOR_UNSAFE_FILL,
                alpha=0.42,
                linewidth=0.0,
                zorder=-4,
            )
        else:
            ax.axhspan(
                FREQ_UPPER_HZ,
                yhi,
                color=COLOR_UNSAFE_FILL,
                alpha=0.42,
                linewidth=0.0,
                zorder=-4,
            )

        add_bus_and_filter_legends(
            ax,
            USER_GFL_BUSES_EXT,
            "GFL buses",
            sg_context=False,
        )

    # ===============================================================
    # Row 2: aggregate nominal GFL response and aggregate safety action
    # ===============================================================
    pnom_r = np.sum(
        active["dp_F_ffr"],
        axis=1,
    )

    zsum_r = np.sum(
        active["z"],
        axis=1,
    )

    t_pnom_e, pnom_e = _emt_aggregate_gfl(
        emt_root,
        True,
        "delta_pF_ffr_system_pu",
    )

    t_zsum_e, zsum_e = _emt_aggregate_gfl(
        emt_root,
        True,
        "z_command_system_pu",
    )

    pnom_line_r, = axes[1, 0].plot(
        tgrid,
        pnom_r,
        color=COLOR_FFR,
        linewidth=1.90,
        zorder=3,
    )

    zsum_line_r, = axes[1, 0].plot(
        tgrid,
        zsum_r,
        color=COLOR_FILTER,
        linewidth=2.15,
        zorder=4,
    )

    pnom_line_e, = axes[1, 1].plot(
        t_pnom_e,
        pnom_e,
        color=COLOR_FFR,
        linewidth=1.90,
        zorder=3,
    )

    zsum_line_e, = axes[1, 1].plot(
        t_zsum_e,
        zsum_e,
        color=COLOR_FILTER,
        linewidth=2.15,
        zorder=4,
    )

    for ax in axes[1, :]:
        _event_lines(
            ax,
            disturbances,
        )

        _style_axis(
            ax,
            zero=True,
        )

        ax.set_ylabel(
            r"$\Delta P_{\mathcal{F}}\,[\mathrm{p.u.}]$",
            fontsize=FONT_YLABEL,
            labelpad=1.0,
        )

    common_row_ylim(
        axes[1, 0],
        axes[1, 1],
        [
            pnom_r,
            zsum_r,
            pnom_e,
            zsum_e,
        ],
        include_zero=True,
        lower_pad=0.08,
        upper_pad=0.13,
    )

    for ax, pnom_line, zsum_line in [
        (
            axes[1, 0],
            pnom_line_r,
            zsum_line_r,
        ),
        (
            axes[1, 1],
            pnom_line_e,
            zsum_line_e,
        ),
    ]:
        leg = ax.legend(
            [
                pnom_line,
                zsum_line,
            ],
            [
                r"$\sum_{i\in\mathcal{F}}\Delta p_i^{\mathrm{nom}}$",
                r"$\sum_{i\in\mathcal{F}} z_i$",
            ],
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            fontsize=8.15,
            handlelength=2.25,
            handletextpad=0.44,
            borderpad=0.20,
            labelspacing=0.16,
            borderaxespad=0.22,
        )

        leg.get_frame().set_facecolor(
            "white"
        )
        leg.get_frame().set_edgecolor(
            "none"
        )

    # ===============================================================
    # Row 3: CBF barrier function
    # ===============================================================
    barrier_red = []
    barrier_emt = []

    for bus in USER_GFL_BUSES_EXT:
        color = BUS_COLORS.get(
            int(bus),
            COLOR_INK,
        )

        # Reduced model
        _, foff_r = _analytical_gfl_frequency(
            baseline,
            dc_model,
            id_map,
            bus,
        )

        _, fon_r = _analytical_gfl_frequency(
            active,
            dc_model,
            id_map,
            bus,
        )

        hoff_r = _summary_barrier(
            [foff_r],
            side,
        )[0]

        hon_r = _summary_barrier(
            [fon_r],
            side,
        )[0]

        axes[2, 0].plot(
            tgrid,
            hoff_r,
            color=color,
            linestyle=ls_off,
            linewidth=lw_off,
            zorder=2,
        )

        axes[2, 0].plot(
            tgrid,
            hon_r,
            color=color,
            linestyle=ls_on,
            linewidth=lw_on,
            zorder=3,
        )

        barrier_red.extend(
            [
                hoff_r,
                hon_r,
            ]
        )

        # dq0-EMT model
        toff_e, foff_e = _emt_gfl_frequency(
            emt_root,
            bus,
            False,
        )

        ton_e, fon_e = _emt_gfl_frequency(
            emt_root,
            bus,
            True,
        )

        hoff_e = _summary_barrier(
            [foff_e],
            side,
        )[0]

        hon_e = _summary_barrier(
            [fon_e],
            side,
        )[0]

        axes[2, 1].plot(
            toff_e,
            hoff_e,
            color=color,
            linestyle=ls_off,
            linewidth=lw_off,
            zorder=2,
        )

        axes[2, 1].plot(
            ton_e,
            hon_e,
            color=color,
            linestyle=ls_on,
            linewidth=lw_on,
            zorder=3,
        )

        barrier_emt.extend(
            [
                hoff_e,
                hon_e,
            ]
        )

    barrier_vals = []

    for y in barrier_red + barrier_emt:
        yy = np.asarray(
            y,
            dtype=float,
        )

        yy = yy[np.isfinite(yy)]

        if yy.size:
            barrier_vals.append(
                yy
            )

    if barrier_vals:
        bb = np.concatenate(
            barrier_vals
        )

        blo = float(
            np.min(bb)
        )

        bhi = float(
            np.max(bb)
        )

        bspan = max(
            bhi - blo,
            1e-3,
        )

        ylo = min(
            blo - 0.08 * bspan,
            -0.015,
        )

        yhi = bhi + 0.09 * bspan

    else:
        ylo, yhi = -0.02, 0.08

    for ax in axes[2, :]:
        ax.set_ylim(
            ylo,
            yhi,
        )

        ax.axhspan(
            ylo,
            0.0,
            color=COLOR_UNSAFE_FILL,
            alpha=0.52,
            linewidth=0.0,
            zorder=-4,
        )

        ax.axhline(
            0.0,
            color=COLOR_LIMIT,
            linestyle=(0, (4.0, 2.2)),
            linewidth=1.0,
            zorder=1,
        )

        _event_lines(
            ax,
            disturbances,
        )

        _style_axis(
            ax,
            zero=False,
        )

        ax.set_ylabel(
            _summary_barrier_label(side),
            fontsize=FONT_YLABEL,
            labelpad=1.0,
        )

        add_bus_and_filter_legends(
            ax,
            USER_GFL_BUSES_EXT,
            "GFL buses",
            sg_context=False,
        )

    def add_sg_support_annotation(ax, text_x, text_y, target_x, target_y):
        ax.annotate(
            "GFL safety action\nimproves SG frequency response",
            xy=(target_x, target_y),
            xytext=(text_x, text_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.1,
            color="#7a7a7a",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#b8b8b8",
                lw=1.0,
                ls=(0, (1.1, 1.5)),
                shrinkA=0.0,
                shrinkB=2.0,
                mutation_scale=10.0,
            ),
            bbox=dict(
                boxstyle="round,pad=0.16",
                fc="white",
                ec="none",
                alpha=0.82,
            ),
            zorder=5,
        )

    # ===============================================================
    # Row 4: SG-bus frequency
    # ===============================================================
    sg_red = plot_bus_overlay(
        axes[3, 0],
        USER_SG_BUSES_EXT,
        lambda bus: _analytical_sg_frequency(
            baseline,
            dc_model,
            id_map,
            bus,
        ),
        lambda bus: _analytical_sg_frequency(
            active,
            dc_model,
            id_map,
            bus,
        ),
        r"$\Delta f_i\,[\mathrm{Hz}]$",
        draw_frequency_bound=False,
    )

    sg_emt = plot_bus_overlay(
        axes[3, 1],
        USER_SG_BUSES_EXT,
        lambda bus: _emt_sg_frequency(
            emt_root,
            bus,
            False,
        ),
        lambda bus: _emt_sg_frequency(
            emt_root,
            bus,
            True,
        ),
        r"$\Delta f_i\,[\mathrm{Hz}]$",
        draw_frequency_bound=False,
    )

    common_row_ylim(
        axes[3, 0],
        axes[3, 1],
        sg_red + sg_emt,
        include_zero=True,
        lower_pad=0.070,
        upper_pad=0.055,
    )

    axes[3, 0].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=3.5,
    )

    axes[3, 1].set_xlabel(
        r"$t\,[\mathrm{s}]$",
        fontsize=FONT_XLABEL,
        labelpad=3.5,
    )

    # The safety filter acts at GFL buses, so make that explicit
    # when showing its network-level effect on SG frequencies.
    for ax in axes[3, :]:
        add_bus_and_filter_legends(
            ax,
            USER_SG_BUSES_EXT,
            "SG buses",
            sg_context=True,
        )

    # Annotation highlighting the network-wide benefit of the GFL safety action.
    add_sg_support_annotation(
        axes[3, 0],
        text_x=1.63,
        text_y=-0.110,
        target_x=0.86,
        target_y=-0.051,
    )

    add_sg_support_annotation(
        axes[3, 1],
        text_x=1.82,
        text_y=-0.104,
        target_x=0.96,
        target_y=-0.044,
    )

    # ---------------------------------------------------------------
    # Final alignment and spacing
    # ---------------------------------------------------------------
    for row in range(3):
        axes[row, 0].tick_params(
            labelbottom=False
        )

        axes[row, 1].tick_params(
            labelbottom=False
        )

    # Perfectly aligned y-axis labels in all eight panels.
    for ax in axes.flat:
        ax.yaxis.set_label_coords(
            -0.076,
            0.50,
        )

    fig.align_ylabels(
        axes[:, 0]
    )

    fig.align_ylabels(
        axes[:, 1]
    )

    fig.subplots_adjust(
        left=0.074,
        right=0.993,
        bottom=0.070,
        top=0.950,
        hspace=0.090,
        wspace=0.140,
    )

    save_figure(
        fig,
        "reduced_vs_dq0_emt_model_4x2_summary",
    )

def make_emt_validation_overlays(
    dc_model,
    baseline,
    active,
    disturbances,
    id_map,
):
    try:
        emt_root = _discover_emt_csv_root()
    except Exception as exc:
        warnings.warn(
            "dq0-EMT validation overlays were skipped because the Julia CSV "
            f"folder could not be resolved: {exc}"
        )
        return

    complete, missing = _emt_available(emt_root)
    if not complete:
        warnings.warn(
            "dq0-EMT validation overlays were skipped because required CSV "
            "files are missing:\n" + "\n".join(f"  {p}" for p in missing)
        )
        return

    print(f"\n[EMT] Using Julia CSV root: {emt_root}")
    _report_emt_parameter_match(emt_root)

    if MAKE_EMT_FREQUENCY_OVERLAY:
        plot_analytical_vs_emt_frequency(
            emt_root,
            dc_model,
            baseline,
            active,
            disturbances,
            id_map,
        )

    if ENABLE_SAFETY_FILTER and MAKE_EMT_FILTER_STATE_OVERLAY:
        plot_analytical_vs_emt_filter_state(
            emt_root,
            dc_model,
            active,
            disturbances,
            id_map,
        )

    if ENABLE_SAFETY_FILTER and MAKE_EMT_GFL_POWER_OVERLAY:
        plot_analytical_vs_emt_gfl_power(
            emt_root,
            active,
            disturbances,
        )

    if MAKE_EMT_FREQUENCY_ERROR_FIGURE:
        plot_analytical_vs_emt_frequency_error(
            emt_root,
            dc_model,
            baseline,
            active,
            disturbances,
            id_map,
        )

    if MAKE_EMT_SUMMARY_PANEL:
        plot_reduced_vs_detailed_summary_panel(
            emt_root,
            dc_model,
            baseline,
            active,
            disturbances,
            id_map,
        )

def print_safety_summary(dc_model, baseline, active, inv_map):
    side = _active_side(baseline, active)
    metric = np.min if side == "lower" else np.max
    print("\nSafety summary (GFL buses)")
    print("Bus    filter OFF [Hz]    filter ON [Hz]    peak |z_i| [pu]")
    for j, b in enumerate(dc_model["GFL_buses"]):
        ext = int(inv_map.get(int(b), int(b)))
        f0 = F_BASE * baseline["omega_F"][:, j]
        f1 = F_BASE * active["omega_F"][:, j]
        print(f"{ext:>3d}    {metric(f0):>+14.4f}    {metric(f1):>+13.4f}    {np.max(np.abs(active['z'][:, j])):>13.5f}")


# Main

def main():
    configure_matplotlib()
    ensure_dir(CSV_EXPORT_DIR)
    ensure_dir(FIGURE_EXPORT_DIR)

    mode = str(SIMULATION_MODE).strip().upper()
    if mode not in {"DC", "AC"}:
        raise ValueError("SIMULATION_MODE must be 'DC' or 'AC'.")

    want_freq = any(str(p).strip().lower() == "freq" for p in PLOT_PANELS)
    want_angle = any(str(p).strip().lower() == "angle" for p in PLOT_PANELS)
    want_pg = any(str(p).strip().lower() == "pg" for p in PLOT_PANELS)
    want_dpf = any(str(p).strip().lower() == "dpf" for p in PLOT_PANELS)

    SG_ext, GFM_ext, GFL_ext = build_partitions(
        USER_SG_BUSES_EXT, USER_GFM_BUSES_EXT, USER_GFL_BUSES_EXT, slack_ext=SLACK_BUS_EXT)

    mpc_ext = load_case_from_py()
    (mpc_base, id_map, inv_map, N, slack_bus,
     Vmag_ref, theta_ref, Ybus, P_ref, Q_ref, u0_base, pf_results) = run_pf_init(mpc_ext)
    baseMVA = float(mpc_base.get("baseMVA", SYS_BASE_MVA))

    if EXPORT_PF_CSV:
        export_pf_solution_csvs(CSV_EXPORT_DIR, pf_results, inv_map)

    dc_model = build_dc_model_with_internals(
        N, slack_bus, Ybus, mpc_base, id_map, inv_map, SG_ext, GFM_ext, GFL_ext, P_ref)

    tripped = []
    if GEN_TRIP_ENABLE:
        for b_ext in GEN_TRIP_BUSES_EXT:
            if b_ext in id_map and id_map[b_ext] in dc_model["G_buses"]:
                tripped.append(dc_model["G_buses"].index(id_map[b_ext]))
    dc_model["tripped_G_positions"] = tripped
    dc_model["gen_trip_event_idx"] = event_time_to_idx(GEN_TRIP_EVENT_TIME_S) if tripped else None

    design = _filter_design(dc_model, inv_map)
    if design["eig"].size:
        print(
            f"Filter design: lambda_min(Gamma)={design['eig'][0]:.3e}, "
            f"lambda_max(Gamma)={design['eig'][-1]:.3e}, "
            f"lambda_max(K^1/2 Gamma K^1/2)={design['weighted_lmax']:.4f}"
        )

    ac_help = None
    if mode == "AC":
        ac_help = build_ac_dae_helpers(N, Ybus, Vmag_ref, theta_ref, u0_base, dc_model)
        ac_help["tripped_G_positions"] = tripped
        ac_help["gen_trip_event_idx"] = dc_model["gen_trip_event_idx"]

    disturbances = build_fixed_disturbances(
        id_map, pf_results, baseMVA, dc_model["m_inter"], dc_model["nI"], dc_model["nG"])
    print(f"Simulation: mode={mode}, SG={SG_ext}, GFL={GFL_ext}, events={len(disturbances)}")

    t0 = time.perf_counter()
    if mode == "AC":
        baseline = simulate_ac(ac_help, dc_model, disturbances, design, filter_enabled=False)
        active = simulate_ac(ac_help, dc_model, disturbances, design,
                             filter_enabled=ENABLE_SAFETY_FILTER) if ENABLE_SAFETY_FILTER else baseline
    else:
        baseline = simulate_dc(dc_model, disturbances, design, filter_enabled=False)
        active = simulate_dc(dc_model, disturbances, design,
                             filter_enabled=ENABLE_SAFETY_FILTER) if ENABLE_SAFETY_FILTER else baseline
    elapsed = time.perf_counter() - t0

    plot_buses_int = [id_map.get(int(b), int(b)) for b in PLOT_BUSES_EXT]
    if EXPORT_TRAJ_CSV:
        export_trajectory_csvs(
            baseline, dc_model, plot_buses_int, inv_map, f"{mode.lower()}_filter_off",
            want_freq, want_angle, want_pg, want_dpf)
        if ENABLE_SAFETY_FILTER:
            export_trajectory_csvs(
                active, dc_model, plot_buses_int, inv_map, f"{mode.lower()}_filter_on",
                want_freq, want_angle, want_pg, want_dpf)

    print_safety_summary(dc_model, baseline, active, inv_map)

    if MAKE_FREQUENCY_SAFETY_FIGURE:
        plot_frequency_safety(dc_model, baseline, active, disturbances, plot_buses_int, inv_map)
    if MAKE_BARRIER_FIGURE:
        plot_barrier_safety(dc_model, baseline, active, disturbances, inv_map)
    if ENABLE_SAFETY_FILTER and MAKE_FILTER_STATE_FIGURE:
        plot_filter_state(active, dc_model, disturbances, inv_map)
    if ENABLE_SAFETY_FILTER and MAKE_GFL_POWER_DECOMPOSITION_FIGURE:
        plot_gfl_power_decomposition(active, disturbances)

    make_emt_validation_overlays(
        dc_model,
        baseline,
        active,
        disturbances,
        id_map,
    )

    print(f"Completed in {elapsed:.3f} s")
    print(f"Figures: {FIGURE_EXPORT_DIR}")
    print(f"CSVs: {CSV_EXPORT_DIR}")


if __name__ == "__main__":
    main()

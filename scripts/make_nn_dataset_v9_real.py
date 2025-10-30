#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_nn_dataset_v9_real.py  — real logs → v9 dataset (trainer-compatible)

Key points:
 • Targets vE,vN,vD are written at 100 Hz via **linear interpolation** between SeaPath samples
   (no ZOH bias; no gaps; no extrapolation beyond overlap).
 • Inputs exactly as nn_observer_v9.py expects:
     IN_COLS  = [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]
     OUT_COLS = [vE, vN, vD]
     AUX      = [w_est_x, w_est_y, w_est_z]  (from quaternion observer)
 • Quaternion observer:
     - step6DOF (IMU-only) every tick
     - step7DOF heading injection when a fresh SeaPath heading sample is available
     - Gains mirror your C++ Config: k1 (acc), k2 (heading), Ki (bias diag), accel_min_norm
 • τ model matches RAN::tau_pods exactly with descending thrust polynomial (no constant term).

Typical usage:
python3 scripts/make_nn_dataset_v9_real.py \
  --root MarineVesselSimulator/data/nn_dataset_v9_TestData3 \
  --indices 1 2 3 4 5 6 7 \
  --out_dir MarineVesselSimulator/data/nn_dataset_v9_real \
  --val_frac 0.15 --test_frac 0.15 \
  --k1 1.0 --k2 0.5 --Ki 0.001 --accel_min_norm 1e-3 \
  --alpha_units rad \
  --lx_o -1.17 --ly1_o -0.79 --ly2_o 0.79 --pod_radius 0.2 \
  --tau_coeffs -312.547 8.87016 413.598 46.922 45.6015
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

# ===================== Columns expected by nn_observer_v9.py =====================
IN_COLS     = ["ax","ay","az","qw","qx","qy","qz","tau_x","tau_y","tau_n"]
OUT_COLS    = ["vE","vN","vD"]
AUX_W_COLS  = ["w_est_x","w_est_y","w_est_z"]  # optional but we provide them

# ===================== Time helpers =====================

def parse_iso_to_epoch(s: str) -> float:
    if not isinstance(s, str): return np.nan
    ss = s.strip()
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    try:
        return pd.Timestamp(ss).timestamp()
    except Exception:
        return np.nan

def zoh_series(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Zero-Order Hold. NaN before first sample."""
    if len(t_src) == 0:
        return np.full_like(t_grid, np.nan, dtype=float)
    idx = np.searchsorted(t_src, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(t_src)-1)
    y = y_src[idx].astype(float, copy=False)
    y = y.copy()
    y[t_grid < t_src[0]] = np.nan
    return y

def linear_series(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation with NaN outside the source span."""
    if len(t_src) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    mask = np.isfinite(t_src) & np.isfinite(y_src)
    if mask.sum() < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    return np.interp(t_grid, t_src[mask], y_src[mask], left=np.nan, right=np.nan)

# ===================== Files I/O =====================

def load_parsed_imu(path: str):
    """
    Headers (example):
      time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
    Returns epoch time and numeric arrays. gyro is rad/s (no deg conv).
    """
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    t_name  = pick("time","timestamp") or df.columns[0]
    ax_name = pick("ax","acc_x","accx","accel_x")
    ay_name = pick("ay","acc_y","accy","accel_y")
    az_name = pick("az","acc_z","accz","accel_z")
    gx_name = pick("gx","gyro_x","wx","omega_x")
    gy_name = pick("gy","gyro_y","wy","omega_y")
    gz_name = pick("gz","gyro_z","wz","omega_z")

    if None in (ax_name,ay_name,az_name,gx_name,gy_name,gz_name):
        raise KeyError(f"{path}: IMU header mismatch. Got {list(df.columns)}")

    t  = df[t_name].astype(str).map(parse_iso_to_epoch).to_numpy()
    ax = pd.to_numeric(df[ax_name], errors="coerce").to_numpy()
    ay = pd.to_numeric(df[ay_name], errors="coerce").to_numpy()
    az = pd.to_numeric(df[az_name], errors="coerce").to_numpy()
    gx = pd.to_numeric(df[gx_name], errors="coerce").to_numpy()
    gy = pd.to_numeric(df[gy_name], errors="coerce").to_numpy()
    gz = pd.to_numeric(df[gz_name], errors="coerce").to_numpy()
    return t, ax, ay, az, gx, gy, gz

def load_parsed_seapath(path: str):
    """
    Headers (from your example):
      timestamp, roll, pitch, heading, heave, roll_rate, pitch_rate, yaw_rate,
      velocity_north, velocity_east, velocity_down
    Returns: t, heading(rad), vE, vN, vD
    """
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    t  = df[cols.get("timestamp", df.columns[0])].astype(str).map(parse_iso_to_epoch).to_numpy()
    hd = pd.to_numeric(df[cols["heading"]], errors="coerce").to_numpy() if "heading" in cols \
         else np.full_like(t, np.nan, dtype=float)
    vN = pd.to_numeric(df[cols["velocity_north"]], errors="coerce").to_numpy() if "velocity_north" in cols \
         else np.full_like(t, np.nan, dtype=float)
    vE = pd.to_numeric(df[cols["velocity_east"]],  errors="coerce").to_numpy() if "velocity_east"  in cols \
         else np.full_like(t, np.nan, dtype=float)
    vD = pd.to_numeric(df[cols["velocity_down"]],  errors="coerce").to_numpy() if "velocity_down"  in cols \
         else np.full_like(t, np.nan, dtype=float)
    return t, np.deg2rad(hd), vE, vN, vD

def load_parsed_alog(path: str, alpha_units: str):
    """
    Headers:
      timestamp,N1,N2,alpha1,alpha2
    Returns: t_epoch, N1, N2, alpha1(rad), alpha2(rad)
    """
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    t  = df[cols.get("timestamp", df.columns[0])].astype(str).map(parse_iso_to_epoch).to_numpy()
    N1 = pd.to_numeric(df[cols["n1"]], errors="coerce").to_numpy()
    N2 = pd.to_numeric(df[cols["n2"]], errors="coerce").to_numpy()
    a1 = pd.to_numeric(df[cols["alpha1"]], errors="coerce").to_numpy()
    a2 = pd.to_numeric(df[cols["alpha2"]], errors="coerce").to_numpy()
    if alpha_units.lower().startswith("deg"):
        a1 = np.deg2rad(a1); a2 = np.deg2rad(a2)
    return t, N1, N2, a1, a2

# ===================== BODY→END rotation (your convention) =====================

def rnb_from_quat(q: np.ndarray) -> np.ndarray:
    """Custom END rotation used across your codebase."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.empty((q.shape[0], 3, 3), dtype=q.dtype)
    # Row East
    R[:,0,0] =  2.0*(xy + wz)
    R[:,0,1] =  1.0 - 2.0*(xx + zz)
    R[:,0,2] =  2.0*(yz - wx)
    # Row North
    R[:,1,0] =  1.0 - 2.0*(yy + zz)
    R[:,1,1] =  2.0*(xy - wz)
    R[:,1,2] =  2.0*(xz + wy)
    # Row Down
    R[:,2,0] =  2.0*(xz - wy)
    R[:,2,1] =  2.0*(yz + wx)
    R[:,2,2] =  1.0 - 2.0*(xx + yy)
    return R

# ===================== Quaternion math & observer =====================

def quat_mul(q, r):
    w1,x1,y1,z1 = q; w2,x2,y2,z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

def heading_sigma_body(q: np.ndarray, yaw_meas: float, k_yaw: float) -> np.ndarray:
    """Heading injection error in BODY (END heading 0°=North, +CW)."""
    if not np.isfinite(yaw_meas): return np.zeros(3)
    Rnb = rnb_from_quat(q[np.newaxis,:])[0]
    # predicted heading axis in NAV (unit vector in horizontal plane)
    h_pred = np.array([Rnb[0,0], Rnb[1,0], 0.0])
    nrm = np.linalg.norm(h_pred)
    if nrm < 1e-12: return np.zeros(3)
    h_pred /= nrm
    # measured heading unit vector in NAV (E,N,0): [sin(psi), cos(psi), 0]
    h_meas = np.array([np.sin(yaw_meas), np.cos(yaw_meas), 0.0])
    # error in NAV then bring to BODY
    e_nav  = np.cross(h_meas, h_pred)
    e_body = Rnb.T @ e_nav
    return k_yaw * e_body

def accel_sigma_body(q: np.ndarray, acc_b: np.ndarray, k_acc: float, accel_min_norm: float) -> np.ndarray:
    """Accelerometer (gravity direction) injection in BODY."""
    if k_acc <= 0.0 or not np.isfinite(acc_b).all(): return np.zeros(3)
    n = np.linalg.norm(acc_b)
    if n < accel_min_norm: return np.zeros(3)
    a_hat = acc_b / n
    Rnb = rnb_from_quat(q[np.newaxis,:])[0]
    d_b = Rnb[2,:]  # Down axis in BODY
    return k_acc * np.cross(a_hat, d_b)

def run_quat_observer(t_grid, ax, ay, az, gx, gy, gz, yaw_meas, fresh_yaw, k1, k2, Ki, accel_min_norm):
    """step6DOF each tick + step7DOF yaw correction when fresh."""
    q = np.array([1.0,0.0,0.0,0.0], dtype=float)
    b = np.zeros(3, dtype=float)
    dt = float(t_grid[1]-t_grid[0]) if len(t_grid) > 1 else 0.01
    qw = np.empty_like(t_grid); qx = np.empty_like(t_grid); qy = np.empty_like(t_grid); qz = np.empty_like(t_grid)
    wx = np.empty_like(t_grid); wy = np.empty_like(t_grid); wz = np.empty_like(t_grid)
    for k in range(len(t_grid)):
        omega = np.array([gx[k], gy[k], gz[k]], dtype=float)  # rad/s
        # IMU step (accel-based)
        sig = accel_sigma_body(q, np.array([ax[k],ay[k],az[k]]), k1, accel_min_norm)
        # yaw correction when fresh SeaPath heading is available
        if fresh_yaw[k] and np.isfinite(yaw_meas[k]):
            sig = sig + heading_sigma_body(q, float(yaw_meas[k]), k2)
        # integrate quaternion with bias correction
        w_est = omega - b + sig
        dq = 0.5 * quat_mul(q, np.array([0.0, *w_est]))
        q = q + dq * dt
        q = q / (np.linalg.norm(q) + 1e-12)
        b = b - Ki * sig * dt
        # outputs
        qw[k],qx[k],qy[k],qz[k] = q
        wx[k],wy[k],wz[k] = w_est
    return qw,qx,qy,qz, wx,wy,wz

# ===================== τ model (exact port of RAN::tau_pods) =====================

def thrusts_from_relative_n(n1: float, n2: float, coeffs: List[float]) -> Tuple[float,float]:
    """T(N) = c5*N^5 + c4*N^4 + c3*N^3 + c2*N^2 + c1*N   (descending, NO constant)"""
    def poly_no_const(N, cs):
        s = 0.0; deg = len(cs)
        # cs is descending: [c5, c4, c3, c2, c1]
        for p in range(1, deg+1):            # N^1..N^deg
            s += cs[deg - p] * (N ** p)      # maps c1..c5 to N^1..N^5
        return s
    return poly_no_const(float(n1), coeffs), poly_no_const(float(n2), coeffs)

def tau_from_thrusters(n1,n2,a1,a2, coeffs, lx_o, ly1_o, ly2_o, pod_radius, n1_fail=False, n2_fail=False):
    if n1_fail: n1 = 0.0
    if n2_fail: n2 = 0.0
    lx1 = lx_o  - pod_radius*np.cos(a1)
    lx2 = lx_o  - pod_radius*np.cos(a2)
    ly1 = ly1_o - pod_radius*np.sin(a1)
    ly2 = ly2_o - pod_radius*np.sin(a2)
    T1,T2 = thrusts_from_relative_n(n1,n2,coeffs)
    c1,s1 = np.cos(a1),np.sin(a1)
    c2,s2 = np.cos(a2),np.sin(a2)
    tau_x = T1*c1 + T2*c2
    tau_y = T1*s1 + T2*s2
    tau_n = lx1*T1*s1 + lx2*T2*s2 + ly1*T1*c1 + ly2*T2*c2
    return float(tau_x), float(tau_y), float(tau_n)

# ===================== Set builder =====================

def build_from_set(root: str, idx: int, dt: float, params: dict) -> pd.DataFrame:
    imu_path = os.path.join(root, f"parsedIMU_{idx}.csv")
    sp_path  = os.path.join(root, f"parsedSeapath_{idx}.csv")
    al_path  = os.path.join(root, f"parsedAlog_{idx}.csv")
    if not (os.path.isfile(imu_path) and os.path.isfile(sp_path) and os.path.isfile(al_path)):
        raise FileNotFoundError(f"Missing one of: {imu_path} | {sp_path} | {al_path}")

    # Load raw series
    t_i, ax, ay, az, gx, gy, gz = load_parsed_imu(imu_path)
    t_s, hdg, vE_s, vN_s, vD_s = load_parsed_seapath(sp_path)
    t_a, N1, N2, a1, a2        = load_parsed_alog(al_path, alpha_units=params["alpha_units"])

    # Time grid over the intersection of ALL THREE sources
    t0 = max(np.nanmin(t_i), np.nanmin(t_s), np.nanmin(t_a))
    t1 = min(np.nanmax(t_i), np.nanmax(t_s), np.nanmax(t_a))
    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 - t0) < 2*dt:
        raise RuntimeError(f"Insufficient overlap for idx={idx}")
    steps = int(np.floor((t1 - t0)/dt)) + 1  # closed interval [t0, t1]
    t_grid = (t0 + np.arange(steps)*dt).astype(float)

    # Resample IMU to grid (linear)
    ax_g = linear_series(t_i, ax, t_grid)
    ay_g = linear_series(t_i, ay, t_grid)
    az_g = linear_series(t_i, az, t_grid)
    gx_g = linear_series(t_i, gx, t_grid)
    gy_g = linear_series(t_i, gy, t_grid)
    gz_g = linear_series(t_i, gz, t_grid)

    # Map SeaPath heading samples to nearest grid slot within ±dt/2
    yaw_meas = np.full_like(t_grid, np.nan, dtype=float)
    fresh    = np.zeros_like(t_grid, dtype=bool)
    if len(t_s):
        k_idx = np.clip(np.round((t_s - t0)/dt).astype(int), 0, len(t_grid)-1)
        yaw_meas[k_idx] = hdg
        fresh[k_idx] = np.isfinite(hdg)

    # Quaternion observer (step6DOF each tick, yaw correction on fresh SeaPath)
    qw,qx,qy,qz, wex,wey,wez = run_quat_observer(
        t_grid, ax_g, ay_g, az_g, gx_g, gy_g, gz_g,
        yaw_meas, fresh,
        k1=params["k1"], k2=params["k2"], Ki=params["Ki"], accel_min_norm=params["accel_min_norm"]
    )

    # Resample ALOG (ZOH), compute τ each grid step
    N1_g = zoh_series(t_a, N1, t_grid)
    N2_g = zoh_series(t_a, N2, t_grid)
    a1_g = zoh_series(t_a, a1, t_grid)
    a2_g = zoh_series(t_a, a2, t_grid)

    tau_x = np.empty_like(t_grid); tau_y = np.empty_like(t_grid); tau_n = np.empty_like(t_grid)
    for i in range(len(t_grid)):
        tx,ty,tn = tau_from_thrusters(
            N1_g[i], N2_g[i], a1_g[i], a2_g[i],
            params["tau_coeffs"], params["lx_o"], params["ly1_o"], params["ly2_o"], params["pod_radius"]
        )
        tau_x[i], tau_y[i], tau_n[i] = tx,ty,tn

    # Ground-truth velocity at 100 Hz: **LINEAR interpolation** from SeaPath
    vE = linear_series(t_s, vE_s, t_grid)
    vN = linear_series(t_s, vN_s, t_grid)
    vD = linear_series(t_s, vD_s, t_grid)

    # Trim off any head rows before first valid interpolation
    first_ok = np.where(np.isfinite(vE) & np.isfinite(vN) & np.isfinite(vD))[0]
    if len(first_ok) == 0:
        raise RuntimeError(f"No valid SeaPath in overlap for idx={idx}")
    start = first_ok[0]
    sl = slice(start, None)

    df = pd.DataFrame({
        "ax": ax_g[sl], "ay": ay_g[sl], "az": az_g[sl],
        "qw": qw[sl], "qx": qx[sl], "qy": qy[sl], "qz": qz[sl],
        "tau_x": tau_x[sl], "tau_y": tau_y[sl], "tau_n": tau_n[sl],
        "vE": vE[sl], "vN": vN[sl], "vD": vD[sl],
        "w_est_x": wex[sl], "w_est_y": wey[sl], "w_est_z": wez[sl],
    })[IN_COLS + OUT_COLS + AUX_W_COLS]

    # Clean any residual NaNs (should be none after trimming)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

# ===================== Split & stats =====================

def three_way_indices(total: int, val_frac: float, test_frac: float):
    v = int(round(val_frac * total))
    t = int(round(test_frac * total))
    v = max(0, min(total, v))
    t = max(0, min(total - v, t))
    tr = total - v - t
    if tr <= 0:
        raise ValueError("Fractions leave no room for training data.")
    return tr, v, t

def compute_norm_stats(train_df: pd.DataFrame) -> dict:
    x = train_df[IN_COLS].to_numpy(np.float64)
    y = train_df[OUT_COLS].to_numpy(np.float64)
    return {
        "x_mean": x.mean(axis=0).tolist(),
        "x_std":  (x.std(axis=0, ddof=0) + 1e-12).tolist(),
        "y_mean": y.mean(axis=0).tolist(),
        "y_std":  (y.std(axis=0, ddof=0) + 1e-12).tolist(),
    }

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Build v9-ready dataset (real logs, 100 Hz, SeaPath GT via linear interpolation).")
    ap.add_argument("--root", required=True, help="Folder containing parsedIMU_X.csv, parsedSeapath_X.csv, parsedAlog_X.csv")
    ap.add_argument("--indices", nargs="+", type=int, required=True, help="Which X indices to include (e.g., 1 2 3)")
    ap.add_argument("--out_dir", required=True, help="Where to write train/val/test + stats")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--dt", type=float, default=0.01, help="Grid period (s) — keep 0.01 for trainer")
    # quat observer params (mirror your Config)
    ap.add_argument("--k1", type=float, default=1.0, help="Accel injection gain")
    ap.add_argument("--k2", type=float, default=0.5, help="Heading injection gain")
    ap.add_argument("--Ki", type=float, default=1e-3, help="Gyro-bias integral (diag)")
    ap.add_argument("--accel_min_norm", type=float, default=1e-3)
    # tau model params
    ap.add_argument("--alpha_units", choices=["rad","deg"], default="rad")
    ap.add_argument("--lx_o", type=float, default=-1.17)
    ap.add_argument("--ly1_o", type=float, default=-0.79)
    ap.add_argument("--ly2_o", type=float, default=0.79)
    ap.add_argument("--pod_radius", type=float, default=0.2)
    ap.add_argument("--tau_coeffs", nargs="+", type=float,
                    default=[-312.547, 8.87016, 413.598, 46.922, 45.6015],
                    help="Descending powers (no constant term)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    params = {
        "k1": args.k1, "k2": args.k2, "Ki": args.Ki, "accel_min_norm": args.accel_min_norm,
        "alpha_units": args.alpha_units,
        "lx_o": args.lx_o, "ly1_o": args.ly1_o, "ly2_o": args.ly2_o, "pod_radius": args.pod_radius,
        "tau_coeffs": list(args.tau_coeffs),
    }

    # Build each set, concat chronologically
    dfs = []
    for idx in args.indices:
        print(f"[build] set {idx}…")
        df = build_from_set(args.root, idx, args.dt, params)
        if len(df) == 0:
            print(f"[warn] set {idx} produced 0 rows (skipped)")
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No data produced. Check inputs/indices.")
    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Chronological split by rows
    N = len(all_df)
    n_tr, n_va, n_te = three_way_indices(N, args.val_frac, args.test_frac)
    train_df = all_df.iloc[:n_tr].reset_index(drop=True)
    val_df   = all_df.iloc[n_tr:n_tr+n_va].reset_index(drop=True)
    test_df  = all_df.iloc[n_tr+n_va:].reset_index(drop=True)

    # Save CSVs
    train_csv = os.path.join(args.out_dir, "train.csv")
    val_csv   = os.path.join(args.out_dir, "val.csv")
    test_csv  = os.path.join(args.out_dir, "test.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Norm stats from TRAIN only
    stats = compute_norm_stats(train_df)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Split summary
    with open(os.path.join(args.out_dir, "split_summary.json"), "w") as f:
        json.dump({
            "counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "fractions": {"val_frac": args.val_frac, "test_frac": args.test_frac,
                          "train_frac": 1.0 - args.val_frac - args.test_frac},
            "dt": args.dt, "indices": args.indices
        }, f, indent=2)

    print(f"[done] wrote:\n  {train_csv} ({len(train_df)} rows)\n  {val_csv} ({len(val_df)} rows)\n  {test_csv} ({len(test_df)} rows)")
    print(f"[done] wrote: {os.path.join(args.out_dir, 'norm_stats.json')}")
    print("[note] Ground truth vE/vN/vD is linear-interpolated at 100 Hz (no ZOH).")

if __name__ == "__main__":
    main()

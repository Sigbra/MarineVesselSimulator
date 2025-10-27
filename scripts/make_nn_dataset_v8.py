#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_v8.py  (window-aware, time-series safe)

Builds train/val CSVs + normalization stats for the v8 observer.

Inputs (10): ax, ay, az, qw, qx, qy, qz, u_prev, v_prev, w_prev
  - ax..az come from SIMDATA cols 46..48 (imu.accel)
  - qw..qz come from SIMDATA cols 52..55 (q_nb: Body→END)
  - u_prev,v_prev,w_prev are BODY linear velocities from the **previous** step
    (constructed as a 1-step shift of u,v,w within each contiguous sequence)

Outputs (3): u, v, w  (BODY frame; SIMDATA cols 1..3)

Options:
- Chronological split (default) preserves time order.
- Optional window-aware shuffle reorders intact windows (stride=seq) safely.
- Optional trimming removes trailing duplicated rows from each SIM file.
- Optional quaternion sign continuity fix (on via --fix_q_sign).

Example
-------
python3 scripts/make_dataset_v8.py \
  --in data/simdata.csv \
  --out_dir data/nn_dataset_v8_X_C0 \
  --split 0.8 \
  --header none \
  --trim_tail \
  --seq 200 --shuffle_windows --seed 123 \
  --fix_q_sign
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

# ----- Column mapping in SIMDATA (0-based) -----
# 0: t
# 1..3:   u, v, w                 (BODY linear vel)          <-- outputs; also used to derive prev
# 4..6:   p, q, r                 (BODY rates)               [NOT USED]
# 7..9:   x, y, z                 (END position)
# 10..12: phi, theta, psi         (END Euler)
# 13..15: xn_d, yn_d, psi_d       [NOT USED]
# 16..19: n_c(0), n_c(1), n(0), n(1)                         [NOT USED]
# 20..23: alpha_c(0), alpha_c(1), alpha(0), alpha(1)         [NOT USED]
# 24..26: tau_XYN_c[0..2]                                    [NOT USED]
# 27..29: tau_XYN[0..2]                                      [NOT USED]
# 30..33: closest.point.pos.x, closest.point.pos.y, x_e, y_e [NOT USED]
# 34..45: x_est (12)              [NOT USED]
# 46..48: imu.accel (ax, ay, az)  (BODY specific force)      <-- input
# 49..51: imu.gyro  (wx, wy, wz)  (BODY rates)               [NOT USED]
# 52..55: q_nb (qw, qx, qy, qz)   (BODY→END quaternion)      <-- input

AX_COLS   = [46, 47, 48]
Q_COLS    = [52, 53, 54, 55]
UVW_COLS  = [1, 2, 3]

# Columns used to detect stationary tails
TAIL_COLS = UVW_COLS + [10,11,12] + AX_COLS + [49,50,51] + Q_COLS

IN_COLS  = ["ax","ay","az","qw","qx","qy","qz","u_prev","v_prev","w_prev"]
OUT_COLS = ["u","v","w"]

# ---------- Quaternion helpers ----------

def quat_normalize_np(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n

def fix_quaternion_sign_continuity(q: np.ndarray) -> np.ndarray:
    """
    Flip sign when dot(q[i], q[i-1])<0 to avoid q↔-q jumps (same rotation).
    """
    if len(q) == 0:
        return q
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i-1], out[i]) < 0.0:
            out[i] = -out[i]
    return out

# ---------- Tail trimming ----------

def trim_trailing_stationary_rows_np(arr: np.ndarray,
                                     cols: List[int],
                                     tol: float = 0.0,
                                     keep_one: bool = True) -> Tuple[np.ndarray, int]:
    N = arr.shape[0]
    if N == 0: return arr, 0
    max_col = max(cols) if cols else -1
    if arr.shape[1] <= max_col:
        return arr, 0
    last = arr[-1, cols]
    diffs = np.abs(arr[:, cols] - last)
    eq_mask = (diffs <= tol).all(axis=1)
    if np.all(eq_mask):
        new_len = 1 if keep_one else 0
    else:
        last_change = np.where(~eq_mask)[0][-1]
        new_len = last_change + (2 if keep_one else 1)
    new_len = max(0, min(N, new_len))
    return arr[:new_len], (N - new_len)

# ---------- Build rows from one SIMDATA file (chronological) ----------

def build_rows_from_one_file(arr: np.ndarray,
                             fix_q_sign: bool = True) -> pd.DataFrame:
    """
    Build a frame with inputs+outputs for a **single** SIMDATA file.
    Prev-body-vel is computed as a 1-step shift inside this file.
    """
    need_cols = max(AX_COLS + Q_COLS + UVW_COLS) + 1
    if arr.shape[1] < need_cols:
        raise ValueError(f"Expected at least {need_cols} columns; got {arr.shape[1]}.")

    ax   = arr[:, AX_COLS].astype(np.float64)      # [N,3]
    q    = arr[:, Q_COLS].astype(np.float64)       # [N,4]
    uvw  = arr[:, UVW_COLS].astype(np.float64)     # [N,3]

    # Normalize quaternion; optionally enforce sign continuity
    q = quat_normalize_np(q)
    if fix_q_sign:
        q = fix_quaternion_sign_continuity(q)

    # Outputs = BODY velocity
    u, v, w = uvw[:,0], uvw[:,1], uvw[:,2]

    # Prev within this file: shift by 1, fill first with itself (teacher forcing)
    u_prev = np.empty_like(u); v_prev = np.empty_like(v); w_prev = np.empty_like(w)
    if len(u) > 0:
        u_prev[0] = u[0]; v_prev[0] = v[0]; w_prev[0] = w[0]
    if len(u) > 1:
        u_prev[1:] = u[:-1]; v_prev[1:] = v[:-1]; w_prev[1:] = w[:-1]

    df = pd.DataFrame({
        "ax":     ax[:,0].astype(np.float32),
        "ay":     ax[:,1].astype(np.float32),
        "az":     ax[:,2].astype(np.float32),
        "qw":     q[:,0].astype(np.float32),
        "qx":     q[:,1].astype(np.float32),
        "qy":     q[:,2].astype(np.float32),
        "qz":     q[:,3].astype(np.float32),
        "u_prev": u_prev.astype(np.float32),
        "v_prev": v_prev.astype(np.float32),
        "w_prev": w_prev.astype(np.float32),
        "u":      u.astype(np.float32),
        "v":      v.astype(np.float32),
        "w":      w.astype(np.float32),
    })[IN_COLS + OUT_COLS]

    # Clean NaN/Inf
    n_bad = np.isnan(df.values).any(axis=1).sum()
    if n_bad:
        print(f"[WARN] Dropping {n_bad} rows with NaN/Inf in one file.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

# ---------- Norm stats (inputs + outputs) ----------

def compute_norm_stats(train_df: pd.DataFrame) -> dict:
    x = train_df[IN_COLS].to_numpy(dtype=np.float64)
    y = train_df[OUT_COLS].to_numpy(dtype=np.float64)
    return {
        "x_mean": x.mean(axis=0).tolist(),
        "x_std":  (x.std(axis=0, ddof=0) + 1e-12).tolist(),
        "y_mean": y.mean(axis=0).tolist(),
        "y_std":  (y.std(axis=0, ddof=0) + 1e-12).tolist(),
    }

# ---------- Windowing helpers ----------

def windows(df: pd.DataFrame, seq: int) -> List[pd.DataFrame]:
    if seq <= 0:
        raise ValueError("--seq must be > 0 to use --shuffle_windows")
    N = len(df); B = N // seq
    if B == 0:
        raise ValueError(f"Sequence length {seq} longer than data ({N}).")
    if N % seq != 0:
        print(f"[make_dataset_v8] Dropping {N - B*seq} tail rows to fit {B} full windows of {seq}.")
    return [df.iloc[i*seq:(i+1)*seq].copy().reset_index(drop=True) for i in range(B)]

def recompute_prev_within_window(win: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure u_prev/v_prev/w_prev at t equals [u,v,w] at t-1 **inside this window**.
    First row copies its own u,v,w.
    """
    u = win["u"].to_numpy()
    v = win["v"].to_numpy()
    w = win["w"].to_numpy()
    u_prev = u.copy(); v_prev = v.copy(); w_prev = w.copy()
    if len(u) > 1:
        u_prev[1:] = u[:-1]; v_prev[1:] = v[:-1]; w_prev[1:] = w[:-1]
    win["u_prev"] = u_prev.astype(np.float32)
    win["v_prev"] = v_prev.astype(np.float32)
    win["w_prev"] = w_prev.astype(np.float32)
    return win

def warn_constants_and_dupes(df: pd.DataFrame, name: str):
    nun = df.nunique()
    constant_cols = [c for c, n in nun.items() if n <= 1]
    if constant_cols:
        print(f"[WARN] {name}: columns constant across all rows: {constant_cols}")
    uniq = df.drop_duplicates().shape[0]
    dup_ratio = 1.0 - (uniq / max(1, len(df)))
    if dup_ratio > 0:
        print(f"[WARN] {name}: {len(df)-uniq} duplicate rows out of {len(df)} total "
              f"(dup_ratio={dup_ratio:.4f}, uniq={uniq}).")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Make v8 dataset (IMU accel + quaternion + prev BODY vel -> BODY vel).")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="SIMDATA CSV(s) to concatenate")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory (train.csv, val.csv, norm_stats.json)")
    ap.add_argument("--split", type=float, default=0.8,
                    help="Train split fraction (0..1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for window shuffling")
    ap.add_argument("--header", choices=["none","infer"], default="none",
                    help="CSV header: none (default) or infer")
    ap.add_argument("--seq", type=int, default=0,
                    help="Window length for safe shuffling (e.g., 200). 0=disable windowing.")
    ap.add_argument("--shuffle_windows", action="store_true",
                    help="Shuffle **windows** of length --seq before split (recomputes prev inside each window).")
    ap.add_argument("--trim_tail", action="store_true",
                    help="Trim trailing stationary rows per file (detects repeated final state).")
    ap.add_argument("--tail_tol", type=float, default=0.0,
                    help="Tolerance for tail equality (default 0.0 = exact).")
    ap.add_argument("--fix_q_sign", action="store_true",
                    help="Enforce quaternion sign continuity across time (recommended).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build per-file then concat to keep file boundaries sane for 'prev'
    dfs = []
    total_removed = 0
    for path in args.inputs:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        df_raw = pd.read_csv(path, header=None) if args.header == "none" else pd.read_csv(path)
        arr = df_raw.to_numpy(dtype=np.float64, copy=False)

        if args.trim_tail:
            arr_trim, removed = trim_trailing_stationary_rows_np(arr, cols=TAIL_COLS,
                                                                 tol=args.tail_tol, keep_one=True)
            if removed > 0:
                print(f"[trim_tail] {os.path.basename(path)}: removed {removed} trailing stationary rows; kept {arr_trim.shape[0]}.")
            total_removed += removed
            arr = arr_trim

        df_one = build_rows_from_one_file(arr, fix_q_sign=args.fix_q_sign)
        dfs.append(df_one)

    if args.trim_tail and total_removed > 0:
        print(f"[trim_tail] Total removed across files: {total_removed}")

    data = pd.concat(dfs, axis=0, ignore_index=True)
    if len(data) == 0:
        raise RuntimeError("No data left after loading and trimming. Check your inputs/flags.")

    # Option A: chronological split (default, no window shuffling)
    if not args.shuffle_windows or args.seq <= 0:
        N = len(data)
        split_idx = int(max(0.0, min(1.0, args.split)) * N)
        train_df = data.iloc[:split_idx].reset_index(drop=True)
        val_df   = data.iloc[split_idx:].reset_index(drop=True)
    else:
        # Option B: window-aware shuffle+split; recompute prev inside each window
        ws = windows(data, args.seq)
        W = len(ws)
        perm = np.random.default_rng(args.seed).permutation(W)
        split_w = int(max(0.0, min(1.0, args.split)) * W)
        train_idx = perm[:split_w]
        val_idx   = perm[split_w:]

        # recompute u_prev/v_prev/w_prev per window AFTER shuffling
        train_ws = [recompute_prev_within_window(ws[i]) for i in train_idx]
        val_ws   = [recompute_prev_within_window(ws[i]) for i in val_idx]

        train_df = pd.concat(train_ws, axis=0, ignore_index=True)
        val_df   = pd.concat(val_ws,   axis=0, ignore_index=True)

    # Diagnostics
    warn_constants_and_dupes(train_df, "train_df")
    warn_constants_and_dupes(val_df,   "val_df")

    # Save CSVs
    train_csv = os.path.join(args.out_dir, "train.csv")
    val_csv   = os.path.join(args.out_dir, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Norm stats from TRAIN split only (inputs+outputs)
    stats = compute_norm_stats(train_df)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote: {train_csv}  ({len(train_df)} rows)")
    print(f"Wrote: {val_csv}    ({len(val_df)} rows)")
    print(f"Wrote: {os.path.join(args.out_dir, 'norm_stats.json')}")
    if args.shuffle_windows and args.seq > 0:
        print(f"Windowed with seq={args.seq}; prev recomputed per-window after shuffling.")

if __name__ == "__main__":
    main()

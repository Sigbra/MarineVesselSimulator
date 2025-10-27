#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_v6.py  (window-aware, time-series safe)

Builds train/val CSVs + normalization stats for QuatVelAttNet from SIMDATA
laid out exactly as given in your C++ exporter.

Inputs written (8):  p,q,r,ax,ay,az,cpsi,spsi
Outputs written (7): vE,vN,vD,qw,qx,qy,qz

Options:
- Chronological split (default) preserves time order.
- Optional window-aware shuffle reorders intact windows (stride=seq) safely.
- Optional trimming removes trailing duplicated rows from each SIM file.

Examples
--------
# Chronological split, trim flat tail
python3 scripts/make_dataset_v6.py \
  --in simdata.csv \
  --out_dir data/nn_dataset_v6_X_C0 \
  --split 0.8 \
  --header none \
  --trim_tail

# Window-aware shuffle (keeps contiguity), trim flat tail
python3 scripts/make_dataset_v6.py \
  --in simdata.csv \
  --out_dir data/nn_dataset_v6_X_C0 \
  --split 0.8 \
  --header none \
  --trim_tail \
  --seq 200 --shuffle_windows --seed 123
"""

import os
import math
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

IN_COLS  = ["p","q","r","ax","ay","az","cpsi","spsi"]
OUT_COLS = ["vE","vN","vD","qw","qx","qy","qz"]

# ---------- Tail-trimming (handles the "copied final state" issue) ----------

# Columns (0-based) that matter to detect a stationary tail in SIMDATA:
# u,v,w (1..3), phi,theta,psi (10..12), accel (46..48), gyro (49..51)
TAIL_COLS = [1, 2, 3, 10, 11, 12, 46, 47, 48, 49, 50, 51]

def trim_trailing_stationary_rows_np(arr: np.ndarray,
                                     cols: List[int] = TAIL_COLS,
                                     tol: float = 0.0,
                                     keep_one: bool = True) -> Tuple[np.ndarray, int]:
    """
    Remove trailing rows that are (within tol) identical to the final row on the
    selected columns. Keeps exactly one final row if keep_one=True.

    Returns:
        trimmed_arr (np.ndarray), num_removed (int)
    """
    N = arr.shape[0]
    if N == 0:
        return arr, 0

    # Ensure the array has all required columns
    max_col = max(cols) if cols else -1
    if arr.shape[1] <= max_col:
        # Not enough columns to evaluate — return unchanged
        return arr, 0

    last = arr[-1, cols]
    diffs = np.abs(arr[:, cols] - last)       # [N, C]
    eq_mask = (diffs <= tol).all(axis=1)      # [N], True where row ~= last on these cols

    if np.all(eq_mask):
        # Entire file is a single repeated state; keep one row (or none)
        new_len = 1 if keep_one else 0
    else:
        # Find last index where row differs from final row
        last_change = np.where(~eq_mask)[0][-1]
        # Keep up to that change plus 1 last row
        new_len = last_change + (2 if keep_one else 1)

    # Guard
    new_len = max(0, min(N, new_len))
    removed = N - new_len
    return arr[:new_len], removed

# ---------- Math helpers (Body→END active rotation; rows [E,N,D]) ----------

def quat_normalize_np(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n

def R_from_euler_rzyx(phi: float, th: float, psi: float) -> np.ndarray:
    """
    Exact convention from your project (rows [East, North, Down]):
      (Ve,Vn,Vd) = R(phi,theta,psi) · [u,v,w]
    """
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth,  sth  = math.cos(th),  math.sin(th)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    return np.array([
        [ spsi*cth,          cpsi*cphi + sphi*sth*spsi,  -cpsi*sphi + sth*spsi*cphi ],
        [ cpsi*cth,         -spsi*cphi + cpsi*sth*sphi,   spsi*sphi + cpsi*cphi*sth ],
        [ -sth,              cth*sphi,                    cth*cphi                   ],
    ], dtype=np.float64)

def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (Body→END, active) → unit quaternion (w,x,y,z)."""
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    else:
        if m00 > m11 and m00 > m22:
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / S; x = 0.25 * S; y = (m01 + m10) / S; z = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / S; x = (m01 + m10) / S; y = 0.25 * S; z = (m12 + m21) / S
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / S; x = (m02 + m20) / S; y = (m12 + m21) / S; z = 0.25 * S
    return quat_normalize_np(np.array([w,x,y,z], dtype=np.float64))

def rotate_body_vel_to_END(R: np.ndarray, u: float, v: float, w_: float) -> Tuple[float,float,float]:
    vec = np.array([u, v, w_], dtype=np.float64)
    out = R @ vec
    return float(out[0]), float(out[1]), float(out[2])

# ---------- Build rows from SIMDATA (strict mapping by index) ----------

def build_rows_from_sim_np(arr: np.ndarray) -> pd.DataFrame:
    """
    arr: numpy array of shape [N, >=52], with columns:
      0: t
      1..12: x' = [u, v, w, p, q, r, x, y, z, phi, theta, psi]
      46..48: imu.accel (ax,ay,az)
      49..51: imu.gyro  (p,q,r)
      (24..29 contain tau_XYN_c / tau_XYN, not used here)
    Returns DataFrame with columns IN_COLS + OUT_COLS.
    """
    if arr.shape[1] < 52:
        raise ValueError(f"Expected at least 52 columns, got {arr.shape[1]}.")

    u     = arr[:, 1].astype(np.float64)
    v     = arr[:, 2].astype(np.float64)
    w_b   = arr[:, 3].astype(np.float64)
    phi   = arr[:,10].astype(np.float64)
    theta = arr[:,11].astype(np.float64)
    psi   = arr[:,12].astype(np.float64)

    ax = arr[:,46].astype(np.float64)
    ay = arr[:,47].astype(np.float64)
    az = arr[:,48].astype(np.float64)
    p  = arr[:,49].astype(np.float64)
    q  = arr[:,50].astype(np.float64)
    r  = arr[:,51].astype(np.float64)

    cpsi = np.cos(psi); spsi = np.sin(psi)
    norm = np.maximum(np.sqrt(cpsi*cpsi + spsi*spsi), 1e-12)
    cpsi /= norm; spsi /= norm

    N = arr.shape[0]
    vE = np.zeros(N, dtype=np.float64)
    vN = np.zeros(N, dtype=np.float64)
    vD = np.zeros(N, dtype=np.float64)
    Q  = np.zeros((N,4), dtype=np.float64)

    for i in range(N):
        R = R_from_euler_rzyx(phi[i], theta[i], psi[i])
        vE[i], vN[i], vD[i] = rotate_body_vel_to_END(R, u[i], v[i], w_b[i])
        Q[i,:] = quat_from_R(R)

    df = pd.DataFrame({
        "p":    p.astype(np.float32),
        "q":    q.astype(np.float32),
        "r":    r.astype(np.float32),
        "ax":   ax.astype(np.float32),
        "ay":   ay.astype(np.float32),
        "az":   az.astype(np.float32),
        "cpsi": cpsi.astype(np.float32),
        "spsi": spsi.astype(np.float32),
        "vE":   vE.astype(np.float32),
        "vN":   vN.astype(np.float32),
        "vD":   vD.astype(np.float32),
        "qw":   Q[:,0].astype(np.float32),
        "qx":   Q[:,1].astype(np.float32),
        "qy":   Q[:,2].astype(np.float32),
        "qz":   Q[:,3].astype(np.float32),
    })[IN_COLS + OUT_COLS]

    # NaN/Inf cleanup
    nans = np.isnan(df.values).any(axis=1).sum()
    if nans:
        print(f"[WARN] Dropping {nans} rows with NaN/Inf before split.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Quick variation check on inputs
    if len(df) >= 5:
        std_inputs = df[IN_COLS].std().to_dict()
        if all(v == 0.0 for v in std_inputs.values()):
            print("[WARN] All input columns have zero std; check your SIM export.")
    return df

# ---------- Normalization stats (inputs + velocity outputs) ----------

def compute_norm_stats(train_df: pd.DataFrame) -> dict:
    x = train_df[IN_COLS].to_numpy(dtype=np.float64)
    yv = train_df[["vE","vN","vD"]].to_numpy(dtype=np.float64)
    return {
        "x_mean": x.mean(axis=0).tolist(),
        "x_std":  (x.std(axis=0, ddof=0) + 1e-12).tolist(),
        "y_mean": yv.mean(axis=0).tolist(),
        "y_std":  (yv.std(axis=0, ddof=0) + 1e-12).tolist(),
    }

# ---------- Windowing helpers ----------

def windows(df: pd.DataFrame, seq: int) -> List[pd.DataFrame]:
    """Split df into non-overlapping windows of length seq. Drop remainder."""
    if seq <= 0:
        raise ValueError("--seq must be > 0 to use --shuffle_windows")
    N = len(df); B = N // seq
    if B == 0:
        raise ValueError(f"Sequence length {seq} longer than data ({N}).")
    if N % seq != 0:
        print(f"[make_dataset_v6] Dropping {N - B*seq} tail rows to fit {B} full windows of {seq}.")
    # Use .copy() to avoid pandas view surprises
    return [df.iloc[i*seq:(i+1)*seq].copy().reset_index(drop=True) for i in range(B)]

def warn_constants_and_dupes(df: pd.DataFrame, name: str,
                             *, hard_fail: bool = False,
                             dup_fail_ratio: float = 0.99,
                             min_unique_fail: int = 5):
    """
    Warn about constant columns and duplicate rows.
    Optionally hard-fail if the dataset degenerates too much.
    """
    nun = df.nunique()
    constant_cols = [c for c, n in nun.items() if n <= 1]
    if constant_cols:
        print(f"[WARN] {name}: columns constant across all rows: {constant_cols}")

    uniq = df.drop_duplicates().shape[0]
    dup_ratio = 1.0 - (uniq / max(1, len(df)))
    if dup_ratio > 0:
        print(f"[WARN] {name}: {len(df)-uniq} duplicate rows out of {len(df)} total "
              f"(dup_ratio={dup_ratio:.4f}, uniq={uniq}).")

    if hard_fail and (dup_ratio >= dup_fail_ratio or uniq < min_unique_fail):
        raise RuntimeError(
            f"{name} looks degenerate (dup_ratio={dup_ratio:.4f} ≥ {dup_fail_ratio} "
            f"or uniq={uniq} < {min_unique_fail})."
        )

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Make dataset for QuatVelAttNet (time-series safe).")
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
                    help="Shuffle **windows** of length --seq before split.")
    ap.add_argument("--trim_tail", action="store_true",
                    help="Trim trailing stationary rows per file (detects repeated final state).")
    ap.add_argument("--tail_tol", type=float, default=0.0,
                    help="Tolerance for tail equality (default 0.0 = exact).")
    ap.add_argument("--hard_fail", action="store_true",
                    help="Raise if split degenerates (use with --dup_fail_ratio / --min_unique_fail).")
    ap.add_argument("--dup_fail_ratio", type=float, default=0.99,
                    help="Fail if duplicate-row ratio ≥ this (default 0.99).")
    ap.add_argument("--min_unique_fail", type=int, default=5,
                    help="Fail if unique rows < this (default 5).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load each file, optionally trim trailing stationary rows, then concat
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

        dfs.append(pd.DataFrame(arr))
    if args.trim_tail and total_removed > 0:
        print(f"[trim_tail] Total removed across files: {total_removed}")

    raw = pd.concat(dfs, axis=0, ignore_index=True)
    if len(raw) == 0:
        raise RuntimeError("No data left after loading and trimming. Check your inputs/flags.")

    # Build target columns (inputs + outputs) for the NN
    arr_all = raw.to_numpy(dtype=np.float64, copy=False)
    data = build_rows_from_sim_np(arr_all)

    # Option A: chronological split (default, no window shuffling)
    if not args.shuffle_windows or args.seq <= 0:
        N = len(data)
        split_idx = int(max(0.0, min(1.0, args.split)) * N)
        train_df = data.iloc[:split_idx].reset_index(drop=True)
        val_df   = data.iloc[split_idx:].reset_index(drop=True)
    else:
        # Option B: window-aware shuffle+split (use permutation indices — not in-place)
        ws = windows(data, args.seq)  # list of copies
        W = len(ws)
        perm = np.random.default_rng(args.seed).permutation(W)
        split_w = int(max(0.0, min(1.0, args.split)) * W)
        train_idx = perm[:split_w]
        val_idx   = perm[split_w:]
        train_df = pd.concat([ws[i] for i in train_idx], axis=0, ignore_index=True)
        val_df   = pd.concat([ws[i] for i in val_idx],   axis=0, ignore_index=True)

    # Diagnostics before save
    warn_constants_and_dupes(train_df, "train_df",
                             hard_fail=args.hard_fail,
                             dup_fail_ratio=args.dup_fail_ratio,
                             min_unique_fail=args.min_unique_fail)
    warn_constants_and_dupes(val_df, "val_df",
                             hard_fail=args.hard_fail,
                             dup_fail_ratio=args.dup_fail_ratio,
                             min_unique_fail=args.min_unique_fail)

    # Save CSVs
    train_csv = os.path.join(args.out_dir, "train.csv")
    val_csv   = os.path.join(args.out_dir, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Norm stats from TRAIN split only
    stats = compute_norm_stats(train_df)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote: {train_csv}  ({len(train_df)} rows)")
    print(f"Wrote: {val_csv}    ({len(val_df)} rows)")
    print(f"Wrote: {os.path.join(args.out_dir, 'norm_stats.json')}")
    if args.shuffle_windows and args.seq > 0:
        print(f"Windowed with seq={args.seq}; files are concatenations of intact {args.seq}-step windows.")

if __name__ == "__main__":
    main()

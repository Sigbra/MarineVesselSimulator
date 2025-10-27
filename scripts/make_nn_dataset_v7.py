#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_v7.py  (window-aware, time-series safe)

Builds train/val CSVs + normalization stats for a velocity-only observer (v7).

Inputs written (7):   ax,ay,az, qw,qx,qy,qz
  - qw..qz are read directly from SIMDATA cols 52..55 (q_nb: Body→END)
  - NOTE: IMU gyro is NOT used as input in this version.
Outputs written (3):  vE,vN,vD
  - computed via v^n = R(q_nb) · [u,v,w]^b

Options:
- Chronological split (default) preserves time order.
- Optional window-aware shuffle reorders intact windows (stride=seq) safely.
- Optional trimming removes trailing duplicated rows from each SIM file.
- Optional quaternion sign continuity fix (on via --fix_q_sign).

Example
-------
python3 scripts/make_dataset_v7.py \
  --in data/simdata.csv \
  --out_dir data/nn_dataset_v7_X_C0 \
  --split 0.8 \
  --header none \
  --trim_tail \
  --seq 200 --shuffle_windows --seed 123 \
  --fix_q_sign
"""

import os
import json
import math
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

# ----- Column mapping in SIMDATA (0-based) -----
# 0: t
# 1..3:   u, v, w             (BODY linear vel)
# 4..6:   p, q, r             (BODY rates)            [NOT USED as input here]
# 7..9:   x, y, z             (END position)
# 10..12: phi, theta, psi     (END Euler)
# 46..48: imu.accel (ax, ay, az)   (BODY specific force)   <-- input
# 49..51: imu.gyro  (wx, wy, wz)   (BODY rates)            [NOT USED]
# 52..55: q_nb (qw, qx, qy, qz)    (BODY→END quaternion)   <-- input

AX_COLS  = [46, 47, 48]
Q_COLS   = [52, 53, 54, 55]
UVW_COLS = [1, 2, 3]

# Columns to detect stationary tails (we can still include gyro/Euler for detection)
TAIL_COLS = [1,2,3, 10,11,12] + AX_COLS + [49,50,51] + Q_COLS

IN_COLS  = ["ax","ay","az","qw","qx","qy","qz"]
OUT_COLS = ["vE","vN","vD"]

# ---------- Quaternion helpers ----------

def quat_normalize_np(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n

def quat_conj_np(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    # q,r: [...,4] (w,x,y,z)
    w1,x1,y1,z1 = np.moveaxis(q, -1, 0)
    w2,x2,y2,z2 = np.moveaxis(r, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w,x,y,z], axis=-1)

def quat_rotate_vec_np(q: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """
    Rotate BODY vector v3 -> END using quaternion q (BODY→END).
    Shapes: q [N,4], v3 [N,3] -> out [N,3]
    """
    q = quat_normalize_np(q)
    zeros = np.zeros_like(v3[..., :1])
    vq = np.concatenate([zeros, v3], axis=-1)         # [N,4]
    return quat_mul_np(quat_mul_np(q, vq), quat_conj_np(q))[..., 1:]

def rotate_body_vel_to_END_with_q(q: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """v^n = R(q) · v^b, via quaternion rotation."""
    return quat_rotate_vec_np(q, uvw)

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

# ---------- Build rows from SIMDATA ----------

def build_rows_from_sim_np(arr: np.ndarray,
                           fix_q_sign: bool = True) -> pd.DataFrame:
    """
    arr: numpy array of shape [N, >=56], with columns as mapped above.
    Returns DataFrame with columns IN_COLS + OUT_COLS.
    """
    need_cols = max(AX_COLS + Q_COLS + UVW_COLS) + 1
    if arr.shape[1] < need_cols:
        raise ValueError(f"Expected at least {need_cols} columns; got {arr.shape[1]}.")

    ax  = arr[:, AX_COLS].astype(np.float64)   # [N,3]
    q   = arr[:, Q_COLS].astype(np.float64)    # [N,4] (qw,qx,qy,qz) BODY→END
    uvw = arr[:, UVW_COLS].astype(np.float64)  # [N,3] (u,v,w) BODY

    # Normalize quaternion; optionally enforce sign continuity (doesn't change rotation)
    q = quat_normalize_np(q)
    if fix_q_sign:
        q = fix_quaternion_sign_continuity(q)

    # Outputs: velocity in END using stored quaternion
    v_end = rotate_body_vel_to_END_with_q(q, uvw)  # [N,3] -> vE,vN,vD

    df = pd.DataFrame({
        "ax":  ax[:,0].astype(np.float32),
        "ay":  ax[:,1].astype(np.float32),
        "az":  ax[:,2].astype(np.float32),
        "qw":  q[:,0].astype(np.float32),
        "qx":  q[:,1].astype(np.float32),
        "qy":  q[:,2].astype(np.float32),
        "qz":  q[:,3].astype(np.float32),
        "vE":  v_end[:,0].astype(np.float32),
        "vN":  v_end[:,1].astype(np.float32),
        "vD":  v_end[:,2].astype(np.float32),
    })[IN_COLS + OUT_COLS]

    # Clean NaN/Inf
    n_bad = np.isnan(df.values).any(axis=1).sum()
    if n_bad:
        print(f"[WARN] Dropping {n_bad} rows with NaN/Inf.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Quick variation check on inputs
    if len(df) >= 5:
        std_inputs = df[IN_COLS].std().to_dict()
        if all(v == 0.0 for v in std_inputs.values()):
            print("[WARN] All input columns have zero std; check your SIM export.")
    return df

# ---------- Norm stats (inputs + velocity outputs) ----------

def compute_norm_stats(train_df: pd.DataFrame) -> dict:
    x = train_df[IN_COLS].to_numpy(dtype=np.float64)
    yv = train_df[OUT_COLS].to_numpy(dtype=np.float64)
    return {
        "x_mean": x.mean(axis=0).tolist(),
        "x_std":  (x.std(axis=0, ddof=0) + 1e-12).tolist(),
        "y_mean": yv.mean(axis=0).tolist(),
        "y_std":  (yv.std(axis=0, ddof=0) + 1e-12).tolist(),
    }

# ---------- Windowing helpers ----------

def windows(df: pd.DataFrame, seq: int) -> List[pd.DataFrame]:
    if seq <= 0:
        raise ValueError("--seq must be > 0 to use --shuffle_windows")
    N = len(df); B = N // seq
    if B == 0:
        raise ValueError(f"Sequence length {seq} longer than data ({N}).")
    if N % seq != 0:
        print(f"[make_dataset_v7] Dropping {N - B*seq} tail rows to fit {B} full windows of {seq}.")
    return [df.iloc[i*seq:(i+1)*seq].copy().reset_index(drop=True) for i in range(B)]

def warn_constants_and_dupes(df: pd.DataFrame, name: str,
                             *, hard_fail: bool = False,
                             dup_fail_ratio: float = 0.99,
                             min_unique_fail: int = 5):
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
    ap = argparse.ArgumentParser(description="Make v7 dataset (IMU accel + stored quaternion -> END velocity).")
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
    ap.add_argument("--fix_q_sign", action="store_true",
                    help="Enforce quaternion sign continuity across time (recommended).")
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

    # Build target columns (inputs + outputs) for the NN using STORED quaternions
    arr_all = raw.to_numpy(dtype=np.float64, copy=False)
    data = build_rows_from_sim_np(arr_all, fix_q_sign=args.fix_q_sign)

    # Option A: chronological split (default, no window shuffling)
    if not args.shuffle_windows or args.seq <= 0:
        N = len(data)
        split_idx = int(max(0.0, min(1.0, args.split)) * N)
        train_df = data.iloc[:split_idx].reset_index(drop=True)
        val_df   = data.iloc[split_idx:].reset_index(drop=True)
    else:
        # Option B: window-aware shuffle+split (keeps contiguity)
        ws = windows(data, args.seq)
        W = len(ws)
        perm = np.random.default_rng(args.seed).permutation(W)
        split_w = int(max(0.0, min(1.0, args.split)) * W)
        train_idx = perm[:split_w]
        val_idx   = perm[split_w:]
        train_df = pd.concat([ws[i] for i in train_idx], axis=0, ignore_index=True)
        val_df   = pd.concat([ws[i] for i in val_idx],   axis=0, ignore_index=True)

    # Diagnostics
    warn_constants_and_dupes(train_df, "train_df")
    warn_constants_and_dupes(val_df,   "val_df")

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

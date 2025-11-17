#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_nn_dataset_v11.py  (time-series safe, with TEST split)

Builds train/val/test CSVs + normalization stats for the velocity-only observer (v11).

v11 expectations (for streaming/stateful GRU):
  Inputs (10): ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n
  Aux    (3): w_est_x, w_est_y, w_est_z            (used only in physics loss)
  Outputs(3): vE, vN, vD                            (END frame)

Sources in SIMDATA (0-based indices):
  ax..az  := cols 46..48 (imu.accel, BODY specific force)
  qw..qz  := cols 52..55 (q_nb: BODY→END quaternion, custom END convention)
  tau_*   := cols 27..29 (actual, BODY@CO)     or cols 24..26 if --tau_source commanded
  u,v,w   := cols 1..3  (BODY linear velocity) → rotated to END for targets
  w_est_* := cols 56..58 (BODY angular rates from q-observer) — aux only

Key defaults for v11:
- Chronological split by default (order preserved).
- Quaternion sign continuity fix **enabled** by default.
- Window shuffling is supported but discouraged for streaming training.

Example
-------
python3 scripts/make_nn_dataset_v11.py \
  --in data/simdata.csv \
  --out_dir data/nn_dataset_v11_X_C6 \
  --val_frac 0.15 --test_frac 0.15 \
  --header none \
  --trim_tail \
  --seq 100 --shuffle_windows --seed 123
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

VERSION = "v11"

# ----- Column mapping in SIMDATA (0-based) -----
# 0: t
# 1..3:   u, v, w                 (BODY linear vel)
# 4..6:   p, q, r                 (BODY rates)              [NOT USED as input]
# 7..9:   x, y, z                 (END position)
# 10..12: phi, theta, psi         (END Euler)
# 24..26: tau_XYN_c               (commanded)
# 27..29: tau_XYN                 (actual, BODY @ CO)
# 34..45: x_est (12)              [NOT USED]
# 46..48: imu.accel (ax, ay, az)  (BODY specific force)     <-- input
# 49..51: imu.gyro  (wx, wy, wz)  (BODY rates)              [NOT USED]
# 52..55: q_nb (qw, qx, qy, qz)   (BODY→END quaternion)     <-- input
# 56..58: w_est (wx, wy, wz)      (BODY rates from q-Obs)   <-- AUX (for physics loss)

AX_COLS     = [46, 47, 48]
Q_COLS      = [52, 53, 54, 55]
UVW_COLS    = [1, 2, 3]
TAU_ACT     = [27, 28, 29]
TAU_CMD     = [24, 25, 26]
W_EST_COLS  = [56, 57, 58]

# Columns used to detect stationary tails (safe to include many signals)
TAIL_COLS = UVW_COLS + [10,11,12] + AX_COLS + [49,50,51] + Q_COLS + W_EST_COLS

IN_COLS     = ["ax","ay","az","qw","qx","qy","qz","tau_x","tau_y","tau_n"]
OUT_COLS    = ["vE","vN","vD"]
AUX_W_COLS  = ["w_est_x","w_est_y","w_est_z"]

# ---------- Custom END rotation (quaternion → R_nb) ----------

def rnb_from_quat_custom_np(q: np.ndarray) -> np.ndarray:
    """
    Map quaternion (w,x,y,z) to the custom END rotation matrix R_nb as in your C++:
      Row 1 (East):  [ 2(xy+wz),  1-2(xx+zz),  2(yz-wx) ]
      Row 2 (North): [ 1-2(yy+zz), 2(xy-wz),  2(xz+wy) ]
      Row 3 (Down):  [ 2(xz-wy),   2(yz+wx),  1-2(xx+yy)]
    q: [N,4]
    returns R: [N,3,3]
    """
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty((q.shape[0], 3, 3), dtype=q.dtype)
    # Row E
    R[:,0,0] = 2.0*(xy + wz)
    R[:,0,1] = 1.0 - 2.0*(xx + zz)
    R[:,0,2] = 2.0*(yz - wx)
    # Row N
    R[:,1,0] = 1.0 - 2.0*(yy + zz)
    R[:,1,1] = 2.0*(xy - wz)
    R[:,1,2] = 2.0*(xz + wy)
    # Row D
    R[:,2,0] = 2.0*(xz - wy)
    R[:,2,1] = 2.0*(yz + wx)
    R[:,2,2] = 1.0 - 2.0*(xx + yy)
    return R

def rotate_body_vel_to_END_with_q(q: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """v^n = R_nb(q) · v^b, using the custom END convention."""
    R = rnb_from_quat_custom_np(q)        # [N,3,3]
    return np.einsum('nij,nj->ni', R, uvw)  # -> [N,3] = [vE,vN,vD]

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
                           tau_cols: List[int],
                           fix_q_sign: bool = True) -> pd.DataFrame:
    """
    Returns DataFrame with columns IN_COLS + OUT_COLS + AUX_W_COLS.
    """
    need_cols = max(AX_COLS + Q_COLS + UVW_COLS + tau_cols + W_EST_COLS) + 1
    if arr.shape[1] < need_cols:
        raise ValueError(f"Expected at least {need_cols} columns; got {arr.shape[1]}.")

    ax    = arr[:, AX_COLS].astype(np.float64)          # [N,3]
    q     = arr[:, Q_COLS].astype(np.float64)           # [N,4] (qw,qx,qy,qz) BODY→END (custom)
    uvw_b = arr[:, UVW_COLS].astype(np.float64)         # [N,3] (u,v,w) BODY
    tau_b = arr[:, tau_cols].astype(np.float64)         # [N,3] (tau_X, tau_Y, tau_N) BODY @ CO
    w_est = arr[:, W_EST_COLS].astype(np.float64)       # [N,3] BODY rates (aux)

    # Normalize quaternion; optionally enforce sign continuity
    qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    q  = q / qn
    if fix_q_sign:
        q = fix_quaternion_sign_continuity(q)

    # Outputs: velocity in END using stored quaternion (custom R_nb)
    v_end = rotate_body_vel_to_END_with_q(q, uvw_b)     # [N,3] -> vE,vN,vD

    df = pd.DataFrame({
        "ax":    ax[:,0].astype(np.float32),
        "ay":    ax[:,1].astype(np.float32),
        "az":    ax[:,2].astype(np.float32),
        "qw":    q[:,0].astype(np.float32),
        "qx":    q[:,1].astype(np.float32),
        "qy":    q[:,2].astype(np.float32),
        "qz":    q[:,3].astype(np.float32),
        "tau_x": tau_b[:,0].astype(np.float32),
        "tau_y": tau_b[:,1].astype(np.float32),
        "tau_n": tau_b[:,2].astype(np.float32),
        "vE":    v_end[:,0].astype(np.float32),
        "vN":    v_end[:,1].astype(np.float32),
        "vD":    v_end[:,2].astype(np.float32),
        "w_est_x": w_est[:,0].astype(np.float32),
        "w_est_y": w_est[:,1].astype(np.float32),
        "w_est_z": w_est[:,2].astype(np.float32),
    })[IN_COLS + OUT_COLS + AUX_W_COLS]

    # Clean NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
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
        print(f"[make_nn_dataset_{VERSION}] Dropping {N - B*seq} tail rows to fit {B} full windows of {seq}.")
    return [df.iloc[i*seq:(i+1)*seq].copy().reset_index(drop=True) for i in range(B)]

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

# ---------- Splitting helpers ----------

def three_way_indices(total: int, val_frac: float, test_frac: float):
    """Return (n_train, n_val, n_test) that sum to total, deterministic rounding."""
    val_n = int(round(val_frac * total))
    test_n = int(round(test_frac * total))
    # ensure we don't exceed
    val_n = max(0, min(total, val_n))
    test_n = max(0, min(total - val_n, test_n))
    train_n = total - val_n - test_n
    if train_n <= 0:
        raise ValueError("Fractions leave no room for training data. Reduce val/test fractions.")
    return train_n, val_n, test_n

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description=f"Make {VERSION} dataset (IMU accel + quaternion + BODY τ_XYN -> END velocity) + aux w_est.")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="SIMDATA CSV(s) to concatenate")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory (train.csv, val.csv, test.csv, norm_stats.json)")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="Validation fraction (0..1). Train = 1 - val - test")
    ap.add_argument("--test_frac", type=float, default=0.15,
                    help="Test fraction (0..1). Train = 1 - val - test")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for window shuffling")
    ap.add_argument("--header", choices=["none","infer"], default="none",
                    help="CSV header: none (default) or infer")
    ap.add_argument("--seq", type=int, default=0,
                    help="Window length for safe shuffling (e.g., 200). 0=disable windowing.")
    ap.add_argument("--shuffle_windows", action="store_true",
                    help="Shuffle **windows** of length --seq before split (window-safe).")
    ap.add_argument("--trim_tail", action="store_true",
                    help="Trim trailing stationary rows per file (detects repeated final state).")
    ap.add_argument("--tail_tol", type=float, default=0.0,
                    help="Tolerance for tail equality (default 0.0 = exact).")
    # v11: fix_q_sign enabled by default; provide opt-out
    ap.add_argument("--no_fix_q_sign", dest="fix_q_sign", action="store_false",
                    help="Disable quaternion sign continuity fix (enabled by default).")
    ap.set_defaults(fix_q_sign=True)
    ap.add_argument("--tau_source", choices=["actual","commanded"], default="actual",
                    help="Use tau_XYN actual (27..29) or commanded (24..26) as inputs.")
    args = ap.parse_args()

    if args.val_frac < 0 or args.test_frac < 0 or (args.val_frac + args.test_frac) >= 1.0:
        raise ValueError("Require 0 ≤ val_frac, test_frac and val_frac+test_frac < 1.")

    os.makedirs(args.out_dir, exist_ok=True)

    tau_cols = TAU_ACT if args.tau_source == "actual" else TAU_CMD

    if args.shuffle_windows and args.seq <= 0:
        raise ValueError("--shuffle_windows requires --seq > 0")

    if args.shuffle_windows:
        print(f"[note] {VERSION}: Window shuffling is for data augmentation only. "
              "Streaming v11 training typically benefits from preserved order.")

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

    # Build target columns (inputs + outputs + aux) using STORED quaternion and chosen τ source
    arr_all = raw.to_numpy(dtype=np.float64, copy=False)
    data = build_rows_from_sim_np(arr_all, tau_cols=tau_cols, fix_q_sign=args.fix_q_sign)

    # 3-way split (chronological by default)
    if not args.shuffle_windows or args.seq <= 0:
        N = len(data)
        n_train, n_val, n_test = three_way_indices(N, args.val_frac, args.test_frac)
        train_df = data.iloc[:n_train].reset_index(drop=True)
        val_df   = data.iloc[n_train:n_train+n_val].reset_index(drop=True)
        test_df  = data.iloc[n_train+n_val:].reset_index(drop=True)
    else:
        ws = windows(data, args.seq)
        W = len(ws)
        nW_train, nW_val, nW_test = three_way_indices(W, args.val_frac, args.test_frac)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(W)
        train_idx = perm[:nW_train]
        val_idx   = perm[nW_train:nW_train+nW_val]
        test_idx  = perm[nW_train+nW_val:]
        train_df = pd.concat([ws[i] for i in train_idx], axis=0, ignore_index=True)
        val_df   = pd.concat([ws[i] for i in val_idx],   axis=0, ignore_index=True)
        test_df  = pd.concat([ws[i] for i in test_idx],  axis=0, ignore_index=True)

    # Diagnostics
    warn_constants_and_dupes(train_df, "train_df")
    warn_constants_and_dupes(val_df,   "val_df")
    warn_constants_and_dupes(test_df,  "test_df")

    # Save CSVs (only the 16 columns required by v11 loaders)
    train_csv = os.path.join(args.out_dir, "train.csv")
    val_csv   = os.path.join(args.out_dir, "val.csv")
    test_csv  = os.path.join(args.out_dir, "test.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Norm stats from TRAIN split only (inputs+velocity outputs)
    stats = compute_norm_stats(train_df)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Persist split summary for reproducibility (tag with version)
    with open(os.path.join(args.out_dir, "split_summary.json"), "w") as f:
        json.dump({
            "version": VERSION,
            "counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "fractions": {"val_frac": args.val_frac, "test_frac": args.test_frac,
                          "train_frac": 1.0 - args.val_frac - args.test_frac},
            "shuffle_windows": bool(args.shuffle_windows and args.seq > 0),
            "seq": int(args.seq),
            "seed": int(args.seed),
            "tau_source": args.tau_source,
            "fix_q_sign": bool(args.fix_q_sign),
            "columns": {"inputs": IN_COLS, "outputs": OUT_COLS, "aux": AUX_W_COLS}
        }, f, indent=2)

    # Optional: write schema for downstream sanity checks
    with open(os.path.join(args.out_dir, "schema_v11.json"), "w") as f:
        json.dump({
            "version": VERSION,
            "columns_order": IN_COLS + OUT_COLS + AUX_W_COLS,
            "dtypes": {c: "float32" for c in (IN_COLS + OUT_COLS + AUX_W_COLS)},
            "notes": "CSV rows are chronological; v11 training is streaming/one-step.",
        }, f, indent=2)

    print(f"[{VERSION}] Wrote: {train_csv}  ({len(train_df)} rows)")
    print(f"[{VERSION}] Wrote: {val_csv}    ({len(val_df)} rows)")
    print(f"[{VERSION}] Wrote: {test_csv}   ({len(test_df)} rows)")
    print(f"[{VERSION}] Wrote: {os.path.join(args.out_dir, 'norm_stats.json')}")
    print(f"[{VERSION}] Wrote: {os.path.join(args.out_dir, 'split_summary.json')}")
    print(f"[{VERSION}] Wrote: {os.path.join(args.out_dir, 'schema_v11.json')}")
    if args.shuffle_windows and args.seq > 0:
        print(f"[{VERSION}] Windowed with seq={args.seq}; files are concatenations of intact {args.seq}-step windows (shuffled across windows only).")
    else:
        print(f"[{VERSION}] Chronological split by rows (no window shuffling).")
    print(f"[{VERSION}] τ source: {args.tau_source}  (cols {tau_cols})  | fix_q_sign={args.fix_q_sign}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a tidy NN training/validation dataset from simulated boat logs.

INPUT (sim CSVs, no header):
  col 0                -> t
  col 1..12            -> x.transpose()   (state vector; map indices to u,v,w,p,q,r)
  col 43..45           -> imu.accel(0..2) (ax, ay, az)
  col 46..48           -> imu.gyro(0..2)  (wx, wy, wz)

OUTPUT (CSV with header):
  out_dir/all_runs.csv  columns: [run_id, t, ax, ay, az, wx, wy, wz, u, v, w, p, q, r]
  out_dir/train.csv
  out_dir/val.csv
  out_dir/manifest.csv
  out_dir/norm_stats.json (computed on TRAIN only)

Usage; From repo root do
python3 scripts/make_nn_dataset.py --data_dir data/ --out_dir data/nn_dataset_X_C0 --pattern "simdata.csv" --train_ratio 0.8 --seed 123

TODO: Remove duplicate rows at the end of simdata before making datasets!
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# --------------- CONFIG: map your x[1..12] to targets ----------------
# x = [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
X_INDEX = {"u":1, "v":2, "w":3, "p":4, "q":5, "r":6}

# IMU columns (0-based absolute CSV indices)
COL_T = 0
COL_AX, COL_AY, COL_AZ = 43, 44, 45
COL_WX, COL_WY, COL_WZ = 46, 47, 48
# x occupies CSV columns 1..12 inclusive (0-based)
X_START, X_END = 1, 12  # pandas slice uses end-exclusive -> [:, 1:13]

# ---------------------------------------------------------------------

def load_sim_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    needed = [COL_T, COL_AX, COL_AY, COL_AZ, COL_WX, COL_WY, COL_WZ]
    if df.shape[1] <= max(needed + [X_END]):
        raise ValueError(f"{path} has only {df.shape[1]} cols; need ≥ {max(needed + [X_END]) + 1}.")
    return df

def extract_xy(df: pd.DataFrame) -> pd.DataFrame:
    ax, ay, az = df.iloc[:, COL_AX], df.iloc[:, COL_AY], df.iloc[:, COL_AZ]
    wx, wy, wz = df.iloc[:, COL_WX], df.iloc[:, COL_WY], df.iloc[:, COL_WZ]
    x_cols = df.iloc[:, X_START:X_END+1]
    def x_at(k1: int) -> pd.Series:  # 1-based within x block
        return x_cols.iloc[:, k1-1]
    out = pd.DataFrame({
        "t":  df.iloc[:, COL_T],
        "ax": ax, "ay": ay, "az": az,
        "wx": wx, "wy": wy, "wz": wz,
        "u":  x_at(X_INDEX["u"]),
        "v":  x_at(X_INDEX["v"]),
        "w":  x_at(X_INDEX["w"]),
        "p":  x_at(X_INDEX["p"]),
        "q":  x_at(X_INDEX["q"]),
        "r":  x_at(X_INDEX["r"]),
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out

def compute_dt(t: pd.Series) -> float:
    arr = t.to_numpy()
    if len(arr) < 2: return float("nan")
    return float(np.median(np.diff(arr)))

def split_by_files(files: List[Path], train_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    k = int(round(train_ratio * len(files)))
    k = min(max(k, 1), len(files)-1) if len(files) > 1 else k
    train_files = [files[i] for i in idx[:k]]
    val_files   = [files[i] for i in idx[k:]]
    return train_files, val_files

def contiguous_intra_split(df_xy: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one run into train/val by time (contiguous)."""
    n = len(df_xy)
    if n < 2:
        return df_xy.iloc[0:0].copy(), df_xy.iloc[0:0].copy()
    n_val = max(1, int(round(val_ratio * n)))
    n_val = min(n_val, n-1)
    cut = n - n_val
    tr = df_xy.iloc[:cut].reset_index(drop=True)
    va = df_xy.iloc[cut:].reset_index(drop=True)
    return tr, va

def main():
    import argparse, json
    from pathlib import Path
    import pandas as pd
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="Folder with sim CSV(s); default = <repo>/data")
    ap.add_argument("--pattern", default="simdata*.csv", help="Glob, e.g. simdata*.csv or simdata.csv")
    ap.add_argument("--out_dir", default=None, help="Output folder; default = <repo>/data/nn_dataset")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="File-level split ratio (used when >1 file)")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Fallback intra-run ratio (contiguous tail)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # ---------- Resolve paths relative to repo root (parent of scripts/) ----------
    script_dir = Path(__file__).resolve().parent          # <repo>/scripts
    repo_root  = script_dir.parent                        # <repo>
    default_data_dir = repo_root / "data"
    default_out_dir  = default_data_dir / "nn_dataset"

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] repo_root={repo_root}")
    print(f"[paths] data_dir ={data_dir}")
    print(f"[paths] out_dir  ={out_dir}")

    # ---------- Gather files ----------
    files = sorted(data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match {data_dir / args.pattern}")

    # Try file-level split if multiple files; otherwise we’ll intra-split
    if len(files) > 1:
        train_files, val_files = split_by_files(files, args.train_ratio, args.seed)
    else:
        train_files, val_files = files, []  # single file → intra-split below

    manifest_rows = []
    frames_train, frames_val = [], []

    # ---------- Process each file ----------
    for f in files:
        df_raw = load_sim_csv(f)
        df_xy  = extract_xy(df_raw)
        base_id = f.stem
        dt = compute_dt(df_xy["t"])

        goes_to_val = (f in val_files) and (len(df_xy) > 0)

        if goes_to_val:
            df_xy_val = df_xy.copy()
            df_xy_val.insert(0, "run_id", base_id)
            frames_val.append(df_xy_val)
            manifest_rows.append({"run_id": base_id, "file": str(f), "rows": len(df_xy), "dt_median": dt, "split": "val"})
        else:
            if len(files) == 1 or len(val_files) == 0:
                # Single file or empty val split → contiguous intra-run split (tail to val)
                tr, va = contiguous_intra_split(df_xy, args.val_ratio)
                tr_id = f"{base_id}_tr"
                va_id = f"{base_id}_va"
                if len(tr) > 0:
                    tr = tr.copy(); tr.insert(0, "run_id", tr_id); frames_train.append(tr)
                    manifest_rows.append({"run_id": tr_id, "file": str(f), "rows": len(tr), "dt_median": dt, "split": "train"})
                if len(va) > 0:
                    va = va.copy(); va.insert(0, "run_id", va_id); frames_val.append(va)
                    manifest_rows.append({"run_id": va_id, "file": str(f), "rows": len(va), "dt_median": dt, "split": "val"})
            else:
                df_xy_tr = df_xy.copy()
                df_xy_tr.insert(0, "run_id", base_id)
                frames_train.append(df_xy_tr)
                manifest_rows.append({"run_id": base_id, "file": str(f), "rows": len(df_xy), "dt_median": dt, "split": "train"})

    # ---------- Concatenate splits ----------
    cols = ["run_id","t","ax","ay","az","wx","wy","wz","u","v","w","p","q","r"]
    df_train = pd.concat(frames_train, axis=0, ignore_index=True) if frames_train else pd.DataFrame(columns=cols)
    df_val   = pd.concat(frames_val,   axis=0, ignore_index=True) if frames_val   else pd.DataFrame(columns=cols)

    # If val still empty, split the largest train run
    if len(df_val) == 0 and len(df_train) > 0:
        print("Note: val split empty; performing intra-run split on largest train run.")
        counts = df_train.groupby("run_id").size().sort_values(ascending=False)
        rid = counts.index[0]
        df_run = df_train[df_train["run_id"] == rid].drop(columns=["run_id"]).reset_index(drop=True)
        tr, va = contiguous_intra_split(df_run, args.val_ratio)
        dt = compute_dt(df_run["t"])
        # Rebuild train/val
        df_train = df_train[df_train["run_id"] != rid]
        tr.insert(0, "run_id", f"{rid}_tr2")
        va.insert(0, "run_id", f"{rid}_va2")
        df_train = pd.concat([df_train, tr], axis=0, ignore_index=True)
        df_val   = va
        manifest_rows.append({"run_id": f"{rid}_tr2", "file": "(from intra-split)", "rows": len(tr), "dt_median": dt, "split": "train"})
        manifest_rows.append({"run_id": f"{rid}_va2", "file": "(from intra-split)", "rows": len(va), "dt_median": dt, "split": "val"})

    # ---------- Save outputs ----------
    df_all = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    (out_dir / "all_runs.csv").write_text(df_all.to_csv(index=False))
    (out_dir / "train.csv").write_text(df_train.to_csv(index=False))
    (out_dir / "val.csv").write_text(df_val.to_csv(index=False))
    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    # ---------- Norm stats (optional; computed on TRAIN only) ----------
    def stats(df: pd.DataFrame, cols: list[str]) -> Dict[str, Dict[str, float]]:
        if len(df) == 0:
            return {"mean": {c: 0.0 for c in cols}, "std": {c: 1.0 for c in cols}}
        mu = df[cols].mean().to_dict()
        sd = (df[cols].std(ddof=0).replace(0.0, 1e-8)).to_dict()
        return {"mean": mu, "std": sd}

    input_cols  = ["ax","ay","az","wx","wy","wz"]
    target_cols = ["u","v","w","p","q","r"]
    norm = {"inputs": stats(df_train, input_cols),
            "targets": stats(df_train, target_cols),
            "note": "computed on TRAIN split only"}
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm, f, indent=2)

    # ---------- Console summary ----------
    n_run_tr = df_train["run_id"].nunique() if len(df_train) else 0
    n_run_va = df_val["run_id"].nunique()   if len(df_val)   else 0
    print(f"Wrote: {out_dir/'train.csv'}  rows={len(df_train)} runs={n_run_tr}")
    print(f"Wrote: {out_dir/'val.csv'}    rows={len(df_val)} runs={n_run_va}")
    print(f"Wrote: {out_dir/'manifest.csv'}")
    if len(df_val) == 0:
        print("WARNING: val is still empty. Consider lowering --val_ratio or adding more data.")


if __name__ == "__main__":
    main()

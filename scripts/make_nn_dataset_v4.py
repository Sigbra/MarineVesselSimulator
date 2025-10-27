#!/usr/bin/env python3
"""
Create a tidy NN training/validation dataset (v4) from simulated boat logs.

Inputs (sim CSVs, no header):
  col 0                -> t
  col 1..12            -> x.transpose()
                          x ordering assumed: [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
  col 43..45           -> imu.accel(0..2)   (ax, ay, az)
  col 46..48           -> imu.gyro(0..2)    (wx, wy, wz)

IMPORTANT (heading to NN):
- The NN expects heading encoded as (cpsi, spsi) = (cos(psi), sin(psi)).
- Use **previous-timestep ground-truth heading** per run segment to form (cpsi, spsi).
  We shift psi_gt by +1 sample within each run_id, then drop the first row of each run.

Outputs (CSV with header):
  out_dir/all_runs.csv  columns:
    [run_id, t, ax, ay, az, wx, wy, wz, cpsi, spsi, Vn, Ve, p, q, r, phi, theta]
  out_dir/train.csv
  out_dir/val.csv
  out_dir/manifest.csv
  out_dir/norm_stats.json (computed on TRAIN only)

Usage (from repo root):
  python3 scripts/make_nn_dataset_v4.py \
      --data_dir data/ \
      --out_dir  data/nn_dataset_v4 \
      --pattern  "simdata*.csv" \
      --train_ratio 0.8 --seed 123

Frames & rotation:
- Body: x forward, y starboard, z down
- END : x east, y north, z down
- v_nav = Rzyx(phi, theta, psi) * v_body, with R rows [East; North; Down]:

  R(0,:) = [ spsi*cth,               cpsi*cphi + sphi*sth*spsi,  -cpsi*sphi + sth*spsi*cphi ]
  R(1,:) = [ cpsi*cth,              -spsi*cphi + cpsi*sth*sphi,   spsi*sphi + cpsi*cphi*sth ]
  R(2,:) = [ -sth,                                  cth*sphi,                        cth*cphi ]

We compute Ve, Vn as row·[u v w].

TODO: Remove duplicate rows at the end of simdata before making datasets!
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# --------------- CONFIG: map your x[1..12] to fields ----------------
# x = [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
X_INDEX = {
    "u": 1, "v": 2, "w": 3,
    "p": 4, "q": 5, "r": 6,
    "phi": 10, "theta": 11, "psi": 12,
}

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


def _x_at_block(df: pd.DataFrame, k1: int) -> pd.Series:
    """Return series for 1-based index k1 inside x block (csv cols 1..12)."""
    x_cols = df.iloc[:, X_START:X_END+1]
    return x_cols.iloc[:, k1-1]


def _compute_nav_vel(u: pd.Series, v: pd.Series, w: pd.Series,
                     phi: pd.Series, theta: pd.Series, psi: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Compute navigation-frame horizontal velocities (Vn, Ve) from body-frame
    velocities and Euler ZYX (roll=phi, pitch=theta, yaw=psi) using the provided R.
    Returns (Vn, Ve).
    """
    cphi = np.cos(phi); sphi = np.sin(phi)
    cth  = np.cos(theta); sth = np.sin(theta)
    cpsi = np.cos(psi); spsi = np.sin(psi)

    Ve = (
        spsi*cth * u
        + (cpsi*cphi + sphi*sth*spsi) * v
        + (-cpsi*sphi + sth*spsi*cphi) * w
    )

    Vn = (
        cpsi*cth * u
        + (-spsi*cphi + cpsi*sth*sphi) * v
        + (spsi*sphi + cpsi*cphi*sth) * w
    )

    return Vn, Ve


def extract_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract required columns and compute Vn, Ve. Also keep psi_gt for later shift."""
    ax, ay, az = df.iloc[:, COL_AX], df.iloc[:, COL_AY], df.iloc[:, COL_AZ]
    wx, wy, wz = df.iloc[:, COL_WX], df.iloc[:, COL_WY], df.iloc[:, COL_WZ]

    u     = _x_at_block(df, X_INDEX["u"]).astype(float)
    v     = _x_at_block(df, X_INDEX["v"]).astype(float)
    w     = _x_at_block(df, X_INDEX["w"]).astype(float)
    p     = _x_at_block(df, X_INDEX["p"]).astype(float)
    q     = _x_at_block(df, X_INDEX["q"]).astype(float)
    r     = _x_at_block(df, X_INDEX["r"]).astype(float)
    phi   = _x_at_block(df, X_INDEX["phi"]).astype(float)
    theta = _x_at_block(df, X_INDEX["theta"]).astype(float)
    psi_gt= _x_at_block(df, X_INDEX["psi"]).astype(float)

    Vn, Ve = _compute_nav_vel(u, v, w, phi, theta, psi_gt)

    out = pd.DataFrame({
        "t":    df.iloc[:, COL_T].astype(float),
        "ax":   ax.astype(float), "ay": ay.astype(float), "az": az.astype(float),
        "wx":   wx.astype(float), "wy": wy.astype(float), "wz": wz.astype(float),
        # keep ground-truth heading for later shifting
        "psi_gt": psi_gt,
        # targets
        "Vn": Vn, "Ve": Ve,
        "p": p, "q": q, "r": r,
        "phi": phi, "theta": theta,
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


def contiguous_intra_split(df_run: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one run into train/val by time (contiguous)."""
    n = len(df_run)
    if n < 2:
        return df_run.iloc[0:0].copy(), df_run.iloc[0:0].copy()
    n_val = max(1, int(round(val_ratio * n)))
    n_val = min(n_val, n-1)
    cut = n - n_val
    tr = df_run.iloc[:cut].reset_index(drop=True)
    va = df_run.iloc[cut:].reset_index(drop=True)
    return tr, va


def add_prev_heading_cs_and_drop_first(df: pd.DataFrame) -> pd.DataFrame:
    """Within each run_id segment, create cpsi/spsi from psi_gt.shift(1),
    then drop the first row (where they are NaN). Removes psi_gt afterwards."""
    def _per_run(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        psi_prev = g["psi_gt"].shift(1)
        g["cpsi"] = np.cos(psi_prev)
        g["spsi"] = np.sin(psi_prev)
        g = g.dropna(subset=["cpsi","spsi"]).reset_index(drop=True)
        return g
    if "run_id" in df.columns:
        df = df.groupby("run_id", as_index=False, group_keys=False).apply(_per_run)
    else:
        df = _per_run(df)
    return df.drop(columns=["psi_gt"])  # remove helper column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="Folder with sim CSV(s); default = <repo>/data")
    ap.add_argument("--pattern", default="simdata*.csv", help="Glob, e.g. simdata*.csv or simdata.csv")
    ap.add_argument("--out_dir", default=None, help="Output folder; default = <repo>/data/nn_dataset_v4")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="File-level split ratio (used when >1 file)")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Fallback intra-run ratio (contiguous tail)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # ---------- Resolve paths relative to repo root (parent of scripts/) ----------
    script_dir = Path(__file__).resolve().parent          # <repo>/scripts
    repo_root  = script_dir.parent                        # <repo>
    default_data_dir = repo_root / "data"
    default_out_dir  = default_data_dir / "nn_dataset_v4"

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

    # Try file-level split if multiple files; otherwise intra-split
    if len(files) > 1:
        train_files, val_files = split_by_files(files, args.train_ratio, args.seed)
    else:
        train_files, val_files = files, []  # single file → intra-split below

    manifest_rows = []
    frames_train, frames_val = [], []

    # ---------- Process each file ----------
    for f in files:
        df_raw = load_sim_csv(f)
        df_run = extract_df(df_raw)
        base_id = f.stem
        dt = compute_dt(df_run["t"])

        goes_to_val = (f in val_files) and (len(df_run) > 0)

        if goes_to_val:
            df_val = df_run.copy()
            df_val.insert(0, "run_id", base_id)
            frames_val.append(df_val)
            manifest_rows.append({"run_id": base_id, "file": str(f), "rows": len(df_run), "dt_median": dt, "split": "val"})
        else:
            if len(files) == 1 or len(val_files) == 0:
                # Single file or empty val split → contiguous intra-run split (tail to val)
                tr, va = contiguous_intra_split(df_run, args.val_ratio)
                tr_id = f"{base_id}_tr"
                va_id = f"{base_id}_va"
                if len(tr) > 0:
                    tr = tr.copy(); tr.insert(0, "run_id", tr_id); frames_train.append(tr)
                    manifest_rows.append({"run_id": tr_id, "file": str(f), "rows": len(tr), "dt_median": dt, "split": "train"})
                if len(va) > 0:
                    va = va.copy(); va.insert(0, "run_id", va_id); frames_val.append(va)
                    manifest_rows.append({"run_id": va_id, "file": str(f), "rows": len(va), "dt_median": dt, "split": "val"})
            else:
                df_tr = df_run.copy()
                df_tr.insert(0, "run_id", base_id)
                frames_train.append(df_tr)
                manifest_rows.append({"run_id": base_id, "file": str(f), "rows": len(df_run), "dt_median": dt, "split": "train"})

    # ---------- Concatenate splits ----------
    cols_out = [
        "run_id","t","ax","ay","az","wx","wy","wz","cpsi","spsi",
        "Vn","Ve","p","q","r","phi","theta"
    ]

    df_train = pd.concat(frames_train, axis=0, ignore_index=True) if frames_train else pd.DataFrame()
    df_val   = pd.concat(frames_val,   axis=0, ignore_index=True) if frames_val   else pd.DataFrame()

    # Compute cpsi/spsi from previous psi_gt within each run_id and drop first rows
    if len(df_train): df_train = add_prev_heading_cs_and_drop_first(df_train)
    if len(df_val):   df_val   = add_prev_heading_cs_and_drop_first(df_val)

    # If val still empty, split the largest train run, then recompute shift
    if len(df_val) == 0 and len(df_train) > 0:
        print("Note: val split empty; performing intra-run split on largest train run.")
        counts = df_train.groupby("run_id").size().sort_values(ascending=False)
        rid = counts.index[0]
        df_run = df_train[df_train["run_id"] == rid].drop(columns=["run_id"]).reset_index(drop=True)
        tr, va = contiguous_intra_split(df_run, args.val_ratio)
        dt = compute_dt(df_run["t"])  # dt before dropping first rows again
        # Rebuild train/val
        df_train = df_train[df_train["run_id"] != rid]
        tr.insert(0, "run_id", f"{rid}_tr2")
        va.insert(0, "run_id", f"{rid}_va2")
        tr = add_prev_heading_cs_and_drop_first(tr)
        va = add_prev_heading_cs_and_drop_first(va)
        df_train = pd.concat([df_train, tr], axis=0, ignore_index=True)
        df_val   = va
        manifest_rows.append({"run_id": f"{rid}_tr2", "file": "(from intra-split)", "rows": len(tr), "dt_median": dt, "split": "train"})
        manifest_rows.append({"run_id": f"{rid}_va2", "file": "(from intra-split)", "rows": len(va), "dt_median": dt, "split": "val"})

    # Final column selection/order
    df_train = df_train[cols_out] if len(df_train) else pd.DataFrame(columns=cols_out)
    df_val   = df_val[cols_out]   if len(df_val)   else pd.DataFrame(columns=cols_out)

    # ---------- Save outputs ----------
    df_all = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    (out_dir / "all_runs.csv").write_text(df_all.to_csv(index=False))
    (out_dir / "train.csv").write_text(df_train.to_csv(index=False))
    (out_dir / "val.csv").write_text(df_val.to_csv(index=False))
    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    # ---------- Norm stats (TRAIN only) ----------
    def stats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
        if len(df) == 0:
            return {"mean": {c: 0.0 for c in cols}, "std": {c: 1.0 for c in cols}}
        mu = df[cols].mean().to_dict()
        sd = (df[cols].std(ddof=0).replace(0.0, 1e-8)).to_dict()
        return {"mean": mu, "std": sd}

    input_cols  = ["ax","ay","az","wx","wy","wz","cpsi","spsi"]
    target_cols = ["Vn","Ve","p","q","r","phi","theta"]
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsedAlog CSV → plot bundle

Reads a parsedAlog_*.csv produced by your extractor and writes svg plots to --out.

Plots generated:
  1) Desired Tau (X, Y, N)
  2) N1 + N2
  3) Alpha1 + Alpha2
  4) Track (NAV_X vs NAV_Y) with heading arrows
  5) Angles: phi + theta + psi (NAV_ROLL, NAV_PITCH, NAV_YAW)
  6) Rates: phi rate + theta rate + psi rate (NAV_ROLL_RATE, NAV_PITCH_RATE, NAV_YAW_RATE)
  7) END velocities: Ve, Vn computed from NAV_SPEED + NAV_HEADING
     (heading convention assumed: 0 = North, clockwise positive → Ve = U*sin(hdg), Vn = U*cos(hdg))

Notes:
  • Timestamp is assumed to be ISO 8601 in 'timestamp' column (with 'Z' OK).
  • Missing columns are handled gracefully: plot is skipped with a warning.
  • Angle unit inference: if values look like degrees (|max| > ~ 2π), they are converted to radians
    for computations (heading arrows + Ve/Vn). Raw plotted angles are NOT auto-converted.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- utilities -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV '{path}': {e}") from e
    if "timestamp" not in df.columns:
        raise RuntimeError("CSV missing required column 'timestamp'.")
    return df


def parse_time(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Returns:
      t_s: seconds since start (float array)
      t_dt: pandas datetime series (timezone-aware if possible)
    """
    # Accept timestamps like "...Z"
    t_dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if t_dt.isna().all():
        raise RuntimeError("Could not parse any values in 'timestamp' as datetime.")
    t0 = t_dt.iloc[~t_dt.isna().to_numpy()].iloc[0]
    t_s = (t_dt - t0).dt.total_seconds().to_numpy(dtype=float)
    return t_s, t_dt


def has_cols(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)


def to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def infer_degrees(series: np.ndarray) -> bool:
    """
    Heuristic: treat as degrees if typical magnitude exceeds ~2π.
    """
    s = series[np.isfinite(series)]
    if s.size == 0:
        return False
    p99 = np.nanpercentile(np.abs(s), 99)
    return p99 > (2.0 * np.pi * 1.2)


def heading_to_rad(h: np.ndarray) -> np.ndarray:
    """
    Convert heading array to radians if it appears to be in degrees.
    """
    if infer_degrees(h):
        return np.deg2rad(h)
    return h


def save_fig(out_dir: str, stem: str, name: str) -> str:
    out_path = os.path.join(out_dir, f"{stem}_{name}.svg")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


# ----------------------------- plotting -----------------------------

def plot_desired_tau(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
    cols = ["DESIRED_TAU_X", "DESIRED_TAU_Y", "DESIRED_TAU_N"]
    if not has_cols(df, cols):
        warn(f"Skipping Desired Tau plot (missing one of {cols}).")
        return None

    x = to_numeric(df[cols[0]])
    y = to_numeric(df[cols[1]])
    n = to_numeric(df[cols[2]])

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, x, label="DESIRED_TAU_X")
    plt.plot(t_s, y, label="DESIRED_TAU_Y")
    plt.plot(t_s, n, label="DESIRED_TAU_N")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Tau (units as logged)")
    plt.title("Desired Tau")
    return save_fig(out_dir, stem, "desired_tau_xyz")


def plot_n1_n2(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
    cols = ["N1", "N2"]
    if not has_cols(df, cols):
        warn(f"Skipping N1+N2 plot (missing one of {cols}).")
        return None

    n1 = to_numeric(df["N1"])
    n2 = to_numeric(df["N2"])

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, n1, label="N1")
    plt.plot(t_s, n2, label="N2")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Command (units as logged)")
    plt.title("Thruster commands: N1 + N2")
    return save_fig(out_dir, stem, "n1_n2")


def plot_alpha1_alpha2(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
    cols = ["ALPHA1", "ALPHA2"]
    if not has_cols(df, cols):
        warn(f"Skipping Alpha1+Alpha2 plot (missing one of {cols}).")
        return None

    a1 = to_numeric(df["ALPHA1"])
    a2 = to_numeric(df["ALPHA2"])

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, a1, label="ALPHA1")
    plt.plot(t_s, a2, label="ALPHA2")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Angle (units as logged)")
    plt.title("Azimuth angles: Alpha1 + Alpha2")
    return save_fig(out_dir, stem, "alpha1_alpha2")


def plot_track_with_heading(df: pd.DataFrame, out_dir: str, stem: str, arrow_count: int = 50) -> Optional[str]:
    cols = ["NAV_X", "NAV_Y", "NAV_HEADING"]
    if not has_cols(df, cols):
        warn(f"Skipping track plot (missing one of {cols}).")
        return None

    x = to_numeric(df["NAV_X"])
    y = to_numeric(df["NAV_Y"])
    hdg = heading_to_rad(to_numeric(df["NAV_HEADING"]))

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(hdg)
    if np.count_nonzero(ok) < 2:
        warn("Skipping track plot (not enough finite NAV_X/NAV_Y/NAV_HEADING samples).")
        return None

    x = x[ok]
    y = y[ok]
    hdg = hdg[ok]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, linewidth=1.5, label="Track (NAV_X, NAV_Y)")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("East (NAV_X)")
    plt.ylabel("North (NAV_Y)")
    plt.title("Track with heading arrows")

    # Arrows: heading 0 = North, clockwise positive
    # Convert heading to direction vector in EN:
    #   Ve dir = sin(hdg), Vn dir = cos(hdg)
    n = x.size
    step = max(1, n // max(1, arrow_count))
    idx = np.arange(0, n, step, dtype=int)

    # Scale arrow length relative to path span
    span = max(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y))
    L = 0.03 * span if np.isfinite(span) and span > 0 else 1.0

    u = L * np.sin(hdg[idx])  # East component
    v = L * np.cos(hdg[idx])  # North component
    plt.quiver(x[idx], y[idx], u, v, angles="xy", scale_units="xy", scale=1.0, width=0.003)

    plt.legend()
    return save_fig(out_dir, stem, "track_heading")


def plot_angles(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
    cols = ["NAV_ROLL", "NAV_PITCH", "NAV_YAW"]
    if not has_cols(df, cols):
        warn(f"Skipping phi/theta/psi plot (missing one of {cols}).")
        return None

    phi = to_numeric(df["NAV_ROLL"])
    theta = to_numeric(df["NAV_PITCH"])
    psi = to_numeric(df["NAV_YAW"])

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, phi, label="phi (NAV_ROLL)")
    plt.plot(t_s, theta, label="theta (NAV_PITCH)")
    plt.plot(t_s, psi, label="psi (NAV_YAW)")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Angle (units as logged)")
    plt.title("Angles: phi + theta + psi")
    return save_fig(out_dir, stem, "angles_phi_theta_psi")


def plot_rates(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
    cols = ["NAV_ROLL_RATE", "NAV_PITCH_RATE", "NAV_YAW_RATE"]
    if not has_cols(df, cols):
        warn(f"Skipping p/q/r plot (missing one of {cols}).")
        return None

    p = to_numeric(df["NAV_ROLL_RATE"])
    q = to_numeric(df["NAV_PITCH_RATE"])
    r = to_numeric(df["NAV_YAW_RATE"])

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, p, label="phi rate (NAV_ROLL_RATE)")
    plt.plot(t_s, q, label="theta rate (NAV_PITCH_RATE)")
    plt.plot(t_s, r, label="psi rate (NAV_YAW_RATE)")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Rate (units as logged)")
    plt.title("Rates: phi rate + theta rate + psi rate")
    return save_fig(out_dir, stem, "rates_phi_theta_psi")


def plot_end_velocities(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
    cols = ["NAV_SPEED", "NAV_HEADING"]
    if not has_cols(df, cols):
        warn(f"Skipping END velocities plot (missing one of {cols}).")
        return None

    spd = to_numeric(df["NAV_SPEED"])
    hdg = heading_to_rad(to_numeric(df["NAV_HEADING"]))

    # Ve/Vn from speed + heading (0 = North, clockwise positive)
    ve = spd * np.sin(hdg)
    vn = spd * np.cos(hdg)

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, ve, label="Ve (East) = speed*sin(heading)")
    plt.plot(t_s, vn, label="Vn (North) = speed*cos(heading)")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Velocity (units of NAV_SPEED)")
    plt.title("END velocities from speed + heading")
    return save_fig(out_dir, stem, "end_velocities_ve_vn")


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot bundle for parsedAlog CSV.")
    ap.add_argument(
        "--input",
        required=True,
        help="Path to parsedAlog_*.csv",
    )
    ap.add_argument(
        "--out",
        default="data/a_path/",
        help="Output folder for plots (created if missing).",
    )
    ap.add_argument(
        "--arrow-count",
        type=int,
        default=50,
        help="Approximate number of heading arrows on the track plot.",
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[FEIL] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = str(Path(args.out))
    ensure_dir(out_dir)

    df = safe_read_csv(str(in_path))
    t_s, _t_dt = parse_time(df)

    stem = in_path.stem  # e.g., parsedAlog_13_11_2025...
    saved = []

    saved.append(plot_desired_tau(df, t_s, out_dir, stem))
    saved.append(plot_n1_n2(df, t_s, out_dir, stem))
    saved.append(plot_alpha1_alpha2(df, t_s, out_dir, stem))
    saved.append(plot_track_with_heading(df, out_dir, stem, arrow_count=int(args.arrow_count)))
    saved.append(plot_angles(df, t_s, out_dir, stem))
    saved.append(plot_rates(df, t_s, out_dir, stem))
    saved.append(plot_end_velocities(df, t_s, out_dir, stem))

    saved = [p for p in saved if p]
    print(f"Done. Wrote {len(saved)} plot(s) to: {out_dir}")
    for p in saved:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

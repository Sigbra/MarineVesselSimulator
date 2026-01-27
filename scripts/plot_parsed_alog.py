#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsedAlog CSV → plot bundle

Reads a parsedAlog_*.csv produced by your extractor and writes svg plots to --out.

Plots generated:
  1) Desired Tau (X, Y, N)
  2) N1 + N2
  3) Alpha1 + Alpha2 (converted from rad → deg for plotting)
  4) Track (NAV_X vs NAV_Y) with heading arrows
  5) Angles: phi + theta + psi (NAV_ROLL, NAV_PITCH, NAV_YAW)
  6) Rates: phi rate + theta rate + psi rate (NAV_ROLL_RATE, NAV_PITCH_RATE, NAV_YAW_RATE)
  7) END velocities: Ve, Vn computed from NAV_SPEED + NAV_HEADING
     (heading convention assumed: 0 = North, clockwise positive → Ve = U*sin(hdg), Vn = U*cos(hdg))

Time windowing:
  • You can limit the plotted interval with --startTime and --endTime.
  • Each can be either:
      - seconds since the first valid CSV timestamp (float), e.g. 120 or 120.5
      - an ISO 8601 datetime string, e.g. 2025-11-13T10:15:30Z
  • If omitted, the interval is unbounded on that side.

Notes:
  • Timestamp is assumed to be ISO 8601 in 'timestamp' column (with 'Z' OK).
  • Missing columns are handled gracefully: plot is skipped with a warning.
  • Angle unit inference: if values look like degrees (|max| > ~ 2π), they are converted to radians
    for computations (heading arrows + Ve/Vn). Raw plotted angles are NOT auto-converted
    (except ALPHA1/ALPHA2 which are explicitly converted rad → deg).
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


def parse_time(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, pd.Timestamp]:
    """
    Returns:
      t_s: seconds since first valid timestamp (float array; NaN where timestamp invalid)
      t_dt: pandas datetime series (timezone-aware; NaT where invalid)
      t0: first valid timestamp (UTC)
    """
    t_dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if t_dt.isna().all():
        raise RuntimeError("Could not parse any values in 'timestamp' as datetime.")

    # first valid timestamp
    t0 = t_dt.iloc[~t_dt.isna().to_numpy()].iloc[0]

    # seconds since t0; keep NaN for NaT entries
    t_s = (t_dt - t0).dt.total_seconds().to_numpy(dtype=float)
    return t_s, t_dt, t0


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


def parse_time_arg_to_seconds(val: Optional[str], t0: pd.Timestamp) -> Optional[float]:
    """
    Interprets a CLI time argument as either:
      - seconds since t0 (float)
      - ISO 8601 datetime string (converted to seconds since t0)

    Returns seconds since t0, or None if val is None.
    """
    if val is None:
        return None

    # Try as seconds (float)
    try:
        s = float(val)
        if np.isfinite(s):
            return s
    except Exception:
        pass

    # Try as datetime
    dt = pd.to_datetime(val, errors="coerce", utc=True)
    if pd.isna(dt):
        raise ValueError(
            f"Could not parse time value '{val}'. Use seconds (e.g. 120.5) or ISO 8601 (e.g. 2025-11-13T10:15:30Z)."
        )
    return float((dt - t0).total_seconds())


def apply_time_window(
    df: pd.DataFrame,
    t_s: np.ndarray,
    start_s: Optional[float],
    end_s: Optional[float],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filters df and t_s to the interval [start_s, end_s]. None means unbounded.
    Keeps t_s as "seconds since original t0" (does not re-zero).
    """
    mask = np.isfinite(t_s)

    if start_s is not None:
        mask &= (t_s >= start_s)
    if end_s is not None:
        mask &= (t_s <= end_s)

    df_f = df.loc[mask].reset_index(drop=True)
    t_s_f = t_s[mask].astype(float, copy=False)
    return df_f, t_s_f


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

    # Logged as radians → convert to degrees for plotting
    a1_deg = np.rad2deg(to_numeric(df["ALPHA1"]))
    a2_deg = np.rad2deg(to_numeric(df["ALPHA2"]))

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, a1_deg, label="ALPHA1 [deg]")
    plt.plot(t_s, a2_deg, label="ALPHA2 [deg]")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Azimuth angles: Alpha1 + Alpha2 (rad → deg)")
    return save_fig(out_dir, stem, "alpha1_alpha2_deg")


def plot_track_with_heading(
    df: pd.DataFrame,
    out_dir: str,
    stem: str,
    arrow_count: int = 50,
    max_step_m: float = 0.1,
) -> Optional[str]:
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

    # -------------------- hard pen-lift on step > max_step_m --------------------
    # NOTE: This assumes NAV_X/NAV_Y are in meters. If they are not, adjust max_step_m accordingly.
    if x.size >= 2 and max_step_m is not None:
        dx = np.diff(x)
        dy = np.diff(y)
        step = np.hypot(dx, dy)

        jump = np.isfinite(step) & (step > float(max_step_m))
        if np.any(jump):
            x = x.copy()
            y = y.copy()
            hdg = hdg.copy()
            # Break at the second point of each oversized step
            x[1:][jump] = np.nan
            y[1:][jump] = np.nan
            hdg[1:][jump] = np.nan

    # Finite samples after pen-lift for arrows
    ok2 = np.isfinite(x) & np.isfinite(y) & np.isfinite(hdg)
    x2, y2, hdg2 = x[ok2], y[ok2], hdg[ok2]
    if x2.size < 2:
        warn("Skipping track plot (not enough finite samples after max-step filtering).")
        return None

    # -------------------- plot --------------------
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, linewidth=1.5, label="Track (NAV_X, NAV_Y)")  # breaks at NaNs
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("East (NAV_X)")
    plt.ylabel("North (NAV_Y)")
    plt.title(f"Track with heading arrows")

    # Arrows (0 = North, clockwise positive): East=sin(hdg), North=cos(hdg)
    n = x2.size
    step_idx = max(1, n // max(1, arrow_count))
    idx = np.arange(0, n, step_idx, dtype=int)

    span = max(np.nanmax(x2) - np.nanmin(x2), np.nanmax(y2) - np.nanmin(y2))
    L = 0.03 * span if np.isfinite(span) and span > 0 else 1.0

    u = L * np.sin(hdg2[idx])
    v = L * np.cos(hdg2[idx])
    plt.quiver(x2[idx], y2[idx], u, v, angles="xy", scale_units="xy", scale=1.0, width=0.003)

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
    ap.add_argument(
        "--startTime",
        default=None,
        help="Start of plotted interval. Seconds since log start (e.g. 120.5) OR ISO 8601 (e.g. 2025-11-13T10:15:30Z).",
    )
    ap.add_argument(
        "--endTime",
        default=None,
        help="End of plotted interval. Seconds since log start (e.g. 300) OR ISO 8601 (e.g. 2025-11-13T10:20:00Z).",
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[FEIL] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = str(Path(args.out))
    ensure_dir(out_dir)

    df = safe_read_csv(str(in_path))
    t_s, _t_dt, t0 = parse_time(df)

    # Apply time window, if any
    try:
        start_s = parse_time_arg_to_seconds(args.startTime, t0)
        end_s = parse_time_arg_to_seconds(args.endTime, t0)
    except ValueError as e:
        print(f"[FEIL] {e}", file=sys.stderr)
        sys.exit(2)

    if start_s is not None and end_s is not None and start_s > end_s:
        print("[FEIL] --startTime must be <= --endTime.", file=sys.stderr)
        sys.exit(2)

    df, t_s = apply_time_window(df, t_s, start_s, end_s)

    if len(df) < 2:
        warn("Time window produced fewer than 2 samples. No plots written.")
        sys.exit(0)

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


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# parsedAlog CSV → plot bundle

# Reads a parsedAlog_*.csv produced by your extractor and writes svg plots to --out.

# Plots generated:
#   1) Desired Tau (X, Y, N)
#   2) N1 + N2
#   3) Alpha1 + Alpha2
#   4) Track (NAV_X vs NAV_Y) with heading arrows
#   5) Angles: phi + theta + psi (NAV_ROLL, NAV_PITCH, NAV_YAW)
#   6) Rates: phi rate + theta rate + psi rate (NAV_ROLL_RATE, NAV_PITCH_RATE, NAV_YAW_RATE)
#   7) END velocities: Ve, Vn computed from NAV_SPEED + NAV_HEADING
#      (heading convention assumed: 0 = North, clockwise positive → Ve = U*sin(hdg), Vn = U*cos(hdg))

# Notes:
#   • Timestamp is assumed to be ISO 8601 in 'timestamp' column (with 'Z' OK).
#   • Missing columns are handled gracefully: plot is skipped with a warning.
#   • Angle unit inference: if values look like degrees (|max| > ~ 2π), they are converted to radians
#     for computations (heading arrows + Ve/Vn). Raw plotted angles are NOT auto-converted.
# """

# import argparse
# import os
# import sys
# from pathlib import Path
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # ----------------------------- utilities -----------------------------

# def ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)


# def safe_read_csv(path: str) -> pd.DataFrame:
#     try:
#         df = pd.read_csv(path)
#     except Exception as e:
#         raise RuntimeError(f"Could not read CSV '{path}': {e}") from e
#     if "timestamp" not in df.columns:
#         raise RuntimeError("CSV missing required column 'timestamp'.")
#     return df


# def parse_time(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
#     """
#     Returns:
#       t_s: seconds since start (float array)
#       t_dt: pandas datetime series (timezone-aware if possible)
#     """
#     # Accept timestamps like "...Z"
#     t_dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
#     if t_dt.isna().all():
#         raise RuntimeError("Could not parse any values in 'timestamp' as datetime.")
#     t0 = t_dt.iloc[~t_dt.isna().to_numpy()].iloc[0]
#     t_s = (t_dt - t0).dt.total_seconds().to_numpy(dtype=float)
#     return t_s, t_dt


# def has_cols(df: pd.DataFrame, cols) -> bool:
#     return all(c in df.columns for c in cols)


# def to_numeric(series: pd.Series) -> np.ndarray:
#     return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


# def infer_degrees(series: np.ndarray) -> bool:
#     """
#     Heuristic: treat as degrees if typical magnitude exceeds ~2π.
#     """
#     s = series[np.isfinite(series)]
#     if s.size == 0:
#         return False
#     p99 = np.nanpercentile(np.abs(s), 99)
#     return p99 > (2.0 * np.pi * 1.2)


# def heading_to_rad(h: np.ndarray) -> np.ndarray:
#     """
#     Convert heading array to radians if it appears to be in degrees.
#     """
#     if infer_degrees(h):
#         return np.deg2rad(h)
#     return h


# def save_fig(out_dir: str, stem: str, name: str) -> str:
#     out_path = os.path.join(out_dir, f"{stem}_{name}.svg")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     return out_path


# def warn(msg: str) -> None:
#     print(f"[WARN] {msg}", file=sys.stderr)


# # ----------------------------- plotting -----------------------------

# def plot_desired_tau(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
#     cols = ["DESIRED_TAU_X", "DESIRED_TAU_Y", "DESIRED_TAU_N"]
#     if not has_cols(df, cols):
#         warn(f"Skipping Desired Tau plot (missing one of {cols}).")
#         return None

#     x = to_numeric(df[cols[0]])
#     y = to_numeric(df[cols[1]])
#     n = to_numeric(df[cols[2]])

#     plt.figure(figsize=(12, 5))
#     plt.plot(t_s, x, label="DESIRED_TAU_X")
#     plt.plot(t_s, y, label="DESIRED_TAU_Y")
#     plt.plot(t_s, n, label="DESIRED_TAU_N")
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel("t [s]")
#     plt.ylabel("Tau (units as logged)")
#     plt.title("Desired Tau")
#     return save_fig(out_dir, stem, "desired_tau_xyz")


# def plot_n1_n2(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
#     cols = ["N1", "N2"]
#     if not has_cols(df, cols):
#         warn(f"Skipping N1+N2 plot (missing one of {cols}).")
#         return None

#     n1 = to_numeric(df["N1"])
#     n2 = to_numeric(df["N2"])

#     plt.figure(figsize=(12, 5))
#     plt.plot(t_s, n1, label="N1")
#     plt.plot(t_s, n2, label="N2")
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel("t [s]")
#     plt.ylabel("Command (units as logged)")
#     plt.title("Thruster commands: N1 + N2")
#     return save_fig(out_dir, stem, "n1_n2")


# def plot_alpha1_alpha2(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
#     cols = ["ALPHA1", "ALPHA2"]
#     if not has_cols(df, cols):
#         warn(f"Skipping Alpha1+Alpha2 plot (missing one of {cols}).")
#         return None

#     a1 = to_numeric(df["ALPHA1"])
#     a2 = to_numeric(df["ALPHA2"])

#     plt.figure(figsize=(12, 5))
#     plt.plot(t_s, a1, label="ALPHA1")
#     plt.plot(t_s, a2, label="ALPHA2")
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel("t [s]")
#     plt.ylabel("Angle (units as logged)")
#     plt.title("Azimuth angles: Alpha1 + Alpha2")
#     return save_fig(out_dir, stem, "alpha1_alpha2")


# def plot_track_with_heading(df: pd.DataFrame, out_dir: str, stem: str, arrow_count: int = 50) -> Optional[str]:
#     cols = ["NAV_X", "NAV_Y", "NAV_HEADING"]
#     if not has_cols(df, cols):
#         warn(f"Skipping track plot (missing one of {cols}).")
#         return None

#     x = to_numeric(df["NAV_X"])
#     y = to_numeric(df["NAV_Y"])
#     hdg = heading_to_rad(to_numeric(df["NAV_HEADING"]))

#     ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(hdg)
#     if np.count_nonzero(ok) < 2:
#         warn("Skipping track plot (not enough finite NAV_X/NAV_Y/NAV_HEADING samples).")
#         return None

#     x = x[ok]
#     y = y[ok]
#     hdg = hdg[ok]

#     plt.figure(figsize=(8, 8))
#     plt.plot(x, y, linewidth=1.5, label="Track (NAV_X, NAV_Y)")
#     plt.grid(True)
#     plt.axis("equal")
#     plt.xlabel("East (NAV_X)")
#     plt.ylabel("North (NAV_Y)")
#     plt.title("Track with heading arrows")

#     # Arrows: heading 0 = North, clockwise positive
#     # Convert heading to direction vector in EN:
#     #   Ve dir = sin(hdg), Vn dir = cos(hdg)
#     n = x.size
#     step = max(1, n // max(1, arrow_count))
#     idx = np.arange(0, n, step, dtype=int)

#     # Scale arrow length relative to path span
#     span = max(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y))
#     L = 0.03 * span if np.isfinite(span) and span > 0 else 1.0

#     u = L * np.sin(hdg[idx])  # East component
#     v = L * np.cos(hdg[idx])  # North component
#     plt.quiver(x[idx], y[idx], u, v, angles="xy", scale_units="xy", scale=1.0, width=0.003)

#     plt.legend()
#     return save_fig(out_dir, stem, "track_heading")


# def plot_angles(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
#     cols = ["NAV_ROLL", "NAV_PITCH", "NAV_YAW"]
#     if not has_cols(df, cols):
#         warn(f"Skipping phi/theta/psi plot (missing one of {cols}).")
#         return None

#     phi = to_numeric(df["NAV_ROLL"])
#     theta = to_numeric(df["NAV_PITCH"])
#     psi = to_numeric(df["NAV_YAW"])

#     plt.figure(figsize=(12, 5))
#     plt.plot(t_s, phi, label="phi (NAV_ROLL)")
#     plt.plot(t_s, theta, label="theta (NAV_PITCH)")
#     plt.plot(t_s, psi, label="psi (NAV_YAW)")
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel("t [s]")
#     plt.ylabel("Angle (units as logged)")
#     plt.title("Angles: phi + theta + psi")
#     return save_fig(out_dir, stem, "angles_phi_theta_psi")


# def plot_rates(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
#     cols = ["NAV_ROLL_RATE", "NAV_PITCH_RATE", "NAV_YAW_RATE"]
#     if not has_cols(df, cols):
#         warn(f"Skipping p/q/r plot (missing one of {cols}).")
#         return None

#     p = to_numeric(df["NAV_ROLL_RATE"])
#     q = to_numeric(df["NAV_PITCH_RATE"])
#     r = to_numeric(df["NAV_YAW_RATE"])

#     plt.figure(figsize=(12, 5))
#     plt.plot(t_s, p, label="phi rate (NAV_ROLL_RATE)")
#     plt.plot(t_s, q, label="theta rate (NAV_PITCH_RATE)")
#     plt.plot(t_s, r, label="psi rate (NAV_YAW_RATE)")
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel("t [s]")
#     plt.ylabel("Rate (units as logged)")
#     plt.title("Rates: phi rate + theta rate + psi rate")
#     return save_fig(out_dir, stem, "rates_phi_theta_psi")


# def plot_end_velocities(df: pd.DataFrame, t_s: np.ndarray, out_dir: str, stem: str) -> Optional[str]:
#     cols = ["NAV_SPEED", "NAV_HEADING"]
#     if not has_cols(df, cols):
#         warn(f"Skipping END velocities plot (missing one of {cols}).")
#         return None

#     spd = to_numeric(df["NAV_SPEED"])
#     hdg = heading_to_rad(to_numeric(df["NAV_HEADING"]))

#     # Ve/Vn from speed + heading (0 = North, clockwise positive)
#     ve = spd * np.sin(hdg)
#     vn = spd * np.cos(hdg)

#     plt.figure(figsize=(12, 5))
#     plt.plot(t_s, ve, label="Ve (East) = speed*sin(heading)")
#     plt.plot(t_s, vn, label="Vn (North) = speed*cos(heading)")
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel("t [s]")
#     plt.ylabel("Velocity (units of NAV_SPEED)")
#     plt.title("END velocities from speed + heading")
#     return save_fig(out_dir, stem, "end_velocities_ve_vn")


# # ----------------------------- main -----------------------------

# def main() -> None:
#     ap = argparse.ArgumentParser(description="Plot bundle for parsedAlog CSV.")
#     ap.add_argument(
#         "--input",
#         required=True,
#         help="Path to parsedAlog_*.csv",
#     )
#     ap.add_argument(
#         "--out",
#         default="data/a_path/",
#         help="Output folder for plots (created if missing).",
#     )
#     ap.add_argument(
#         "--arrow-count",
#         type=int,
#         default=50,
#         help="Approximate number of heading arrows on the track plot.",
#     )

#     args = ap.parse_args()

#     in_path = Path(args.input)
#     if not in_path.exists():
#         print(f"[FEIL] Input file not found: {args.input}", file=sys.stderr)
#         sys.exit(1)

#     out_dir = str(Path(args.out))
#     ensure_dir(out_dir)

#     df = safe_read_csv(str(in_path))
#     t_s, _t_dt = parse_time(df)

#     stem = in_path.stem  # e.g., parsedAlog_13_11_2025...
#     saved = []

#     saved.append(plot_desired_tau(df, t_s, out_dir, stem))
#     saved.append(plot_n1_n2(df, t_s, out_dir, stem))
#     saved.append(plot_alpha1_alpha2(df, t_s, out_dir, stem))
#     saved.append(plot_track_with_heading(df, out_dir, stem, arrow_count=int(args.arrow_count)))
#     saved.append(plot_angles(df, t_s, out_dir, stem))
#     saved.append(plot_rates(df, t_s, out_dir, stem))
#     saved.append(plot_end_velocities(df, t_s, out_dir, stem))

#     saved = [p for p in saved if p]
#     print(f"Done. Wrote {len(saved)} plot(s) to: {out_dir}")
#     for p in saved:
#         print(f"  - {p}")


# if __name__ == "__main__":
#     main()

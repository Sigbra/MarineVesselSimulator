#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alog → CSV extractor (MOOS topics)

Reads a MOOS *.alog file and writes a CSV with columns:
    timestamp + TARGETS (in the order given below)

Behavior:
  • Streams the .alog line-by-line (no RAM blow-ups).
  • Uses the `LOGSTART` unix epoch found in the alog header as the absolute start time, and adds each row's "time since start" to get ISO 8601 UTC for each sample.
  • No filename parsing is used; `LOGSTART` is the sole source of absolute time.
  • Writes a row whenever any TARGET variable updates, carrying forward the last known values for the others (opt-out with --require-all).
  • Optional live plot (--display) for thruster channels (N1/N2/ALPHA1/ALPHA2).

Notes on case:
  • MOOS variables can be case-sensitive, but .alog content is commonly uppercase.
  • This parser matches topics case-insensitively by uppercasing the topic from file and the targets internally.
  • The CSV header preserves the exact spellings in TARGETS_ORDERED (e.g., NAV_u, NAV_v),
    while matching against NAV_U / NAV_V in the log if needed.
"""

import argparse
import csv
import datetime as dt
import os
import sys
from math import pi
from typing import Optional

try:
    import matplotlib.pyplot as plt  # only used with --display
except Exception:
    plt = None  # type: ignore


def parse_logstart_epoch(path: str) -> Optional[dt.datetime]:
    """Scan early lines for LOGSTART <epoch> and return UTC datetime if present."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(400):  # scan first few hundred lines
                line = fh.readline()
                if not line:
                    break
                if "LOGSTART" in line:
                    # example: '%% LOGSTART           1758524469.654832'
                    parts = line.strip().split()
                    try:
                        epoch = float(parts[-1])
                        return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
                    except Exception:
                        return None
    except OSError:
        return None
    return None


def iso_utc(ts: dt.datetime) -> str:
    return ts.replace(tzinfo=dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


# ------------------------------- targets -------------------------------
# Preserve *column order* by using an ordered list (not a set).
TARGETS_ORDERED = [
    "N1",
    "N2",
    "ALPHA1",
    "ALPHA2",
    "DESIRED_TAU_X",
    "DESIRED_TAU_Y",
    "DESIRED_TAU_N",
    "NAV_SPEED",
    "NAV_u",
    "NAV_v",
    "NAV_HEAVE",
    "NAV_ROLL_RATE",
    "NAV_PITCH_RATE",
    "NAV_YAW_RATE",
    "NAV_X",
    "NAV_Y",
    "NAV_DEPTH",
    "NAV_ROLL",
    "NAV_PITCH",
    "NAV_YAW",
    "NAV_HEADING",
    "NAV_COURSE",
]

# Case-insensitive matching: compare uppercase.
TARGETS_U = {t.upper() for t in TARGETS_ORDERED}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract selected MOOS topics from .alog to CSV.")
    ap.add_argument(
        "--input",
        default="RAN_Data/TestData5/MOOSLog_13_11_2025_____09_53_05/MOOSLog_13_11_2025_____09_53_05",
        help="Path to .alog file",
    )
    ap.add_argument(
        "--out-csv",
        default="MarineVesselSimulator/data/Data_plots_final/REAL5/parsedAlog.csv",
        help="Destination CSV (will be overwritten).",
    )

    # Row policy
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="Only write a row when *all* target fields have been observed at least once.",
    )

    # Source filtering / debugging
    ap.add_argument(
        "--publisher",
        default=None,
        help="Only accept rows from this publisher (e.g., app_podSIM).",
    )
    ap.add_argument(
        "--echo",
        type=int,
        default=0,
        help="Echo the first K accepted rows (time, topic, publisher, value) for verification.",
    )

    # Display
    ap.add_argument("--display", action="store_true", help="Show live plot while parsing (thruster channels only).")
    ap.add_argument("--plot-window", type=float, default=10.0, help="Seconds in rolling window.")
    ap.add_argument("--interval", type=float, default=0.02, help="Pause between plot updates (s).")
    ap.add_argument("--stride", type=int, default=2, help="Plot every Nth written row.")

    args = ap.parse_args()

    # Resolve start datetime
    start_dt = parse_logstart_epoch(args.input)
    if not start_dt:
        print("[FEIL] Could not read LOGSTART from alog header.", file=sys.stderr)
        sys.exit(1)

    # Prepare output CSV
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        out_fh = open(args.out_csv, "w", newline="", buffering=1)
        writer = csv.writer(out_fh)
        writer.writerow(["timestamp"] + TARGETS_ORDERED)
    except OSError as e:
        print(f"[FEIL] Could not open output '{args.out_csv}': {e}", file=sys.stderr)
        sys.exit(1)

    # Optional plotting setup (still focused on thrusters)
    do_plot = bool(args.display and plt is not None)
    if args.display and plt is None:
        print("[ADVARSEL] Matplotlib not available; --display ignored.")
        do_plot = False

    if do_plot:
        plt.ion()
        fig, (ax_n, ax_a) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        fig.suptitle("Alog thrusters – rolling window")

        (l_n1,) = ax_n.plot([], [], label="N1")
        (l_n2,) = ax_n.plot([], [], label="N2")
        ax_n.set_ylabel("N1/N2")
        ax_n.set_ylim(-1, 1)
        ax_n.legend(loc="upper right")
        ax_n.grid(True)

        (l_a1,) = ax_a.plot([], [], label="ALPHA1")
        (l_a2,) = ax_a.plot([], [], label="ALPHA2")
        ax_a.set_ylabel("alpha")
        ax_a.set_ylim(-pi / 2, pi / 2)
        ax_a.set_xlabel("Time (UTC epoch seconds)")
        ax_a.legend(loc="upper right")
        ax_a.grid(True)

        t_hist = []
        n1_hist, n2_hist, a1_hist, a2_hist = [], [], [], []
        # Rolling index pointer (more efficient than scanning from 0 each time)
        roll_i0 = 0

    # State: store last values keyed by UPPERCASE topic
    last = {t.upper(): None for t in TARGETS_ORDERED}
    written = 0

    # Read & parse
    with open(args.input, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            if not raw:
                break
            line = raw.rstrip("\n")
            if not line or line.startswith("%"):
                continue

            # Expect: <dt_since_start>  <TOPIC>  <publisher>  <value...>
            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                t_rel = float(parts[0])
            except ValueError:
                continue

            topic_u = parts[1].upper()
            publisher = parts[2]

            if args.publisher and publisher != args.publisher:
                continue
            if topic_u not in TARGETS_U:
                continue

            # Value: use last token as float (works for standard numeric MOOS vars)
            try:
                val = float(parts[-1])
            except ValueError:
                continue

            # Optional echo for verification
            if args.echo and args.echo > 0:
                print(f"{t_rel:10.5f}  {topic_u:16s}  {publisher:>24s}  {val}")
                args.echo -= 1

            # Update state
            last[topic_u] = val

            # Row policy
            have_all = all(v is not None for v in last.values())
            if args.require_all and not have_all:
                continue

            # Absolute timestamp
            t_abs = start_dt + dt.timedelta(seconds=t_rel)
            ts_iso = iso_utc(t_abs)

            # Write row in the requested order (preserve original header spellings)
            row = [ts_iso]
            for col in TARGETS_ORDERED:
                v = last[col.upper()]
                row.append("" if v is None else f"{v:.6f}")
            writer.writerow(row)
            written += 1

            # Plotting (downsampled) – thruster channels only
            if do_plot and (written % max(1, int(args.stride)) == 0):
                t_epoch = t_abs.timestamp()
                t_hist.append(t_epoch)

                def _val_or_nan(k: str) -> float:
                    vv = last.get(k.upper())
                    return float("nan") if vv is None else float(vv)

                n1_hist.append(_val_or_nan("N1"))
                n2_hist.append(_val_or_nan("N2"))
                a1_hist.append(_val_or_nan("ALPHA1"))
                a2_hist.append(_val_or_nan("ALPHA2"))

                # Maintain rolling start index
                t_min = t_epoch - float(args.plot_window)
                while roll_i0 < len(t_hist) and t_hist[roll_i0] < t_min:
                    roll_i0 += 1

                tx = t_hist[roll_i0:]
                l_n1.set_data(tx, n1_hist[roll_i0:])
                l_n2.set_data(tx, n2_hist[roll_i0:])
                l_a1.set_data(tx, a1_hist[roll_i0:])
                l_a2.set_data(tx, a2_hist[roll_i0:])

                for ax in (ax_n, ax_a):
                    ax.set_xlim(t_min, t_epoch)
                    ax.relim()
                    ax.autoscale_view(scalex=True, scaley=False)

                plt.pause(float(args.interval))

    out_fh.close()
    print(f"Ferdig. Skrev {written} rader til {args.out_csv}.")
    if do_plot and plt is not None:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Alog → CSV extractor (thruster channels)

# Reads a MOOS *.alog file and writes a CSV with columns:
#     timestamp, N1, N2, alpha1, alpha2

# Behavior (similar vibe to the SeaPath file parser you asked for):
#   • Streams the .alog line-by-line (no RAM blow-ups).
#   • Uses the `LOGSTART` unix epoch found in the alog header as the absolute start time, and adds each row's (float) "time since start" to get ISO 8601 UTC for each sample.
#   • No filename parsing is used; `LOGSTART` is the sole source of absolute time.
#   • Writes a row whenever any of the four variables updates, carrying forward
#     the last known values for the others (opt-out with --require-all).
#   • Optional live plot (--display) with a rolling window.

# Defaults:
#   --input     TestData/MOOSLog.alog
#   --out-csv   MarineVesselSimulator/data/nn_dataset_v9_real/parsedAlog.csv

# Notes:
#   • Input file is opened read-only and never modified.
#   • The output file is created/overwritten (parent folders auto-created).
# """

# import argparse
# import csv
# import datetime as dt
# import os
# import re
# import sys
# from typing import Optional
# from math import pi

# try:
#     import matplotlib.pyplot as plt  # only used with --display
# except Exception:
#     plt = None  # type: ignore

# # ------------------------------- filename → start time -------------------------------

# _FN_RE = re.compile(
#     r"MOOSLog_(?P<day>\d{1,2})_(?P<mon>\d{1,2})_(?P<year>\d{4})_+"  # many underscores
#     r"(?P<hh>\d{1,2})_(?P<mm>\d{1,2})_(?P<ss>\d{1,2})",
#     re.IGNORECASE,
# )


# def parse_start_dt_from_filename(path: str) -> Optional[dt.datetime]:
#     base = os.path.basename(path)
#     m = _FN_RE.search(base)
#     if not m:
#         # Sometimes the parent directory holds the pattern
#         parent = os.path.basename(os.path.dirname(path))
#         m = _FN_RE.search(parent)
#     if not m:
#         return None
#     day = int(m.group('day')); mon = int(m.group('mon')); year = int(m.group('year'))
#     hh = int(m.group('hh')); mm = int(m.group('mm')); ss = int(m.group('ss'))
#     try:
#         return dt.datetime(year, mon, day, hh, mm, ss, tzinfo=dt.timezone.utc)
#     except ValueError:
#         return None


# def parse_logstart_epoch(path: str) -> Optional[dt.datetime]:
#     """Scan early lines for LOGSTART <epoch> and return UTC datetime if present."""
#     try:
#         with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
#             for _ in range(200):  # scan first few hundred lines
#                 line = fh.readline()
#                 if not line:
#                     break
#                 if 'LOGSTART' in line:
#                     # example: '%% LOGSTART           1758524469.654832'
#                     parts = line.strip().split()
#                     try:
#                         epoch = float(parts[-1])
#                         return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
#                     except Exception:
#                         return None
#     except OSError:
#         return None
#     return None

# # ------------------------------- parser -------------------------------

# # Topics of interest (uppercased)
# TARGETS = {"N1", "N2", "ALPHA1", "ALPHA2"}


# def iso_utc(ts: dt.datetime) -> str:
#     return ts.replace(tzinfo=dt.timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


# def main():
#     ap = argparse.ArgumentParser(description="Extract N1/N2/alpha1/alpha2 from .alog to CSV.")
#     ap.add_argument('--input', default='RAN_Data/TestData5/MOOSLog_13_11_2025_____09_53_05/MOOSLog_13_11_2025_____09_53_05', help='Path to .alog file')
#     ap.add_argument('--out-csv', default='MarineVesselSimulator/data/Data_plots_final/REAL5/parsedAlog.csv',
#                    help='Destination CSV (will be overwritten).')

#     # Row policy
#     ap.add_argument('--require-all', action='store_true',
#                    help='Only write a row when all four fields are available at that timestamp.')

#     # Source filtering / debugging
#     ap.add_argument('--publisher', default=None,
#                    help='Only accept rows from this publisher (e.g., app_podSIM).')
#     ap.add_argument('--echo', type=int, default=0,
#                    help='Echo the first K accepted rows (time, topic, publisher, value) for verification.')

#     # Display
#     ap.add_argument('--display', action='store_true', help='Show live plot while parsing.')
#     ap.add_argument('--plot-window', type=float, default=10.0, help='Seconds in rolling window.')
#     ap.add_argument('--interval', type=float, default=0.02, help='Pause between plot updates (s).')
#     ap.add_argument('--stride', type=int, default=2, help='Plot every Nth written row.')

#     args = ap.parse_args()

#     # Resolve start datetime
#     start_dt = parse_logstart_epoch(args.input)
#     if not start_dt:
#         print('[FEIL] Could not read LOGSTART from alog header.', file=sys.stderr)
#         sys.exit(1)

#     # Prepare output CSV
#     out_dir = os.path.dirname(args.out_csv)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#     try:
#         out_fh = open(args.out_csv, 'w', newline='', buffering=1)
#         writer = csv.writer(out_fh)
#         writer.writerow(['timestamp', 'N1', 'N2', 'alpha1', 'alpha2'])
#     except OSError as e:
#         print(f"[FEIL] Could not open output '{args.out_csv}': {e}", file=sys.stderr)
#         sys.exit(1)

#     # Optional plotting setup
#     do_plot = bool(args.display and plt is not None)
#     if args.display and plt is None:
#         print('[ADVARSEL] Matplotlib not available; --display ignored.')
#         do_plot = False

#     if do_plot:
#         plt.ion()
#         fig, (ax_n, ax_a) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
#         fig.suptitle('Alog thrusters – rolling window')
#         # Lines
#         (l_n1,) = ax_n.plot([], [], label='N1')
#         (l_n2,) = ax_n.plot([], [], label='N2')
#         ax_n.set_ylabel('N1/N2'); ax_n.set_ylim(-1, 1); ax_n.legend(loc='upper right'); ax_n.grid(True)
#         (l_a1,) = ax_a.plot([], [], label='alpha1')
#         (l_a2,) = ax_a.plot([], [], label='alpha2')
#         ax_a.set_ylabel('alpha'); ax_a.set_ylim(-pi/2, pi/2); ax_a.set_xlabel('Time (UTC)'); ax_a.legend(loc='upper right'); ax_a.grid(True)
#         t_hist = []  # seconds since epoch
#         n1_hist = []; n2_hist = []; a1_hist = []; a2_hist = []

#     # State
#     last = { 'N1': None, 'N2': None, 'ALPHA1': None, 'ALPHA2': None }
#     written = 0

#     # Read & parse
#     with open(args.input, 'r', encoding='utf-8', errors='ignore') as fh:
#         for raw in fh:
#             if not raw:
#                 break
#             line = raw.rstrip('\n')
#             if not line or line.startswith('%'):
#                 continue

#             # Expect: <dt_since_start>  <TOPIC>  <publisher>  <value...>
#             # Split on whitespace for first three columns, then leave remainder as value
#             parts = line.split()
#             if len(parts) < 4:
#                 continue
#             try:
#                 t_rel = float(parts[0])
#             except ValueError:
#                 continue
#             topic = parts[1].upper()
#             publisher = parts[2]
#             if args.publisher and publisher != args.publisher:
#                 continue
#             if topic not in TARGETS:
#                 continue

#             # Value
#             try:
#                 val_str = parts[-1]
#                 val = float(val_str)
#             except ValueError:
#                 continue

#             # Optional echo for verification
#             if args.echo and args.echo > 0:
#                 print(f"{t_rel:10.5f}  {topic:7s}  {publisher:>24s}  {val}")
#                 args.echo -= 1

#             # Update state and decide whether to write
#             last[topic] = val

#             have_all = all(last[k] is not None for k in ('N1', 'N2', 'ALPHA1', 'ALPHA2'))
#             if args.require_all and not have_all:
#                 continue

#             # Absolute timestamp
#             t_abs = start_dt + dt.timedelta(seconds=t_rel)
#             ts_iso = iso_utc(t_abs)

#             # Write row (carry forward known values; leave blanks for Nones if not require_all)
#             row = [
#                 ts_iso,
#                 '' if last['N1'] is None else f"{last['N1']:.6f}",
#                 '' if last['N2'] is None else f"{last['N2']:.6f}",
#                 '' if last['ALPHA1'] is None else f"{last['ALPHA1']:.6f}",
#                 '' if last['ALPHA2'] is None else f"{last['ALPHA2']:.6f}",
#             ]
#             writer.writerow(row)
#             written += 1

#             # Plotting (downsampled)
#             if do_plot and (written % max(1, int(args.stride)) == 0):
#                 t_hist.append(t_abs.timestamp())
#                 n1_hist.append(last['N1'] if last['N1'] is not None else float('nan'))
#                 n2_hist.append(last['N2'] if last['N2'] is not None else float('nan'))
#                 a1_hist.append(last['ALPHA1'] if last['ALPHA1'] is not None else float('nan'))
#                 a2_hist.append(last['ALPHA2'] if last['ALPHA2'] is not None else float('nan'))

#                 # rolling window
#                 t_end = t_hist[-1]
#                 t_min = t_end - float(args.plot_window)
#                 # find start index
#                 start_idx = 0
#                 for i, tt in enumerate(t_hist):
#                     if tt >= t_min:
#                         start_idx = i; break
#                 tx = t_hist[start_idx:]
#                 l_n1.set_data(tx, n1_hist[start_idx:])
#                 l_n2.set_data(tx, n2_hist[start_idx:])
#                 l_a1.set_data(tx, a1_hist[start_idx:])
#                 l_a2.set_data(tx, a2_hist[start_idx:])
#                 for ax in (ax_n, ax_a):
#                     ax.set_xlim(t_min, t_end)
#                     ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
#                 plt.pause(float(args.interval))

#     out_fh.close()
#     print(f"Ferdig. Skrev {written} rader til {args.out_csv}.")
#     if do_plot and plt is not None:
#         plt.ioff(); plt.show()


# if __name__ == '__main__':
#     main()


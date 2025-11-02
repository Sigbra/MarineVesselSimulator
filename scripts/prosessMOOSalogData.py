#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alog → CSV extractor (thruster channels)

Reads a MOOS *.alog file and writes a CSV with columns:
    timestamp, N1, N2, alpha1, alpha2

Behavior (similar vibe to the SeaPath file parser you asked for):
  • Streams the .alog line-by-line (no RAM blow-ups).
  • Uses the `LOGSTART` unix epoch found in the alog header as the absolute start time, and adds each row's (float) "time since start" to get ISO 8601 UTC for each sample.
  • No filename parsing is used; `LOGSTART` is the sole source of absolute time.
  • Writes a row whenever any of the four variables updates, carrying forward
    the last known values for the others (opt-out with --require-all).
  • Optional live plot (--display) with a rolling window.

Defaults:
  --input     TestData/MOOSLog.alog
  --out-csv   MarineVesselSimulator/data/nn_dataset_v9_real/parsedAlog.csv

Notes:
  • Input file is opened read-only and never modified.
  • The output file is created/overwritten (parent folders auto-created).
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys
from typing import Optional
from math import pi

try:
    import matplotlib.pyplot as plt  # only used with --display
except Exception:
    plt = None  # type: ignore

# ------------------------------- filename → start time -------------------------------

_FN_RE = re.compile(
    r"MOOSLog_(?P<day>\d{1,2})_(?P<mon>\d{1,2})_(?P<year>\d{4})_+"  # many underscores
    r"(?P<hh>\d{1,2})_(?P<mm>\d{1,2})_(?P<ss>\d{1,2})",
    re.IGNORECASE,
)


def parse_start_dt_from_filename(path: str) -> Optional[dt.datetime]:
    base = os.path.basename(path)
    m = _FN_RE.search(base)
    if not m:
        # Sometimes the parent directory holds the pattern
        parent = os.path.basename(os.path.dirname(path))
        m = _FN_RE.search(parent)
    if not m:
        return None
    day = int(m.group('day')); mon = int(m.group('mon')); year = int(m.group('year'))
    hh = int(m.group('hh')); mm = int(m.group('mm')); ss = int(m.group('ss'))
    try:
        return dt.datetime(year, mon, day, hh, mm, ss, tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def parse_logstart_epoch(path: str) -> Optional[dt.datetime]:
    """Scan early lines for LOGSTART <epoch> and return UTC datetime if present."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            for _ in range(200):  # scan first few hundred lines
                line = fh.readline()
                if not line:
                    break
                if 'LOGSTART' in line:
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

# ------------------------------- parser -------------------------------

# Topics of interest (uppercased)
TARGETS = {"N1", "N2", "ALPHA1", "ALPHA2"}


def iso_utc(ts: dt.datetime) -> str:
    return ts.replace(tzinfo=dt.timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def main():
    ap = argparse.ArgumentParser(description="Extract N1/N2/alpha1/alpha2 from .alog to CSV.")
    ap.add_argument('--input', default='TestData/MOOSLog.alog', help='Path to .alog file')
    ap.add_argument('--out-csv', default='MarineVesselSimulator/data/nn_dataset_v9_real/parsedAlog.csv',
                   help='Destination CSV (will be overwritten).')

    # Row policy
    ap.add_argument('--require-all', action='store_true',
                   help='Only write a row when all four fields are available at that timestamp.')

    # Source filtering / debugging
    ap.add_argument('--publisher', default=None,
                   help='Only accept rows from this publisher (e.g., app_podSIM).')
    ap.add_argument('--echo', type=int, default=0,
                   help='Echo the first K accepted rows (time, topic, publisher, value) for verification.')

    # Display
    ap.add_argument('--display', action='store_true', help='Show live plot while parsing.')
    ap.add_argument('--plot-window', type=float, default=10.0, help='Seconds in rolling window.')
    ap.add_argument('--interval', type=float, default=0.02, help='Pause between plot updates (s).')
    ap.add_argument('--stride', type=int, default=2, help='Plot every Nth written row.')

    args = ap.parse_args()

    # Resolve start datetime
    start_dt = parse_logstart_epoch(args.input)
    if not start_dt:
        print('[FEIL] Could not read LOGSTART from alog header.', file=sys.stderr)
        sys.exit(1)

    # Prepare output CSV
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        out_fh = open(args.out_csv, 'w', newline='', buffering=1)
        writer = csv.writer(out_fh)
        writer.writerow(['timestamp', 'N1', 'N2', 'alpha1', 'alpha2'])
    except OSError as e:
        print(f"[FEIL] Could not open output '{args.out_csv}': {e}", file=sys.stderr)
        sys.exit(1)

    # Optional plotting setup
    do_plot = bool(args.display and plt is not None)
    if args.display and plt is None:
        print('[ADVARSEL] Matplotlib not available; --display ignored.')
        do_plot = False

    if do_plot:
        plt.ion()
        fig, (ax_n, ax_a) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        fig.suptitle('Alog thrusters – rolling window')
        # Lines
        (l_n1,) = ax_n.plot([], [], label='N1')
        (l_n2,) = ax_n.plot([], [], label='N2')
        ax_n.set_ylabel('N1/N2'); ax_n.set_ylim(-1, 1); ax_n.legend(loc='upper right'); ax_n.grid(True)
        (l_a1,) = ax_a.plot([], [], label='alpha1')
        (l_a2,) = ax_a.plot([], [], label='alpha2')
        ax_a.set_ylabel('alpha'); ax_a.set_ylim(-pi/2, pi/2); ax_a.set_xlabel('Time (UTC)'); ax_a.legend(loc='upper right'); ax_a.grid(True)
        t_hist = []  # seconds since epoch
        n1_hist = []; n2_hist = []; a1_hist = []; a2_hist = []

    # State
    last = { 'N1': None, 'N2': None, 'ALPHA1': None, 'ALPHA2': None }
    written = 0

    # Read & parse
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            if not raw:
                break
            line = raw.rstrip('\n')
            if not line or line.startswith('%'):
                continue

            # Expect: <dt_since_start>  <TOPIC>  <publisher>  <value...>
            # Split on whitespace for first three columns, then leave remainder as value
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t_rel = float(parts[0])
            except ValueError:
                continue
            topic = parts[1].upper()
            publisher = parts[2]
            if args.publisher and publisher != args.publisher:
                continue
            if topic not in TARGETS:
                continue

            # Value
            try:
                val_str = parts[-1]
                val = float(val_str)
            except ValueError:
                continue

            # Optional echo for verification
            if args.echo and args.echo > 0:
                print(f"{t_rel:10.5f}  {topic:7s}  {publisher:>24s}  {val}")
                args.echo -= 1

            # Update state and decide whether to write
            last[topic] = val

            have_all = all(last[k] is not None for k in ('N1', 'N2', 'ALPHA1', 'ALPHA2'))
            if args.require_all and not have_all:
                continue

            # Absolute timestamp
            t_abs = start_dt + dt.timedelta(seconds=t_rel)
            ts_iso = iso_utc(t_abs)

            # Write row (carry forward known values; leave blanks for Nones if not require_all)
            row = [
                ts_iso,
                '' if last['N1'] is None else f"{last['N1']:.6f}",
                '' if last['N2'] is None else f"{last['N2']:.6f}",
                '' if last['ALPHA1'] is None else f"{last['ALPHA1']:.6f}",
                '' if last['ALPHA2'] is None else f"{last['ALPHA2']:.6f}",
            ]
            writer.writerow(row)
            written += 1

            # Plotting (downsampled)
            if do_plot and (written % max(1, int(args.stride)) == 0):
                t_hist.append(t_abs.timestamp())
                n1_hist.append(last['N1'] if last['N1'] is not None else float('nan'))
                n2_hist.append(last['N2'] if last['N2'] is not None else float('nan'))
                a1_hist.append(last['ALPHA1'] if last['ALPHA1'] is not None else float('nan'))
                a2_hist.append(last['ALPHA2'] if last['ALPHA2'] is not None else float('nan'))

                # rolling window
                t_end = t_hist[-1]
                t_min = t_end - float(args.plot_window)
                # find start index
                start_idx = 0
                for i, tt in enumerate(t_hist):
                    if tt >= t_min:
                        start_idx = i; break
                tx = t_hist[start_idx:]
                l_n1.set_data(tx, n1_hist[start_idx:])
                l_n2.set_data(tx, n2_hist[start_idx:])
                l_a1.set_data(tx, a1_hist[start_idx:])
                l_a2.set_data(tx, a2_hist[start_idx:])
                for ax in (ax_n, ax_a):
                    ax.set_xlim(t_min, t_end)
                    ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
                plt.pause(float(args.interval))

    out_fh.close()
    print(f"Ferdig. Skrev {written} rader til {args.out_csv}.")
    if do_plot and plt is not None:
        plt.ioff(); plt.show()


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline IMU parser – reads frames from a FILE (CSV/hex/bin), parses them,
optionally visualizes a rolling window, **and writes PARSED & SCALED data**
(only rows where checksum == OK) to a CSV.

Defaults:
  - input  : TestData/imu.csv
  - output : MarineVesselSimulator/data/nn_dataset_v9_real/parsedIMU.csv

Input file formats (choose via --input-format or let --input-format=auto detect):
  - csv : CSV with a column named 'raw_hex' (as produced by the live logger)
  - hex : Text file with one HEX line per frame (whitespace allowed)
  - bin : Binary file of concatenated 40-byte frames (each exactly 40 bytes)

Output CSV columns: [time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  - time   : UTC ISO 8601 from NTP-in-frame
  - gyro_* : **scaled** values in rad/s * (20*|g|)
  - acc_*  : **scaled** values = raw * (20*|g|)

Notes:
  - Scaling matches the original visualization: scale = 20 * |g| and we first
    convert gyro rad/s → rad/s, then apply the same scale.
  - Visualization can be toggled with --display (off by default).
"""

import argparse
import csv
import datetime as dt
import os
import struct
import sys
from math import isfinite, pi
from typing import Iterable, List, Tuple

# Import matplotlib lazily only if display is requested
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:  # Matplotlib may not be available in headless envs; guarded below
    plt = None  # type: ignore

NTP_UNIX_EPOCH_DIFF = 2208988800  # seconds from 1900-01-01 to 1970-01-01
RAD2DEG = 180.0 / pi              # gyro: rad/s -> rad/s
FRAME_LEN = 40                    # binary frame size (bytes)

# ------------------------------- Utils -------------------------------

def clean_hex(s: str) -> str:
    return ''.join(ch for ch in s.strip() if ch not in [' ', '\t', '\n', '\r'])

def hex_to_bytes(s: str) -> bytes:
    s = clean_hex(s)
    if len(s) % 2 != 0:
        raise ValueError(f"Odd number of hex digits: {len(s)} for string: {s[:32]}...")
    try:
        return bytes.fromhex(s)
    except ValueError as e:
        raise ValueError(f"Invalid hex: {s}") from e

def checksum_ok(frame: bytes) -> bool:
    if len(frame) < 5:
        return False
    calc = sum(frame[2:-1]) & 0xFF
    return calc == frame[-1]

def parse_time_from_frame(sec32: int, frac_ns32: int) -> Tuple[float, str, int, int]:
    """Sub-second is nanoseconds (0..999,999,999). Overflow handled."""
    carry_sec, ns = divmod(int(frac_ns32), 1_000_000_000)
    sec_total = int(sec32) + carry_sec
    sub_sec = ns / 1_000_000_000.0
    sub_ms = ns // 1_000_000
    sub_us = ns // 1_000

    if sec_total < NTP_UNIX_EPOCH_DIFF:
        unix_ts = sec_total + sub_sec
    else:
        unix_ts = (sec_total - NTP_UNIX_EPOCH_DIFF) + sub_sec

    try:
        iso = dt.datetime.utcfromtimestamp(unix_ts).isoformat(timespec="milliseconds") + "Z"
    except (OverflowError, OSError):
        iso = f"(timestamp out of range) unix={unix_ts:.6f}"

    return unix_ts, iso, int(sub_ms), int(sub_us)

def u32_to_float_be(u: int) -> float:
    return struct.unpack("!f", u.to_bytes(4, "big"))[0]

# ------------------------------- Parser (unchanged) -------------------------------

def parse_frame(frame: bytes) -> dict:
    """start, len, cmd, (sec, ns), [optional 0x00000001], 6x float32, checksum."""
    min_len = 3 + 8 + 6 * 4 + 1
    if len(frame) < min_len:
        raise ValueError(f"Frame too short: {len(frame)} bytes")

    start = frame[0]
    length = frame[1]
    cmd = frame[2]

    sec = int.from_bytes(frame[3:7], "big")
    ns  = int.from_bytes(frame[7:11], "big")
    unix_ts, iso, sub_ms, sub_us = parse_time_from_frame(sec, ns)

    idx = 11
    # Optional sentinel 0x00000001
    maybe_one = int.from_bytes(frame[idx:idx + 4], "big")
    if maybe_one == 1:
        idx += 4

    floats: List[float] = []
    for _ in range(6):
        u = int.from_bytes(frame[idx:idx + 4], "big")
        floats.append(u32_to_float_be(u))
        idx += 4

    extra = frame[idx:-1] if idx != len(frame) - 1 else b""

    gyro_rad = tuple(floats[:3])   # rad/s
    acc_raw  = tuple(floats[3:6])  # as delivered

    ok = checksum_ok(frame)

    return dict(
        start=start,
        length=length,
        command=cmd,
        seconds=sec,
        nanoseconds=ns,
        unix_ts=unix_ts,
        iso=iso,
        sub_ms=sub_ms,
        sub_us=sub_us,
        gyro_rad=gyro_rad,
        acc_raw=acc_raw,
        checksum_ok=ok,
        extra_bytes=extra,
        frame_len=len(frame),
    )

# ------------------------------- File source (→ HEX strings) -------------------------------

def _detect_input_format(path: str) -> str:
    # Text sniff
    try:
        with open(path, 'rb') as f:
            head = f.read(4096)
    except OSError:
        return 'hex'  # fallback

    try:
        text = head.decode('utf-8', errors='ignore')
        if text:
            first = (text.splitlines() or [''])[0].lower()
            if 'raw_hex' in first:
                return 'csv'
            # mostly hex-like -> treat as hex lines
            sample = ''.join(ch for ch in text if ch.strip())
            if sample and all(ch in '0123456789abcdefABCDEF,:;' for ch in sample):
                return 'hex'
    except Exception:
        pass

    # Binary size heuristic
    try:
        size = os.path.getsize(path)
        if size % FRAME_LEN == 0 and size > 0:
            return 'bin'
    except Exception:
        pass

    return 'hex'

def file_frames_as_hex(path: str, fmt: str = 'auto') -> Iterable[str]:
    if fmt == 'auto':
        fmt = _detect_input_format(path)
        print(f"[INFO] Auto-detected input format: {fmt}")

    if fmt == 'csv':
        with open(path, 'r', newline='') as fh:
            peek = fh.read(4096)
            fh.seek(0)
            if 'raw_hex' in (peek.splitlines()[0] if peek else '').lower():
                reader = csv.DictReader(fh)
                key = next((k for k in reader.fieldnames or [] if k.lower() == 'raw_hex'), None)
                if not key:
                    raise ValueError("CSV missing 'raw_hex' column header")
                for row in reader:
                    raw_hex = clean_hex(row.get(key, ''))
                    if raw_hex:
                        yield raw_hex
            else:
                # Fallback: simple CSV without a header; use last column
                fh.seek(0)
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    raw_hex = clean_hex(row[-1])
                    if raw_hex:
                        yield raw_hex
    elif fmt == 'hex':
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                s = clean_hex(line)
                if s:
                    yield s
    elif fmt == 'bin':
        with open(path, 'rb') as fh:
            i = 0
            while True:
                chunk = fh.read(FRAME_LEN)
                if not chunk:
                    break
                if len(chunk) != FRAME_LEN:
                    print(f"[ADVARSEL] Feil lengde {len(chunk)} på posisjon {i*FRAME_LEN} (forventet {FRAME_LEN}); ignorerer siste rest.")
                    break
                yield chunk.hex()
                i += 1
    else:
        raise ValueError(f"Unknown input format: {fmt}")

# ------------------------------- Main -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline parser (fra fil) – skriver PARSET & SKALERT CSV (checksum=OK).\n"
            "Støtter input: CSV med 'raw_hex', HEX-linjer, eller BIN (40-byte rammer)."
        )
    )
    parser.add_argument("--input", default="TestData/imu.csv",
                        help="Path to input file (csv/hex/bin). Default: TestData/imu.csv")
    parser.add_argument("--input-format", choices=["auto", "csv", "hex", "bin"], default="auto",
                        help="How to interpret the input file (default auto-detect).")
    parser.add_argument("--out-csv", default="MarineVesselSimulator/data/nn_dataset_v9_real/parsedIMU.csv",
                        help=("Destination CSV for parsed & scaled data. Parent dirs will be created.\n"
                              "Default: MarineVesselSimulator/data/nn_dataset_v9_real/parsedIMU.csv"))

    # Display toggle
    parser.add_argument("--display", action="store_true",
                        help="Show rolling visualization while parsing (off by default).")

    # Plot params (used only when --display)
    parser.add_argument("--interval", type=float, default=0.01,
                        help="Pause after update (s) for live-plot (display only).")
    parser.add_argument("--plot-window", type=float, default=5.0,
                        help="Width of rolling plot window in seconds (display only).")
    parser.add_argument("--stride", type=int, default=10,
                        help="Plot only every Nth sample (display only).")
    parser.add_argument("--stride-offset", type=int, default=0,
                        help="Plot-start offset (0-based). 0→0,10,20… (display only).")
    parser.add_argument("--plot-limit", type=int, default=0,
                        help="Stop plotting after N samples (0 = unlimited, display only).")

    # Scaling (applied to BOTH stored CSV and plotting)
    parser.add_argument("--g", type=float, default=9.81,
                        help="g in m/s^2; overall scale = 20*|g| for both gyro and acc.")

    args = parser.parse_args()

    # Final scale used for BOTH storage and plotting
    scale = 20.0 * abs(args.g)

    # Prepare output
    out_name = args.out_csv
    out_dir = os.path.dirname(out_name)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[FEIL] Could not create output directory '{out_dir}': {e}", file=sys.stderr)
            sys.exit(1)
    try:
        out_fh = open(out_name, mode="w", newline="", buffering=1)
        out_csv = csv.writer(out_fh)
        out_csv.writerow(["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])  # scaled
        print(f"[INFO] Writing PARSED & SCALED data (checksum=OK) to: {out_name}")
    except Exception as e:
        print(f"[FEIL] Could not open output file '{out_name}' for writing: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Scale = 20*|g| = {scale:.6f}")
    print(f"[INFO] Gyro stored as rad/s * scale; Acc stored as raw * scale.")

    # Optional plotting state
    plotting = bool(args.display and plt is not None)
    if args.display and plt is None:
        print("[ADVARSEL] Matplotlib not available; --display ignored.")
        plotting = False

    if plotting:
        plt.ion()
        fig, (ax_g, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Gyro (rad/s * {scale:.2f}) og Akselerometer (scaled, 20*g={scale:.2f}) – rullende {args.plot_window:.1f}s")
        (lgx,) = ax_g.plot([], [], label=f"gyro_x (rad/s * {scale:.2f})")
        (lgy,) = ax_g.plot([], [], label=f"gyro_y (rad/s * {scale:.2f})")
        (lgz,) = ax_g.plot([], [], label=f"gyro_z (rad/s * {scale:.2f})")
        ax_g.set_ylabel("Gyro (rad/s * scale)")
        ax_g.legend(loc="upper right"); ax_g.grid(True)
        (lax,) = ax_a.plot([], [], label=f"acc_x * {scale:.2f}")
        (lay,) = ax_a.plot([], [], label=f"acc_y * {scale:.2f}")
        (laz,) = ax_a.plot([], [], label=f"acc_z * {scale:.2f}")
        ax_a.set_ylabel("Acc (scaled)"); ax_a.set_xlabel("Tid (s, UNIX; fra NTP)")
        ax_a.legend(loc="upper right"); ax_a.grid(True)
        # histories
        ts_all: List[float] = []
        gdx_all: List[float] = []; gdy_all: List[float] = []; gdz_all: List[float] = []
        axx_all: List[float] = []; ayy_all: List[float] = []; azz_all: List[float] = []
        plotted = 0

    total_frames = 0
    written = 0

    try:
        for raw_hex in file_frames_as_hex(args.input, args.input_format):
            total_frames += 1
            try:
                frame = hex_to_bytes(raw_hex)
                parsed = parse_frame(frame)
            except Exception as e:
                print(f"[{total_frames}] Parse error: {e}", file=sys.stderr)
                continue

            if not parsed.get('checksum_ok'):
                continue  # only store checksum OK

            # ---- SCALE for storage (and potentially for plotting) ----
            ax_raw_x, ax_raw_y, ax_raw_z = parsed['acc_raw']
            gx_rad, gy_rad, gz_rad = parsed['gyro_rad']

            # Acc scaled (raw * scale)
            ax_s = ax_raw_x * scale
            ay_s = ax_raw_y * scale
            az_s = ax_raw_z * scale
            # Gyro: rad/s -> rad/s -> scale
            gx_s = gx_rad * scale
            gy_s = gy_rad * scale
            gz_s = gz_rad * scale

            out_csv.writerow([parsed['iso'], f"{ax_s:.9g}", f"{ay_s:.9g}", f"{az_s:.9g}",
                              f"{gx_s:.9g}", f"{gy_s:.9g}", f"{gz_s:.9g}"])
            written += 1

            # ---- Optional plotting ----
            if plotting:
                stride = max(1, int(args.stride))
                offset = max(0, int(args.stride_offset)) % stride
                # Plot only every Nth (stride)
                if ((total_frames - 1) % stride) != offset:
                    continue
                if args.plot_limit and plotted >= args.plot_limit:
                    continue

                t = parsed['unix_ts']
                ts_all.append(t)
                gdx_all.append(gx_s); gdy_all.append(gy_s); gdz_all.append(gz_s)
                axx_all.append(ax_s); ayy_all.append(ay_s); azz_all.append(az_s)

                t_min = t - float(args.plot_window)
                start_idx = 0
                for i, tt in enumerate(ts_all):
                    if tt >= t_min:
                        start_idx = i
                        break

                tsw  = ts_all[start_idx:]
                gdxw = gdx_all[start_idx:]; gdyw = gdy_all[start_idx:]; gdzw = gdz_all[start_idx:]
                axxw = axx_all[start_idx:]; ayyw = ayy_all[start_idx:]; azzw = azz_all[start_idx:]

                lgx.set_data(tsw, gdxw); lgy.set_data(tsw, gdyw); lgz.set_data(tsw, gdzw)
                lax.set_data(tsw, axxw); lay.set_data(tsw, ayyw); laz.set_data(tsw, azzw)

                # axis updates
                if plt is not None:
                    ax_g.set_xlim(t_min, t)
                    ax_a.set_xlim(t_min, t)
                    for ax in (ax_g, ax_a):
                        ax.relim(); ax.autoscale_view()
                    # compact status line
                    def fmt3(tup): return tuple(f"{v:+.6f}" if isfinite(v) else str(v) for v in tup)
                    print(
                        f"[{plotted}] time={parsed['iso']} (ms={parsed['sub_ms']}, us={parsed['sub_us']}) "
                        f"gyro_scaled(rad/s*{scale:.2f})={fmt3((gx_s, gy_s, gz_s))} "
                        f"acc_scaled={(f'{ax_s:+.6f}', f'{ay_s:+.6f}', f'{az_s:+.6f}')}"
                    )
                    plt.pause(float(args.interval))
                plotted += 1

    except KeyboardInterrupt:
        print("\nAvbrutt av bruker (Ctrl+C).")
    finally:
        try:
            out_fh.close()
        except Exception:
            pass
        print(f"Ferdig. Skrev {written} rader (checksum=OK) til {out_name}.")
        if plotting and plt is not None:
            plt.ioff(); plt.show()


if __name__ == "__main__":
    main()

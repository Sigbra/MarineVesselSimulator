#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SeaPath file parser – reads packets from a FILE (CSV/hex/bin), parses each
132-byte SeaPath KM Binary packet, optionally visualizes a rolling window,
and writes a CSV with ONLY the requested fields in this order:

[timestamp, roll, pitch, heading, heave,
 roll_rate, pitch_rate, yaw_rate,
 velocity_north, velocity_east, velocity_down]

Defaults:
  - input  : TestData/seapath.csv        (auto-detected format)
  - output : MarineVesselSimulator/data/nn_dataset_v9_real/parsedSeaPath.csv

Input file formats (choose via --input-format or let --input-format=auto):
  - csv : CSV with columns **timestamp, frame** (we ignore the CSV timestamp and parse the packet's own UTC time). The `frame` column must contain a **hex-encoded payload**. This can be either:
      • KM Binary (132 bytes) encoded as hex, or
      • Hex-encoded ASCII containing one or more NMEA sentences (e.g., $PSXN,23 / $PSXN,24 / $..RMC).
    For backward compatibility, a `raw_hex` column is also accepted.
  - hex : Text file with one HEX line per packet (whitespace allowed)
  - bin : Binary file of concatenated 132-byte packets

Notes:
  - When the payload is hex-encoded NMEA, we parse $PSXN,23 (attitude), $PSXN,24 (rates + vertical velocity), and $..RMC (course/speed/time) from the same payload to build one row.
  - No unit scaling or conversion is applied beyond obvious derivations (north/east velocity from course & speed; down = vertical_velocity from PSXN,24).
  - Visualization can be toggled with --display (off by default).
  - The input file is read-only; the script never modifies or deletes it.
  - No scaling or unit conversion is applied. Values are stored exactly as
    parsed from the packet (angles/rates/velocities as provided).
  - Visualization can be toggled with --display (off by default).
  - The input file is read-only; the script never modifies or deletes it.
  - No scaling or unit conversion is applied. Values are stored exactly as
    parsed from the packet (angles/rates/velocities as provided).
  - Visualization can be toggled with --display (off by default).
  - The input file is read-only; the script never modifies or deletes it.
"""

import argparse
import csv
import datetime as dt
import os
import struct
import sys
from typing import Iterable, List, Tuple

# Lazy import for plotting (only when --display)
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None  # type: ignore

# ------------------------------- SeaPath layout -------------------------------

# Struct from provided UDP parser
SEAPATH_STRUCT = struct.Struct("=4s H H I I I d d " + "f" * 21 + "I I f")
EXPECTED_SIZE = 132

FIELDS = [
    "magic", "length", "version", "utc_sec", "utc_nsec", "status",
    "latitude", "longitude", "ellipsoid_height",
    "roll", "pitch", "heading", "heave",
    "roll_rate", "pitch_rate", "yaw_rate",
    "velocity_north", "velocity_east", "velocity_down",
    "lat_error", "lon_error", "hgt_error",
    "roll_error", "pitch_error", "heading_error", "heave_error",
    "acc_north", "acc_east", "acc_down",
    "sec_delheave", "nsec_delheave", "delheave",
]

SELECTED_ORDER = [
    "timestamp",  # computed
    "roll", "pitch", "heading", "heave",
    "roll_rate", "pitch_rate", "yaw_rate",
    "velocity_north", "velocity_east", "velocity_down",
]

# ------------------------------- File helpers -------------------------------

def _clean_hex(s: str) -> str:
    """Remove spaces/tabs/newlines around hex, but keep hex digits."""
    if not isinstance(s, str):
        return ""
    return ''.join(ch for ch in s.strip() if ch not in (' ', '\t', '\n', '\r'))

def _only_hex(s: str) -> str:
    """Keep only hex chars (strips quotes, 0x, commas/semicolons/spaces, etc.)."""
    if not isinstance(s, str):
        return ""
    return ''.join(ch for ch in s if ch in '0123456789abcdefABCDEF')


def _detect_input_format(path: str) -> str:
    # Try quick sniff
    try:
        with open(path, 'rb') as f:
            head = f.read(4096)
    except OSError:
        return 'hex'

    # Heuristic: header mentions raw_hex -> CSV
    try:
        text = head.decode('utf-8', errors='ignore')
        if text:
            first = (text.splitlines() or [''])[0].lower()
            if ('frame' in first) or ('raw_hex' in first):
                return 'csv'
            # looks like hex text
            sample = ''.join(ch for ch in text if ch.strip())
            if sample and all(ch in '0123456789abcdefABCDEF,:;' for ch in sample):
                return 'hex'
    except Exception:
        pass

    # Binary size multiple of 132 -> bin
    try:
        size = os.path.getsize(path)
        if size % EXPECTED_SIZE == 0 and size > 0:
            return 'bin'
    except Exception:
        pass

    return 'hex'

def file_packets_as_bytes(path: str, fmt: str = 'auto') -> Iterable[bytes]:
    if fmt == 'auto':
        fmt = _detect_input_format(path)
        print(f"[INFO] Auto-detected input format: {fmt}")

    if fmt == 'csv':
        with open(path, 'r', newline='') as fh:
            peek = fh.read(4096)
            fh.seek(0)
            header = (peek.splitlines()[0] if peek else '')
            header_lower = header.lower()
            # Use DictReader if a known column name is present
            if ('frame' in header_lower) or ('raw_hex' in header_lower):
                # Try to detect delimiter for robustness (commas/semicolons)
                try:
                    dialect = csv.Sniffer().sniff(peek, delimiters=",;	")
                    fh.seek(0)
                    reader = csv.DictReader(fh, dialect=dialect)
                except Exception:
                    fh.seek(0)
                    reader = csv.DictReader(fh)
                orig_names = reader.fieldnames or []
                fieldnames = [ (k or '').strip() for k in orig_names ]
                # Map lowercase → original name
                lower_to_orig = { ((orig_names[i] or '').strip().lower()): orig_names[i] for i in range(len(orig_names)) }
                key = lower_to_orig.get('frame') or lower_to_orig.get('raw_hex')
                if not key:
                    # Fallback: choose last header column; downstream will attempt to parse
                    key = orig_names[-1] if orig_names else None
                for row in reader:
                    s = _only_hex(row.get(key, ''))
                    if len(s) % 2 == 1:
                        s = s[:-1]
                    if not s:
                        continue
                    try:
                        b = bytes.fromhex(s)
                    except ValueError:
                        continue
                    # Yield regardless of size; parser will decide (KM Binary vs NMEA)
                    yield b
            else:
                # Fallback: simple CSV without a header; use last column as hex
                try:
                    dialect = csv.Sniffer().sniff(peek, delimiters=",;	")
                    fh.seek(0)
                    reader = csv.reader(fh, dialect=dialect)
                except Exception:
                    fh.seek(0)
                    reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    s = _only_hex(row[-1])
                    if len(s) % 2 == 1:
                        s = s[:-1]
                    if not s:
                        continue
                    try:
                        b = bytes.fromhex(s)
                    except ValueError:
                        continue
                    yield b
    elif fmt == 'hex':
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                s = _only_hex(line)
                if len(s) % 2 == 1:
                    s = s[:-1]
                if not s:
                    continue
                try:
                    b = bytes.fromhex(s)
                except ValueError:
                    continue
                yield b
    elif fmt == 'bin':
        with open(path, 'rb') as fh:
            while True:
                chunk = fh.read(EXPECTED_SIZE)
                if not chunk:
                    break
                if len(chunk) != EXPECTED_SIZE:
                    print(f"[ADVARSEL] Ugyldig restlengde: {len(chunk)} (forventet {EXPECTED_SIZE}). Ignorerer.")
                    break
                yield chunk
    else:
        raise ValueError(f"Unknown input format: {fmt}")

# ------------------------------- Parsing -------------------------------

def _parse_float_safe(x: str):
    try:
        return float(x)
    except Exception:
        return None

from math import cos, sin, radians


def _parse_rmc(fields):
    """Parse RMC fields; return dict with timestamp_iso, unix_ts, sog_mps, cog_deg."""
    # RMC: $..RMC,hhmmss.ss,A,lat,NS,lon,EW,sog,cog,ddmmyy,...
    if len(fields) < 10:
        return {}
    t_str = fields[1]
    status = fields[2] if len(fields) > 2 else ''
    sog_kn = _parse_float_safe(fields[7]) if len(fields) > 7 else None
    cog_deg = _parse_float_safe(fields[8]) if len(fields) > 8 else None
    d_str = fields[9] if len(fields) > 9 else ''

    # Build UTC timestamp
    ts_iso = None
    unix_ts = None
    try:
        # time
        hh = int(t_str[0:2]); mm = int(t_str[2:4]); ss = int(t_str[4:6])
        frac = t_str[6:]
        micros = 0
        if frac.startswith('.'):
            micros = int(float('0' + frac) * 1e6)
        # date ddmmyy
        dd = int(d_str[0:2]); MM = int(d_str[2:4]); yy = int(d_str[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy
        dt_obj = dt.datetime(year, MM, dd, hh, mm, ss, micros, tzinfo=dt.timezone.utc)
        ts_iso = dt_obj.isoformat().replace('+00:00', 'Z')
        unix_ts = dt_obj.timestamp()
    except Exception:
        pass

    # Convert speed to m/s
    sog_mps = None
    if sog_kn is not None:
        sog_mps = sog_kn * 0.514444

    return {
        'timestamp_iso': ts_iso,
        'unix_ts': unix_ts,
        'sog_mps': sog_mps,
        'cog_deg': cog_deg
    }
    t_str = fields[1]
    status = fields[2] if len(fields) > 2 else ''
    sog_kn = _parse_float_safe(fields[7]) if len(fields) > 7 else None
    cog_deg = _parse_float_safe(fields[8]) if len(fields) > 8 else None
    d_str = fields[9] if len(fields) > 9 else ''

    # Build UTC timestamp
    ts_iso = None
    try:
        # time
        hh = int(t_str[0:2]); mm = int(t_str[2:4]); ss = int(t_str[4:6])
        frac = t_str[6:]
        micros = 0
        if frac.startswith('.'):
            micros = int(float('0' + frac) * 1e6)
        # date ddmmyy
        dd = int(d_str[0:2]); MM = int(d_str[2:4]); yy = int(d_str[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy
        dt_obj = dt.datetime(year, MM, dd, hh, mm, ss, micros, tzinfo=dt.timezone.utc)
        ts_iso = dt_obj.isoformat().replace('+00:00', 'Z')
    except Exception:
        pass

    # Convert speed to m/s
    sog_mps = None
    if sog_kn is not None:
        sog_mps = sog_kn * 0.514444

    return {
        'timestamp_iso': ts_iso,
        'sog_mps': sog_mps,
        'cog_deg': cog_deg
    }


def _parse_psxn23(fields):
    # $PSXN,23,roll,pitch,heading,heave
    out = {}
    try:
        out['roll'] = _parse_float_safe(fields[2])
        out['pitch'] = _parse_float_safe(fields[3])
        out['heading'] = _parse_float_safe(fields[4])
        out['heave'] = _parse_float_safe(fields[5])
    except Exception:
        pass
    return out


def _parse_psxn24(fields):
    # $PSXN,24,roll_rate,pitch_rate,yaw_rate,vertical_velocity
    out = {}
    try:
        out['roll_rate'] = _parse_float_safe(fields[2])
        out['pitch_rate'] = _parse_float_safe(fields[3])
        out['yaw_rate'] = _parse_float_safe(fields[4])
        out['vertical_velocity'] = _parse_float_safe(fields[5])
    except Exception:
        pass
    return out


def _derive_velocities_from_rmc(sog_mps, cog_deg):
    if sog_mps is None or cog_deg is None:
        return None, None
    rad = radians(cog_deg)
    vn = sog_mps * cos(rad)
    ve = sog_mps * sin(rad)
    return vn, ve


def parse_seapath(packet: bytes):
    """
    Try KM Binary first (132 bytes). If it doesn't look like that, treat as
    hex-encoded ASCII with NMEA sentences and parse PSXN/RMC.
    """
    # NMEA path: if it starts with '$' (0x24) somewhere near the start
    if packet and (packet[0] == 0x24 or b'$' in packet[:4]):
        text = packet.decode('ascii', errors='ignore')
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith('$')]
        # aggregate results within this payload
        agg = {}
        rmc_info = {}
        for ln in lines:
            body = ln.split('*', 1)[0]  # strip checksum if present
            fields = body.split(',')
            if len(fields) < 2:
                continue
            tag = fields[0][1:]  # drop '$'
            # RMC from any talker: ..RMC
            if tag.endswith('RMC'):
                rmc_info = _parse_rmc(fields)
            elif tag == 'PSXN':
                if len(fields) >= 2 and fields[1] == '23':
                    agg.update(_parse_psxn23(fields))
                elif len(fields) >= 2 and fields[1] == '24':
                    agg.update(_parse_psxn24(fields))

        # derive velocities from RMC
        vn, ve = _derive_velocities_from_rmc(rmc_info.get('sog_mps'), rmc_info.get('cog_deg'))
        if vn is not None and ve is not None:
            agg['velocity_north'] = vn
            agg['velocity_east'] = ve
        # down velocity from PSXN,24 vertical velocity (down-positive)
        if 'vertical_velocity' in agg and agg['vertical_velocity'] is not None:
            agg['velocity_down'] = agg['vertical_velocity']

        # timestamp
        if rmc_info.get('timestamp_iso'):
            agg['timestamp'] = rmc_info['timestamp_iso']
            if rmc_info.get('unix_ts') is not None:
                agg['unix_ts'] = rmc_info['unix_ts']
        else:
            agg['timestamp'] = 'Invalid'

        return agg if len(agg) > 1 else None

    # KM Binary fallback (first 132 bytes)
    if len(packet) >= EXPECTED_SIZE:
        try:
            unpacked = SEAPATH_STRUCT.unpack(packet[:EXPECTED_SIZE])
        except struct.error:
            return None
        d = dict(zip(FIELDS, unpacked))
        try:
            d["magic"] = d["magic"].decode(errors="ignore")
        except Exception:
            pass
        try:
            base = dt.datetime.utcfromtimestamp(int(d["utc_sec"]))
            ts = base + dt.timedelta(microseconds=int(d["utc_nsec"]) // 1000)
            d["timestamp"] = ts.isoformat()
        except Exception:
            d["timestamp"] = "Invalid"
        return d

    return None

def main():
    parser = argparse.ArgumentParser(
        description=(
            "SeaPath offline parser (fra fil) – skriver valgt CSV i fast rekkefølge, "
            "og kan vise rullende plott med --display."
        )
    )
    parser = argparse.ArgumentParser(
        description=(
            "SeaPath offline parser (fra fil) – skriver valgt CSV i fast rekkefølge, "
            "og kan vise rullende plott med --display."
        )
    )
    parser.add_argument("--input", default="TestData/seapath.csv",
                        help="Path to input file (csv/hex/bin). Default: TestData/seapath.csv")
    parser.add_argument("--input-format", choices=["auto", "csv", "hex", "bin"], default="auto",
                        help="How to interpret the input file (default auto-detect).")
    parser.add_argument("--out-csv", default="MarineVesselSimulator/data/nn_dataset_v9_real/parsedSeaPath.csv",
                        help=("Destination CSV for selected SeaPath fields. Parent dirs will be created.\n"
                              "Default: MarineVesselSimulator/data/nn_dataset_v9_real/parsedSeaPath.csv"))

    # Display toggle & params
    parser.add_argument("--display", action="store_true", help="Show rolling visualization while parsing.")
    parser.add_argument("--interval", type=float, default=0.02, help="Pause after update (s) when displaying.")
    parser.add_argument("--plot-window", type=float, default=8.0, help="Rolling window width in seconds.")
    parser.add_argument("--stride", type=int, default=5, help="Plot only every Nth packet.")
    parser.add_argument("--stride-offset", type=int, default=0, help="Plot-start offset (0-based).")
    parser.add_argument("--plot-limit", type=int, default=0, help="Stop plotting after N samples (0=unlimited).")

    args = parser.parse_args()

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
        out_csv.writerow(SELECTED_ORDER)
        print(f"[INFO] Writing selected SeaPath fields to: {out_name}")
    except Exception as e:
        print(f"[FEIL] Could not open output file '{out_name}' for writing: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional plotting state
    plotting = bool(args.display and plt is not None)
    if args.display and plt is None:
        print("[ADVARSEL] Matplotlib not available; --display ignored.")
        plotting = False

    if plotting:
        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        ax_att, ax_rates, ax_motion = axes
        fig.suptitle(f"SeaPath – rolling {args.plot_window:.1f}s window")
        # attitude
        (l_roll,) = ax_att.plot([], [], label="roll")
        (l_pitch,) = ax_att.plot([], [], label="pitch")
        (l_head,) = ax_att.plot([], [], label="heading")
        ax_att.set_ylabel("Attitude"); ax_att.legend(loc="upper right"); ax_att.grid(True)
        # rates
        (l_rr,) = ax_rates.plot([], [], label="roll_rate")
        (l_pr,) = ax_rates.plot([], [], label="pitch_rate")
        (l_yr,) = ax_rates.plot([], [], label="yaw_rate")
        ax_rates.set_ylabel("Rates"); ax_rates.legend(loc="upper right"); ax_rates.grid(True)
        # motion
        (l_heave,) = ax_motion.plot([], [], label="heave")
        (l_vn,) = ax_motion.plot([], [], label="velocity_north")
        (l_ve,) = ax_motion.plot([], [], label="velocity_east")
        (l_vd,) = ax_motion.plot([], [], label="velocity_down")
        ax_motion.set_ylabel("Heave & Vel"); ax_motion.set_xlabel("UNIX time (s)")
        ax_motion.legend(loc="upper right"); ax_motion.grid(True)

        # histories
        ts: List[float] = []
        roll: List[float] = []; pitch: List[float] = []; head: List[float] = []
        rr: List[float] = []; pr: List[float] = []; yr: List[float] = []
        heave: List[float] = []; vn: List[float] = []; ve: List[float] = []; vd: List[float] = []
        plotted = 0

    total = 0
    written = 0

    try:
        for packet in file_packets_as_bytes(args.input, args.input_format):
            total += 1
            d = parse_seapath(packet)
            if not d:
                continue

            # Write selected fields in exact order
            row = [
                d.get("timestamp", "Invalid"),
                d.get("roll", ''), d.get("pitch", ''), d.get("heading", ''), d.get("heave", ''),
                d.get("roll_rate", ''), d.get("pitch_rate", ''), d.get("yaw_rate", ''),
                d.get("velocity_north", ''), d.get("velocity_east", ''), d.get("velocity_down", ''),
            ]
            out_csv.writerow(row)
            written += 1

            if plotting:
                stride = max(1, int(args.stride))
                offset = max(0, int(args.stride_offset)) % stride
                if ((total - 1) % stride) != offset:
                    continue
                if args.plot_limit and plotted >= args.plot_limit:
                    continue

                # time axis: prefer RMC unix_ts; else KM Binary utc_sec/nsec; else parse ISO; else fallback
                if d.get('unix_ts') is not None:
                    t = float(d['unix_ts'])
                elif ('utc_sec' in d) or ('utc_nsec' in d):
                    try:
                        t = float(d.get('utc_sec', 0)) + float(d.get('utc_nsec', 0)) / 1e9
                    except Exception:
                        t = float(total)
                else:
                    try:
                        ts_iso = d.get('timestamp')
                        t = dt.datetime.fromisoformat(ts_iso.replace('Z', '+00:00')).timestamp() if ts_iso else float(total)
                    except Exception:
                        t = float(total)

                ts.append(t)
                roll.append(d.get("roll", 0.0)); pitch.append(d.get("pitch", 0.0)); head.append(d.get("heading", 0.0))
                rr.append(d.get("roll_rate", 0.0)); pr.append(d.get("pitch_rate", 0.0)); yr.append(d.get("yaw_rate", 0.0))
                heave.append(d.get("heave", 0.0)); vn.append(d.get("velocity_north", 0.0)); ve.append(d.get("velocity_east", 0.0)); vd.append(d.get("velocity_down", 0.0))

                # rolling window slice
                tmin = ts[-1] - float(args.plot_window)
                start = 0
                for i, tt in enumerate(ts):
                    if tt >= tmin:
                        start = i; break

                tsw = ts[start:]
                l_roll.set_data(tsw, roll[start:]); l_pitch.set_data(tsw, pitch[start:]); l_head.set_data(tsw, head[start:])
                l_rr.set_data(tsw, rr[start:]); l_pr.set_data(tsw, pr[start:]); l_yr.set_data(tsw, yr[start:])
                l_heave.set_data(tsw, heave[start:]); l_vn.set_data(tsw, vn[start:]); l_ve.set_data(tsw, ve[start:]); l_vd.set_data(tsw, vd[start:])

                # axes update
                if plt is not None:
                    for ax in (ax_att, ax_rates, ax_motion):
                        ax.set_xlim(tmin, ts[-1])
                        ax.relim(); ax.autoscale_view()
                    plt.pause(float(args.interval))
                plotted += 1

    except KeyboardInterrupt:
        print("\nAvbrutt av bruker (Ctrl+C).")
    finally:
        try:
            out_fh.close()
        except Exception:
            pass
        print(f"Ferdig. Skrev {written} rader til {out_name}.")
        if plotting and plt is not None:
            plt.ioff(); plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live parser & rullende plot (siste 5 s) – bruker KUN hver 10. rad fra CSV.

- Leser KUN kolonne 2 (indeks 1) i CSV; kolonne 1 ignoreres helt.
- Tid fra rammen: sek + nanosekunder (32-bit), håndterer overflyt; x-akse i UNIX-sekunder (avledet fra NTP).
- Gyro/acc tolkes som float32 (big-endian).
- Checksum: sum(bytes 2..nest siste) & 0xFF.
- Akselerometer skaleres fast med 20*|g| (samme skala på alle akser).
- Gyro: rad/s -> deg/s, og deretter skaleres også med 20*|g|.
- Plott viser alltid siste 5 sekunder (justerbart).
- CSV nedprøves: behandler kun hver `--stride`-te rad, med startforskyvning `--stride-offset`.

eksempel; python3 data_parsing_viz10.py --csv ../../TestData2/imu_xxxxxx_xxxxxx.csv
"""

import argparse
import csv
import datetime as dt
import struct
import sys
from math import isfinite, pi
from typing import List, Tuple

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

import matplotlib.pyplot as plt

NTP_UNIX_EPOCH_DIFF = 2208988800  # sek fra 1900-01-01 til 1970-01-01
RAD2DEG = 180.0 / pi              # gyro: rad/s -> deg/s


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
    """Sub-sekund er nanosekunder (0..999,999,999). Overflyt håndteres."""
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


# ------------------------------- Parser -------------------------------

def parse_frame(frame: bytes) -> dict:
    """start, len, cmd, (sec, ns), [ev. 0x00000001], 6x float32, checksum."""
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
    # Opsjonell sentinel 0x00000001
    maybe_one = int.from_bytes(frame[idx:idx + 4], "big")
    if maybe_one == 1:
        idx += 4

    raw_ints: List[int] = []
    floats: List[float] = []
    for _ in range(6):
        u = int.from_bytes(frame[idx:idx + 4], "big")
        raw_ints.append(u)
        floats.append(u32_to_float_be(u))
        idx += 4

    extra = frame[idx:-1] if idx != len(frame) - 1 else b""

    gyro_rad = tuple(floats[:3])   # rad/s
    acc_raw  = tuple(floats[3:6])  # rå

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


def read_hex_rows(csv_path: str) -> List[str]:
    """Leser hex-strenger fra CSV **kolonne 2** (indeks 1). Kolonne 1 ignoreres FULLSTENDIG."""
    rows: List[str] = []
    if _HAVE_PANDAS:
        try:
            df = pd.read_csv(csv_path)
            if df.shape[1] >= 2:
                series = df.iloc[:, 1].dropna().astype(str)
                return series.tolist()
        except Exception:
            pass
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            rows.append(row[1])
    return rows


# ------------------------------- Main -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live parser + rullende plot (siste 5 s) for gyro/acc. "
                    "Bruker kun hver N-te rad fra CSV (default 10)."
    )
    parser.add_argument("--csv", required=True, help="Sti til CSV med hex i kolonne 2 (kolonne 1 ignoreres).")
    parser.add_argument("--interval", type=float, default=0.01, help="Pause mellom rader (s) for live-oppdatering.")
    parser.add_argument("--limit", type=int, default=0, help="Stopp etter N behandlede rader (0 = alle).")
    parser.add_argument("--g", type=float, default=9.81, help="g i m/s^2 (skalering = 20*|g| for både gyro og acc).")
    parser.add_argument("--plot-window", type=float, default=5.0, help="Bredde på rullende plottvindu i sekunder.")
    parser.add_argument("--stride", type=int, default=10, help="Bruk kun hver N-te rad (default 10).")
    parser.add_argument("--stride-offset", type=int, default=0, help="Startforskyvning (0-basert). 0→0,10,20...")
    args = parser.parse_args()

    scale = 20.0 * abs(args.g)     # FAST skalering for både ACC og GYRO (etter rad->deg)
    window_s  = float(args.plot_window)
    stride    = max(1, int(args.stride))
    offset    = max(0, int(args.stride_offset)) % stride

    hex_rows_all = read_hex_rows(args.csv)
    if not hex_rows_all:
        print("Fant ingen hex-strenger i kolonne 2 av CSV.", file=sys.stderr)
        sys.exit(1)

    # Nedprøving: ta kun hver N-te rad med gitt offset
    hex_rows = [s for i, s in enumerate(hex_rows_all) if i % stride == offset]
    if not hex_rows:
        print(f"Ingen rader valgt etter nedprøving (stride={stride}, offset={offset}).", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Stride-nedprøving: bruker kun rader der (i % {stride}) == {offset}. "
          f"Valgt {len(hex_rows)} av {len(hex_rows_all)} rader.")
    print(f"[INFO] Fast skalering aktiv: scale = 20*|g| = {scale:.6f}")
    print(f"[INFO] Gyro konverteres rad/s -> deg/s (×{RAD2DEG:.6f}) før skalering.")

    # Live-plott: 2 paneler
    plt.ion()
    fig, (ax_g, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Gyro (deg/s * {scale:.2f}) og Akselerometer (scaled, 20*g={scale:.2f}) – rullende {window_s:.1f}s")

    # Fullhistorikk; plotting bruker rullende utsnitt
    ts_all: List[float] = []
    gdx_all: List[float] = []; gdy_all: List[float] = []; gdz_all: List[float] = []  # gyro i deg/s, så skalert
    axx_all: List[float] = []; ayy_all: List[float] = []; azz_all: List[float] = []  # acc skalert

    # Linjer/akser
    (lgx,) = ax_g.plot([], [], label=f"gyro_x (deg/s * {scale:.2f})")
    (lgy,) = ax_g.plot([], [], label=f"gyro_y (deg/s * {scale:.2f})")
    (lgz,) = ax_g.plot([], [], label=f"gyro_z (deg/s * {scale:.2f})")
    ax_g.set_ylabel("Gyro (deg/s * scale)")
    ax_g.legend(loc="upper right")
    ax_g.grid(True)

    (lax,) = ax_a.plot([], [], label=f"acc_x * {scale:.2f}")
    (lay,) = ax_a.plot([], [], label=f"acc_y * {scale:.2f}")
    (laz,) = ax_a.plot([], [], label=f"acc_z * {scale:.2f}")
    ax_a.set_ylabel("Acc (scaled, m/s²)")
    ax_a.set_xlabel("Tid (s, UNIX; fra NTP)")
    ax_a.legend(loc="upper right")
    ax_a.grid(True)

    count = 0
    for raw in hex_rows:
        if args.limit and count >= args.limit:
            break
        if not raw or not raw.strip():
            continue

        try:
            frame = hex_to_bytes(raw)
            parsed = parse_frame(frame)
        except Exception as e:
            print(f"[{count}] Feil ved parsing: {e}", file=sys.stderr)
            continue

        # Tid (UNIX s) avledet fra NTP-feltene
        t = parsed["unix_ts"]
        ts_all.append(t)

        # Gyro: rad/s -> deg/s -> skaler med 20*g
        gx_rad, gy_rad, gz_rad = parsed["gyro_rad"]
        gx = gx_rad * RAD2DEG * scale
        gy = gy_rad * RAD2DEG * scale
        gz = gz_rad * RAD2DEG * scale
        gdx_all.append(gx); gdy_all.append(gy); gdz_all.append(gz)

        # Acc: skaler med 20*g
        ax_s = parsed["acc_raw"][0] * scale
        ay_s = parsed["acc_raw"][1] * scale
        az_s = parsed["acc_raw"][2] * scale
        axx_all.append(ax_s); ayy_all.append(ay_s); azz_all.append(az_s)

        # Rullende vindu: siste window_s sekunder
        t_min = t - window_s
        start_idx = 0
        for i, tt in enumerate(ts_all):
            if tt >= t_min:
                start_idx = i
                break

        tsw  = ts_all[start_idx:]
        gdxw = gdx_all[start_idx:]; gdyw = gdy_all[start_idx:]; gdzw = gdz_all[start_idx:]
        axxw = axx_all[start_idx:]; ayyw = ayy_all[start_idx:]; azzw = azz_all[start_idx:]

        # Oppdater plott
        lgx.set_data(tsw, gdxw); lgy.set_data(tsw, gdyw); lgz.set_data(tsw, gdzw)
        lax.set_data(tsw, axxw); lay.set_data(tsw, ayyw); laz.set_data(tsw, azzw)

        ax_g.set_xlim(t_min, t)
        ax_a.set_xlim(t_min, t)
        for ax in (ax_g, ax_a):
            ax.relim()
            ax.autoscale_view()

        # Utskrift
        def fmt3(tup): return tuple(f"{v:+.6f}" if isfinite(v) else str(v) for v in tup)
        print(
            f"[{count}] time={parsed['iso']} (ms={parsed['sub_ms']}, us={parsed['sub_us']}) "
            f"gyro_scaled(deg/s*{scale:.2f})={fmt3((gx, gy, gz))} "
            f"acc_scaled={(f'{ax_s:+.6f}', f'{ay_s:+.6f}', f'{az_s:+.6f}')}  "
            f"checksum={'OK' if parsed['checksum_ok'] else 'FAIL'}"
        )

        plt.pause(args.interval)
        count += 1

    print("Ferdig. Lukk plottvinduet for å avslutte.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

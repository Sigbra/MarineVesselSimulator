# scripts/nn_observer_v1.py
# -------------------------------------------------------
# GRU observer with probabilistic head (mu + logvar).
# - No input/output normalization.
# - Trains on sliding windows with Gaussian NLL.
# - Exports a *stateless* TorchScript module:
#       (mu, logvar, h_next) = model_stateless(x, h_prev)
#   where:
#       x:      [B,6]  IMU sample(s)
#       h_prev: [L,B,H] hidden state from previous tick (or None)
#       h_next: [L,B,H] next hidden state (feed into next call)
#
# Train (from repo root):
#
#  python3 scripts/nn_observer_v1.py --train data/nn_dataset/train.csv --val data/nn_dataset/val.csv --out data/nn_model_v1 --seq 50 --epochs 200 --gpu
#
# Now with live plotting of per-epoch train/val NLL. Use --no_plot to disable.
#
# IMPORTANT CHANGE:
# - The network architecture is *fixed* inside VelRateGRUObserver.
# -------------------------------------------------------

import argparse
from pathlib import Path
import json
from typing import Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUT_COLS  = ["ax","ay","az","wx","wy","wz"]
TARGET_COLS = ["u","v","w","p","q","r"]


# ---------------- Model (GRU + probabilistic head; FIXED ARCH) ---------------- #

class VelRateGRUObserver(nn.Module):
    """
    Core model (used for training with sequences).

    Fixed architecture (cannot be changed via CLI or runtime input):
        IN_DIM     = 6          # ax, ay, az, wx, wy, wz
        HIDDEN     = 128
        NUM_LAYERS = 2
        HEAD: Linear(HIDDEN -> HIDDEN) + ReLU + Linear(HIDDEN -> 12)  # [mu(6), logvar(6)]

    For export we wrap it into a stateless TorchScript module.
    """
    IN_DIM: int = 6
    HIDDEN: int = 128
    NUM_LAYERS: int = 2

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(self.IN_DIM, self.HIDDEN, num_layers=self.NUM_LAYERS, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, 12)  # [mu(6), logvar(6)]
        )

    def _split_head(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.head(z)
        mu, logvar = y[..., :6], y[..., 6:]
        logvar = torch.clamp(logvar, -10.0, 5.0)
        return mu, logvar

    def forward_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training path: x [B,T,6] -> (mu, logvar) each [B,T,6]
        """
        if x.dim() != 3 or x.size(-1) != self.IN_DIM:
            raise ValueError(f"forward_seq expects [B,T,{self.IN_DIM}]")
        out, _ = self.gru(x)                 # [B,T,H]
        mu, logvar = self._split_head(out)   # [B,T,6]
        return mu, logvar

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(IN_DIM={self.IN_DIM}, "
                f"HIDDEN={self.HIDDEN}, NUM_LAYERS={self.NUM_LAYERS}, HEAD='[Linear-ReLU-Linear]->(mu,logvar)')")


# ---------------- Loss (Gaussian NLL) ---------------- #

def gaussian_nll_diag(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # All shapes broadcastable (...,6)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (y - mu)**2 * inv_var)
    return nll.sum(dim=-1).mean()  # mean over batch*time


# ---------------- Live Plot ---------------- #

class LivePlot:
    """Lightweight live plot for per-epoch train/val NLL."""
    def __init__(self, out_dir: Path, title: str = "Training / Validation NLL"):
        self.enabled = True
        self.out_dir = Path(out_dir)
        self.epochs, self.tr_vals, self.va_vals = [], [], []
        try:
            import matplotlib.pyplot as plt  # lazy import
            self.plt = plt
            self.fig, self.ax = plt.subplots()
            (self.l_tr,) = self.ax.plot([], [], label="train_nll")
            (self.l_va,) = self.ax.plot([], [], label="val_nll")
            self.ax.set_xlabel("epoch")
            self.ax.set_ylabel("NLL")
            self.ax.set_title(title)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            plt.ion()
            plt.show()
        except Exception as e:
            print(f"[plot] disabled ({e})")
            self.enabled = False

    def update(self, ep: int, tr: float, va: Optional[float]):
        if not self.enabled:
            return
        self.epochs.append(ep)
        self.tr_vals.append(tr)
        self.va_vals.append(va if va is not None else math.nan)
        self.l_tr.set_data(self.epochs, self.tr_vals)
        self.l_va.set_data(self.epochs, self.va_vals)
        # dynamic axes
        x_max = max(10, ep)
        self.ax.set_xlim(1, x_max)
        ys = [y for y in (self.tr_vals + self.va_vals) if math.isfinite(y)]
        if ys:
            ymin, ymax = min(ys), max(ys)
            if ymin == ymax:
                ymax = ymin + 1e-6
            pad = 0.05 * (ymax - ymin)
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.plt.tight_layout()
        self.plt.pause(0.001)

    def save_png(self, name: str = "loss_curve.png"):
        if not self.enabled:
            return
        try:
            path = self.out_dir / name
            self.plt.savefig(path)
            print(f"[plot] saved: {path}")
        except Exception as e:
            print(f"[plot] save failed: {e}")


# ---------------- Dataset (sliding windows, no normalization) ---------------- #

class WindowDataset(torch.utils.data.Dataset):
    """
    Builds sliding windows (seq_len) from a CSV with columns:
      run_id, t, ax, ay, az, wx, wy, wz, u, v, w, p, q, r
    Returns (Xwin, Ywin) with shapes [T,6], [T,6] (dense supervision).
    """
    def __init__(self, csv_path: str, seq_len: int = 25):
        df = pd.read_csv(csv_path)
        # Verify columns
        for c in ["run_id","t"] + INPUT_COLS + TARGET_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        # drop NaNs/infs just in case
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=INPUT_COLS + TARGET_COLS).reset_index(drop=True)

        self.seq_len = seq_len
        self.X = df[INPUT_COLS].to_numpy(np.float32)
        self.Y = df[TARGET_COLS].to_numpy(np.float32)
        self.runs = df["run_id"].to_numpy()

        # build valid end indices per run (no cross-run windows)
        self.idx = []
        start = 0
        N = len(self.runs)
        while start < N:
            r = self.runs[start]
            end = start
            while end < N and self.runs[end] == r:
                end += 1
            if end - start >= self.seq_len:
                for j in range(start + self.seq_len - 1, end):
                    self.idx.append((start, j))
            start = end

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        run_start, end_idx = self.idx[i]
        start_idx = end_idx - self.seq_len + 1
        Xwin = self.X[start_idx:end_idx+1, :]   # [T,6]
        Ywin = self.Y[start_idx:end_idx+1, :]   # [T,6]
        return torch.from_numpy(Xwin), torch.from_numpy(Ywin)


# ---------------- Train / Validate ---------------- #

def train_one_epoch(model, opt, loader, device):
    model.train()
    total_nll, total_n = 0.0, 0
    for xb, yb in loader:                      # xb,yb: [B,T,6]
        xb, yb = xb.to(device), yb.to(device)
        mu, logvar = model.forward_seq(xb)     # [B,T,6] each
        base_nll = gaussian_nll_diag(mu, logvar, yb)
        loss = base_nll + 1e-5 * (logvar**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # track *NLL* (not the regularized loss) for comparability with val
        bs = xb.size(0)
        total_nll += base_nll.item() * bs
        total_n   += bs
    return (total_nll / total_n) if total_n else float("nan")

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_nll, total_n = 0.0, 0
    sum_abs_err = torch.zeros(6, device=device)
    sum_sq_err  = torch.zeros(6, device=device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        mu, logvar = model.forward_seq(xb)
        nll = gaussian_nll_diag(mu, logvar, yb)
        total_nll += nll.item() * xb.size(0)
        total_n   += xb.size(0)
        err = (mu - yb).reshape(-1, 6)
        sum_abs_err += err.abs().sum(dim=0)
        sum_sq_err  += (err**2).sum(dim=0)
    if total_n == 0:
        return float("nan"), [float("nan")]*6, [float("nan")]*6
    mae  = (sum_abs_err / (total_n * loader.dataset.seq_len)).tolist()
    rmse = torch.sqrt(sum_sq_err / (total_n * loader.dataset.seq_len)).tolist()
    return total_nll / total_n, mae, rmse


# ---------------- Export: *stateless* TorchScript ---------------- #

class _ExportStateless(nn.Module):
    """
    Stateless wrapper guaranteed to script:
      forward(x, h_prev) -> (mu, logvar, h_next)
        x: [B,6]
        h_prev: Optional[Tensor] [L,B,H] or None
        h_next: [L,B,H]
    C++ keeps and feeds h_next back at each tick.
    """
    def __init__(self, core: VelRateGRUObserver):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]):
        # TorchScript-safe shape checks (no f-strings / tuple conversions)
        if not (x.dim() == 2 and x.size(-1) == self.core.IN_DIM):
            raise RuntimeError(f"expected x of shape [B,{self.core.IN_DIM}]")
        B = x.size(0)
        L = int(self.core.gru.num_layers)
        H = int(self.core.gru.hidden_size)

        # Initialize or fix hidden state (TorchScript-safe branching)
        if h_prev is None:
            h_prev = torch.zeros(L, B, H, device=x.device)
        else:
            cond_bad_shape = (h_prev.dim() != 3) or (h_prev.size(0) != L) or (h_prev.size(1) != B) or (h_prev.size(2) != H)
            cond_bad_dev   = (h_prev.device != x.device)
            if cond_bad_shape or cond_bad_dev:
                h_prev = torch.zeros(L, B, H, device=x.device)

        out, h_next = self.core.gru(x.unsqueeze(1), h_prev)  # [B,1,H], [L,B,H]
        y = self.core.head(out[:, -1, :])                    # [B,12]
        mu, logvar = y[..., :6], y[..., 6:]
        logvar = torch.clamp(logvar, -10.0, 5.0)
        return mu, logvar, h_next


def export_torchscript_stateless(model, out_path):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    wrapped = _ExportStateless(model_cpu)
    scripted = torch.jit.script(wrapped)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    scripted.save(str(tmp))
    tmp.replace(out_path)  # atomic rename
    print(f"Saved TorchScript (script, stateless): {out_path}")



# ---------------- CLI / Main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (e.g., data/nn_dataset_X_C0/train.csv)")
    ap.add_argument("--val", default=None,   help="Path to val.csv (e.g., data/nn_dataset_X_C0/val.csv)")
    ap.add_argument("--out", default="data/nn_model_v1", help="Output folder (relative to repo root or absolute)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=30, help="GRU unroll/window length")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no_plot", action="store_true", help="Disable live plotting (useful on headless servers)")
    args = ap.parse_args()

    # Resolve paths relative to repo root (parent of scripts/)
    script_dir = Path(__file__).resolve().parent        # <repo>/scripts
    repo_root  = script_dir.parent                      # <repo>
    def resolve(p):
        if p is None: return None
        p = Path(p)
        return p if p.is_absolute() else (repo_root / p).resolve()

    train_path = resolve(args.train)
    val_path   = resolve(args.val) if args.val else None
    out_dir    = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] repo_root={repo_root}")
    print(f"[paths] train    ={train_path}")
    if val_path: print(f"[paths] val      ={val_path}")
    print(f"[paths] out      ={out_dir}")

    # Quick sanity: columns exist
    req_cols = ["run_id","t"] + INPUT_COLS + TARGET_COLS
    df_chk = pd.read_csv(train_path, nrows=1)
    missing = [c for c in req_cols if c not in df_chk.columns]
    if missing:
        raise SystemExit(f"Missing columns in train.csv: {missing}")

    # Datasets/loaders
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    ds_train = WindowDataset(str(train_path), seq_len=args.seq)
    n_train = len(ds_train)
    if n_train == 0:
        raise SystemExit("No training windows created. Reduce --seq or ensure train runs have ≥ seq rows.")
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        pin_memory=(device == "cuda"), num_workers=2
    )

    dl_val = None
    if val_path:
        ds_val = WindowDataset(str(val_path), seq_len=args.seq)
        n_val = len(ds_val)
        if n_val == 0:
            raise SystemExit("No validation windows created. Reduce --seq or ensure val runs have ≥ seq rows.")
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=max(1024, args.batch), shuffle=False,
            pin_memory=(device == "cuda"), num_workers=2
        )

    print(f"Train windows: {n_train} (seq={args.seq})")
    if dl_val: print(f"Val windows:   {len(ds_val)} (seq={args.seq})")

    # Model / Optimizer (architecture is FIXED; not configurable by CLI)
    model = VelRateGRUObserver().to(device)
    print(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train
    best_val = float("inf")
    best_path = out_dir / "best_weights.pt"
    plot = None if args.no_plot else LivePlot(out_dir)
    for ep in range(1, args.epochs + 1):
        tr_nll = train_one_epoch(model, opt, dl_train, device)
        if dl_val is not None:
            val_nll, mae, rmse = evaluate(model, dl_val, device)
            msg = (f"[{ep:03d}] train_nll={tr_nll:.6f} | val_nll={val_nll:.6f} | "
                   f"MAE[{', '.join(TARGET_COLS)}]=[{', '.join(f'{v:.4f}' for v in mae)}] | "
                   f"RMSE=[{', '.join(f'{v:.4f}' for v in rmse)}]")
            print(msg)
            if plot: plot.update(ep, tr_nll, val_nll)
            if np.isfinite(val_nll) and val_nll < best_val:
                best_val = val_nll
                torch.save(model.state_dict(), best_path)
        else:
            print(f"[{ep:03d}] train_nll={tr_nll:.6f} | epoch done")
            if plot: plot.update(ep, tr_nll, None)

    # Save final weights and export TorchScript (stateless)
    weights_path = out_dir / "weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights: {weights_path}")
    if best_val < float("inf"):
        print(f"Best val_nll: {best_val:.6f} | checkpoint: {best_path}")

    export_torchscript_stateless(model, out_dir / "model_stateless.pt")

    # Save a tiny config for reproducibility (reflect fixed architecture)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "arch": {
                "in_dim": VelRateGRUObserver.IN_DIM,
                "hidden": VelRateGRUObserver.HIDDEN,
                "layers": VelRateGRUObserver.NUM_LAYERS,
                "head": ["Linear", "ReLU", "Linear->(mu,logvar)"]
            },
            "seq": args.seq,
            "lr": args.lr,
            "epochs": args.epochs
        }, f, indent=2)

    # Save final plot image
    if plot:
        plot.save_png()

    print("Done.")

if __name__ == "__main__":
    main()

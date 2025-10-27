# scripts/nn_observer_v5.py
# -------------------------------------------------------
# GRU observer (v5) with ENSEMBLE predictive covariance for EKF.
#
# Same as v4 in architecture, training flow, plots, and exporters — but:
# - Targets now include two heading outputs (cpsi_t, spsi_t) so D_out = 9
# - Physics-informed loss uses the predicted (cpsi, spsi) (unit-circle reprojection)
# - Dataset columns expected:
#     [run_id, t, ax, ay, az, wx, wy, wz, cpsi, spsi, Vn, Ve, p, q, r, phi, theta, cpsi_t, spsi_t]
#
# Inputs (8D):  [ax, ay, az, wx, wy, wz, cpsi, spsi]  (cpsi/spsi from previous timestep)
# Targets (9D): [Vn, Ve, p, q, r, phi, theta, cpsi_t, spsi_t]
#
# Normalization (train-split stats):
#   - TRAIN/VAL use normalized inputs/targets (x̂ = (x-μ_in)/σ_in, ŷ = (y-μ_out)/σ_out).
#   - TorchScript exports accept RAW inputs and return PHYSICAL outputs (μ, logvar).
#     Internally: normalize inputs, run core, then denormalize outputs and scale variances.
#
# Training:
#   - Warmup with MSE + λ_phys * physics model_loss (uses predicted angles; SSA-safe).
#   - Switch to NLL (diag Gaussian) after --switch_to_nll_ep.
#
# Ensemble:
#   - Train M members (different seeds / optional bagging).
#   - Export per-member stateless TS (PHYSICAL μ, logvar) and an ensemble TS (PHYSICAL μ_mean, FULL cov).
#
# Example:
# python3 scripts/nn_observer_v5.py --train data/nn_dataset_v5/train.csv --val data/nn_dataset_v5/val.csv \
#   --out data/nn_model_v5 --seq 100 --epochs 1000 --gpu --ensemble 3 --bagging_frac 0.9 \
#   --switch_to_nll_ep 300 --norm_json data/nn_dataset_v5/norm_stats.json --lambda_phys 1e-3
# -------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Optional, Tuple, List, Dict

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- I/O columns ---------------- #
INPUT_COLS   = ["ax","ay","az","wx","wy","wz","cpsi","spsi"]  # 8
TARGET_COLS  = ["Vn","Ve","p","q","r","phi","theta","cpsi_t","spsi_t"]  # 9


# ---------------- Normalization helpers (train-split only) ---------------- #

class NormStats:
    def __init__(self,
                 in_mu: np.ndarray, in_std: np.ndarray,
                 out_mu: np.ndarray, out_std: np.ndarray):
        self.in_mu  = in_mu.astype(np.float32)
        self.in_std = np.maximum(in_std.astype(np.float32), 1e-8)
        self.out_mu  = out_mu.astype(np.float32)
        self.out_std = np.maximum(out_std.astype(np.float32), 1e-8)

    @staticmethod
    def from_json(path: Path) -> "NormStats":
        d = json.loads(Path(path).read_text())
        in_mu  = np.array([d["inputs"]["mean"][k] for k in INPUT_COLS], dtype=np.float32)
        in_std = np.array([d["inputs"]["std"][k]  for k in INPUT_COLS], dtype=np.float32)
        out_mu  = np.array([d["targets"]["mean"][k] for k in TARGET_COLS], dtype=np.float32)
        out_std = np.array([d["targets"]["std"][k]  for k in TARGET_COLS], dtype=np.float32)
        return NormStats(in_mu, in_std, out_mu, out_std)

    @staticmethod
    def from_train_csv(train_csv: Path) -> "NormStats":
        df = pd.read_csv(train_csv)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=INPUT_COLS + TARGET_COLS)
        in_mu  = df[INPUT_COLS].mean().to_numpy(np.float32)
        in_std = df[INPUT_COLS].std(ddof=0).replace(0.0, 1e-8).to_numpy(np.float32)
        out_mu  = df[TARGET_COLS].mean().to_numpy(np.float32)
        out_std = df[TARGET_COLS].std(ddof=0).replace(0.0, 1e-8).to_numpy(np.float32)
        return NormStats(in_mu, in_std, out_mu, out_std)

    def to_json_dict(self) -> Dict:
        return {
            "inputs":  {"mean": {k: float(v) for k, v in zip(INPUT_COLS, self.in_mu)},
                        "std":  {k: float(v) for k, v in zip(INPUT_COLS, self.in_std)}},
            "targets": {"mean": {k: float(v) for k, v in zip(TARGET_COLS, self.out_mu)},
                        "std":  {k: float(v) for k, v in zip(TARGET_COLS, self.out_std)}},
            "note": "computed/used for training normalization (train split only)"
        }


# ---------------- Variational (locked) dropout ---------------- #

class LockedDropout(nn.Module):
    """
    Time-locked (variational) dropout: one mask per sequence shared across time.
    Expects [B,T,H]. If shape differs, falls back to standard dropout.
    Disabled in eval mode (like nn.Dropout).
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.p <= 0.0):
            return x
        if x.dim() != 3:
            # Fallback for non-[B,T,H] shapes (e.g., export path)
            return F.dropout(x, p=self.p, training=True)
        B, T, H = x.shape
        keep = 1.0 - self.p
        # mask ~ Bernoulli(keep) / keep  (inverted dropout)
        mask = (torch.rand(B, 1, H, device=x.device, dtype=x.dtype) < keep) / keep
        return x * mask


# ---------------- Model: FC(8→80) + GRU×3(H=80) + LockedDropout + Heads(80→9,80→9) ---------------- #

class GRUObserverV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim     = 8
        self.embed_dim  = 80
        self.hidden     = 80
        self.num_layers = 4
        self.dropout_p_between = 0.2   # between GRU layers (locked)
        self.dropout_p_after   = 0.4   # after final GRU (locked)

        self.input_fc = nn.Sequential(
            nn.Linear(self.in_dim, self.embed_dim),
            nn.ReLU()
        )

        # Stack 1-layer GRUs so we can drop between them
        self.gru_layers = nn.ModuleList()
        for l in range(self.num_layers):
            in_size = self.embed_dim if l == 0 else self.hidden
            self.gru_layers.append(
                nn.GRU(in_size, self.hidden, num_layers=1, batch_first=True)
            )

        self.lockdrop_between = LockedDropout(self.dropout_p_between)
        self.lockdrop_after   = LockedDropout(self.dropout_p_after)

        self.fc_mu = nn.Linear(self.hidden, len(TARGET_COLS))  # 9
        self.fc_lv = nn.Linear(self.hidden, len(TARGET_COLS))  # 9

    def _heads(self, z: torch.Tensor):
        z = self.lockdrop_after(z)               # locked dropout after all GRUs
        mu = self.fc_mu(z)
        logvar = torch.clamp(self.fc_lv(z), -10.0, 5.0)
        return mu, logvar

    def forward_seq(self, x_hat: torch.Tensor):
        if x_hat.dim() != 3 or x_hat.size(-1) != self.in_dim:
            raise ValueError(f"forward_seq expects [B,T,{self.in_dim}]")

        x = self.input_fc(x_hat)                 # [B,T,80]

        # GRU × num_layers with locked dropout BETWEEN layers
        for l, gru in enumerate(self.gru_layers):
            x, _ = gru(x)                        # [B,T,80]
            if l < self.num_layers - 1:          # between-layer dropout
                x = self.lockdrop_between(x)     # [B,T,80], time-locked mask

        mu_hat, logvar_hat = self._heads(x)      # [B,T,9]
        return mu_hat, logvar_hat


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, embed_dim={self.embed_dim}, hidden={self.hidden}, "
            f"num_layers={self.num_layers}, "
            f"dropout_between={self.dropout_p_between}, dropout_after={self.dropout_p_after}, "
            f"dropout_type='Locked (time-locked)', "
            f"HEADS='mu_hat:{self.hidden}→{len(TARGET_COLS)}, "
            f"logvar_hat:{self.hidden}→{len(TARGET_COLS)} (normalized)')"
        )


# ---------------- Losses (computed in NORMALIZED space) ---------------- #

def mse_loss(mu_hat: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return torch.mean((mu_hat - y_hat) ** 2)

def nll_diag(mu_hat: torch.Tensor, logvar_hat: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar_hat)
    nll = 0.5 * (logvar_hat + (y_hat - mu_hat)**2 * inv_var)
    return nll.sum(dim=-1).mean()


# ---------------- Physics-informed model loss (PHYSICAL units) ---------------- #
# Runs only during MSE warmup (weighted by --lambda_phys).
# Uses ψ computed from predicted (cpsi, spsi) with unit-circle reprojection and SSA.

def _angle_diff(a1: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
    d = a1 - a0
    return torch.atan2(torch.sin(d), torch.cos(d))

def _finite_diff_first(x: torch.Tensor, dt: float, *, is_angle: bool = False) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("_finite_diff_first expects [B,T]")
    d = _angle_diff(x[:, 1:], x[:, :-1]) if is_angle else (x[:, 1:] - x[:, :-1])
    return d / dt

def _reproject_to_unit(c: torch.Tensor, s: torch.Tensor, eps: float = 1e-12):
    r = torch.sqrt(torch.clamp(c*c + s*s, min=eps))
    return c / r, s / r

def model_loss(mu_phys: torch.Tensor, psi_seq_unused: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Physics-informed penalty in physical units (no GT yaw needed).
      mu_phys: [B,T,9] = [Vn, Ve, p, q, r, phi, theta, cpsi, spsi]   (PHYSICAL)
      psi_seq_unused: ignored (kept for API compatibility)
      dt: scalar seconds
    """
    B, T, D = mu_phys.shape
    if T < 2:
        return mu_phys.new_tensor(0.0)
    if D != 9:
        raise ValueError(f"model_loss expects mu_phys[...,9]; got {D}")

    Vn, Ve, p, q, r, phi, th, cpsi_raw, spsi_raw = [mu_phys[..., i] for i in range(9)]  # [B,T]

    # Keep (cpsi,spsi) on unit circle for stable atan2 and meaningful yaw
    cpsi, spsi = _reproject_to_unit(cpsi_raw, spsi_raw)
    psi = torch.atan2(spsi, cpsi)  # [B,T]

    # ZYX kinematics (Rzyx convention). Clamp cos(theta) to avoid singularity.
    cphi = torch.cos(phi); sphi = torch.sin(phi)
    cth  = torch.cos(th).clamp_min(1e-3); sth = torch.sin(th)
    tan_th = sth / cth

    phi_dot_rhs   = p + q * sphi * tan_th + r * cphi * tan_th
    theta_dot_rhs = q * cphi - r * sphi
    psi_dot_rhs   = (q * sphi + r * cphi) / cth

    dphi_dt   = _finite_diff_first(phi, dt, is_angle=True)
    dtheta_dt = _finite_diff_first(th,  dt, is_angle=True)
    dpsi_dt   = _finite_diff_first(psi, dt, is_angle=True)

    r_phi   = dphi_dt   - phi_dot_rhs[:, :-1]
    r_theta = dtheta_dt - theta_dot_rhs[:, :-1]
    r_psi   = dpsi_dt   - psi_dot_rhs[:, :-1]

    # Unit-circle consistency penalty for (cpsi,spsi)
    circle_err = (cpsi_raw*cpsi_raw + spsi_raw*spsi_raw - 1.0)
    L_circle = (circle_err * circle_err).mean()

    # Kinematic residuals (SSA-safe) + circle penalty
    L_kin = (r_phi.pow(2).mean() + r_theta.pow(2).mean() + r_psi.pow(2).mean()) + L_circle
    return L_kin


# ---------------- Live Plot ---------------- #

class LivePlot:
    def __init__(self, out_dir: Path, title: str = "Training / Validation Loss", yscale: str = "linear"):
        self.enabled = True
        self.out_dir = Path(out_dir)
        self.epochs, self.tr_vals, self.va_vals, self.labels = [], [], [], []
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig, self.ax = plt.subplots()
            (self.l_tr,) = self.ax.plot([], [], label="train")
            (self.l_va,) = self.ax.plot([], [], label="val")
            self.ax.set_xlabel("epoch")
            self.ax.set_ylabel("loss")
            self.ax.set_title(title)
            self.ax.set_yscale(yscale)
            self._is_log = (yscale == "log")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            plt.ion(); plt.show()
        except Exception as e:
            print(f"[plot] disabled ({e})")
            self.enabled = False
            return

    def update(self, ep: int, tr: float, va: Optional[float], loss_name: str):
        if not self.enabled:
            return
        self.epochs.append(ep)
        self.tr_vals.append(tr)
        self.va_vals.append(va if va is not None else math.nan)

        self.l_tr.set_data(self.epochs, self.tr_vals)
        self.l_va.set_data(self.epochs, self.va_vals)

        x_max = max(10, ep); self.ax.set_xlim(1, x_max)

        if self._is_log:
            self.ax.relim()
            self.ax.autoscale_view(scalex=False, scaley=True)
        else:
            ys = [y for y in (self.tr_vals + self.va_vals) if math.isfinite(y)]
            if ys:
                ymin, ymax = min(ys), max(ys)
                if ymin == ymax:
                    ymax = ymin + 1e-6
                pad = 0.05 * (ymax - ymin)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        self.ax.set_ylabel(loss_name)
        self.plt.tight_layout()
        self.plt.pause(0.001)

    def save_png(self, name: str = "loss_curve.png"):
        if not self.enabled:
            return
        try:
            path = self.out_dir / name
            self.plt.tight_layout()
            self.fig.savefig(path)
            print(f"[plot] saved: {path}")
        except Exception as e:
            print(f"[plot] save failed: {e}")


class MetricPlot:
    """
    Live plot for N-channel metrics (e.g., MSE per output) over epochs.
    """
    def __init__(self, out_dir: Path, title: str, ylabel: str, labels: List[str], png_name: str, yscale: str = "linear"):
        self.enabled = True
        self.out_dir = Path(out_dir)
        self.png_name = png_name
        self.labels = labels
        self.epochs: List[int] = []
        self.values = [[] for _ in range(len(labels))]
        self._is_log = (yscale == "log")
        self._eps = 1e-12

        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig, self.ax = plt.subplots()
            self.lines = []
            for lab in labels:
                (line,) = self.ax.plot([], [], label=lab)
                self.lines.append(line)

            self.ax.set_xlabel("epoch")
            self.ax.set_ylabel(ylabel)
            self.ax.set_title(title)
            self.ax.set_yscale(yscale)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)

            if self._is_log:
                self.ax.set_ylim(self._eps, 1.0)

            plt.ion()
            plt.show()
        except Exception as e:
            print(f"[plot] {title} disabled ({e})")
            self.enabled = False

    def update(self, ep: int, vals: List[float]):
        if not self.enabled or len(vals) != len(self.labels):
            return

        self.epochs.append(ep)

        for i, v in enumerate(vals):
            vv = float(v)
            if self._is_log:
                if not np.isfinite(vv) or vv <= 0.0:
                    vv = self._eps
                else:
                    vv = max(vv, self._eps)
            self.values[i].append(vv)
            self.lines[i].set_data(self.epochs, self.values[i])

        x_max = max(10, ep)
        self.ax.set_xlim(1, x_max)

        ys = [y for series in self.values for y in series if np.isfinite(y)]
        if ys:
            ymin, ymax = min(ys), max(ys)
            if self._is_log:
                ymin = max(ymin, self._eps)
                if ymax <= ymin:
                    ymax = ymin * 1.000001
                self.ax.set_ylim(ymin / 1.2, ymax * 1.2)
            else:
                if ymin == ymax:
                    ymax = ymin + 1e-6
                pad = 0.05 * (ymax - ymin)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def save_png(self):
        if not self.enabled:
            return
        try:
            path = self.out_dir / self.png_name
            self.plt.tight_layout()
            self.fig.savefig(path)
            print(f"[plot] saved: {path}")
        except Exception as e:
            print(f"[plot] save failed ({self.png_name}): {e}")


# ---------------- Dataset (normalizes x and y using train stats) ---------------- #

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, seq_len: int, norm: NormStats):
        df = pd.read_csv(csv_path)
        for c in ["run_id","t"] + INPUT_COLS + TARGET_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=INPUT_COLS + TARGET_COLS).reset_index(drop=True)

        X = df[INPUT_COLS].to_numpy(np.float32)
        Y = df[TARGET_COLS].to_numpy(np.float32)

        self.X_hat = (X - norm.in_mu) / norm.in_std
        self.Y_hat = (Y - norm.out_mu) / norm.out_std

        self.seq_len = seq_len
        self.runs = df["run_id"].to_numpy()

        self.idx: List[Tuple[int,int]] = []
        start = 0; N = len(self.runs)
        while start < N:
            r = self.runs[start]; end = start
            while end < N and self.runs[end] == r:
                end += 1
            if end - start >= self.seq_len:
                for j in range(start + self.seq_len - 1, end):
                    self.idx.append((start, j))
            start = end

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        _, end_idx = self.idx[i]
        start_idx = end_idx - self.seq_len + 1
        Xwin = self.X_hat[start_idx:end_idx+1, :]   # [T,8] normalized
        Ywin = self.Y_hat[start_idx:end_idx+1, :]   # [T,9] normalized
        return torch.from_numpy(Xwin), torch.from_numpy(Ywin)


# ---------------- Train / Validate (normalized space; metrics in physical) ---------------- #

def train_one_epoch(model, opt, loader, device, use_nll: bool,
                    out_mu: torch.Tensor, out_std: torch.Tensor,
                    lambda_phys: float, phys_dt: float):
    model.train()
    total_loss, total_n = 0.0, 0
    total_mse, total_nll = 0.0, 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        mu_hat, logvar_hat = model.forward_seq(xb)
        loss_mse = mse_loss(mu_hat, yb)
        loss_nll = nll_diag(mu_hat, logvar_hat, yb)

        if (not use_nll) and (lambda_phys > 0.0) and (phys_dt > 0.0):
            # Denormalize to PHYSICAL; mu_hat is [B,T,9]
            mu_phys = mu_hat * out_std.view(1,1,-1) + out_mu.view(1,1,-1)
            # model_loss expects [B,T,9] = [Vn,Ve,p,q,r,phi,theta,cpsi,spsi]
            L_phys = model_loss(mu_phys, mu_phys.new_zeros(mu_phys.size(0), mu_phys.size(1)), phys_dt)
            loss = loss_mse + float(lambda_phys) * L_phys
        else:
            loss = loss_nll if use_nll else loss_mse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_mse  += float(loss_mse.item()) * bs
        total_nll  += float(loss_nll.item()) * bs
        total_n    += bs
    if total_n == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / total_n, total_mse / total_n, total_nll / total_n


@torch.inference_mode()
def evaluate(model, loader, device, out_mu: torch.Tensor, out_std: torch.Tensor):
    model.eval()
    total_mse, total_nll, total_n = 0.0, 0.0, 0
    D = len(TARGET_COLS)
    sum_sq_err  = torch.zeros(D, device=device, dtype=torch.float64)

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        mu_hat, logvar_hat = model.forward_seq(xb)      # normalized
        loss_mse = mse_loss(mu_hat, yb)
        loss_nll = nll_diag(mu_hat, logvar_hat, yb)
        total_mse += float(loss_mse.item()) * xb.size(0)
        total_nll += float(loss_nll.item()) * xb.size(0)
        total_n   += xb.size(0)

        # Denormalize to physical space for per-channel MSE
        mu_phys = (mu_hat * out_std + out_mu).reshape(-1, D).to(torch.float64)
        y_phys  = (yb      * out_std + out_mu).reshape(-1, D).to(torch.float64)
        err = (mu_phys - y_phys)
        sum_sq_err  += (err**2).sum(dim=0)

    if total_n == 0:
        return float("nan"), float("nan"), [float("nan")]*D
    denom = float(total_n * loader.dataset.seq_len)
    mse_ch = (sum_sq_err / denom).tolist()  # per-channel MSE in PHYSICAL units
    return total_mse / total_n, total_nll / total_n, mse_ch


# ---------------- Export: single-model *stateless* TorchScript (PHYSICAL outputs) ---------------- #

class _ExportStateless(nn.Module):
    """
    forward(x_raw, h_prev) -> (mu_phys, logvar_phys, h_next)
      x_raw:  [B,8]  (PHYSICAL inputs: ax,ay,az,wx,wy,wz,cpsi,spsi)
      h_prev: Optional[Tensor] [L,B,H] or None
      h_next: [L,B,H]

    Internals:
      x̂ = (x_raw - μ_in)/σ_in
      (μ̂, logvar̂) = core(x̂)
      μ_phys = μ̂*σ_out + μ_out
      logvar_phys = logvar̂ + 2*log(σ_out)
    """
    __constants__ = ['IN_DIM']
    def __init__(self, core: GRUObserverV5,
                 in_mu: torch.Tensor, in_std: torch.Tensor,
                 out_mu: torch.Tensor, out_std: torch.Tensor):
        super().__init__()
        self.core = core
        self.IN_DIM = 8
        OUT_DIM = int(core.fc_mu.out_features)
        # register buffers for stats
        self.register_buffer("in_mu",  in_mu.view(1, self.IN_DIM))
        self.register_buffer("in_std", in_std.view(1, self.IN_DIM))
        self.register_buffer("out_mu",  out_mu.view(1, OUT_DIM))
        self.register_buffer("out_std", out_std.view(1, OUT_DIM))
        self.register_buffer("log_out_std2", (2.0 * torch.log(out_std.view(1, OUT_DIM))))

    def forward(self, x_raw: torch.Tensor, h_prev: Optional[torch.Tensor]):
        if not (x_raw.dim() == 2 and x_raw.size(-1) == self.IN_DIM):
            raise RuntimeError(f"expected x_raw of shape [B,{self.IN_DIM}]")
        B = x_raw.size(0)

        # Read shape metadata from the core (no direct .gru access)
        L = int(self.core.num_layers)
        H = int(self.core.hidden)

        # normalize input
        x_hat = (x_raw - self.in_mu) / self.in_std  # [B,8]
        x_emb = self.core.input_fc(x_hat).unsqueeze(1)   # [B,1,embed]

        # hidden
        if h_prev is None:
            h_prev = torch.zeros(L, B, H, device=x_raw.device)
        else:
            bad = (h_prev.dim()!=3) or (h_prev.size(0)!=L) or (h_prev.size(1)!=B) or (h_prev.size(2)!=H) or (h_prev.device!=x_raw.device)
            if bad:
                h_prev = torch.zeros(L, B, H, device=x_raw.device)

        # Run stacked 1-layer GRUs (no dropout in export)
        x = x_emb
        h_next_list = []
        for idx, gru_l in enumerate(self.core.gru_layers):   # enumerate ModuleList (TS-supported)
            h_l_prev = h_prev[idx:idx+1, :, :]               # [1,B,H]
            out_l, h_l_next = gru_l(x, h_l_prev)             # out_l:[B,1,H], h_l_next:[1,B,H]
            h_next_list.append(h_l_next)
            x = out_l

        out = x                    # [B,1,H]
        z = out[:, -1, :]          # [B,H]

        mu_hat    = self.core.fc_mu(z)                         # [B,D]
        logvar_hat = torch.clamp(self.core.fc_lv(z), -10.0, 5.0)

        # denormalize to PHYSICAL
        mu_phys     = mu_hat * self.out_std + self.out_mu
        logvar_phys = logvar_hat + self.log_out_std2

        h_next = torch.cat(h_next_list, dim=0)  # [L,B,H]
        return mu_phys, logvar_phys, h_next


def export_torchscript_stateless(model: GRUObserverV5, out_path: Path, norm: NormStats):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    wrapped = _ExportStateless(
        model_cpu,
        torch.from_numpy(norm.in_mu),
        torch.from_numpy(norm.in_std),
        torch.from_numpy(norm.out_mu),
        torch.from_numpy(norm.out_std),
    )
    scripted = torch.jit.script(wrapped)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    scripted.save(str(tmp))
    tmp.replace(out_path)
    print(f"Saved TorchScript (stateless single-model): {out_path}")


# ---------------- Export: ENSEMBLE *stateless* TorchScript (PHYSICAL outputs) ---------------- #

class _EnsembleExportStateless(nn.Module):
    """
    Each member is a stateless TS module:
        member(x_raw, Hprev_m) -> (mu_phys_m, logvar_phys_m, Hnext_m)

    Predictive covariance (PHYSICAL units):
        Σ* = E_m[Σ_m] + Cov_m[μ_m]
      where Σ_m = diag(exp(logvar_phys_m)) is aleatoric (diag),
      and Cov_m[μ_m] is the (full) ensemble covariance across means.

    forward(x_raw, Hprev) -> (mu_phys_mean, cov_phys, Hnext)
      x_raw:        [B,8]      (PHYSICAL inputs)
      Hprev:        [M,L,B,H]  or None
      mu_phys_mean: [B,D]      (PHYSICAL mean)
      cov_phys:     [B,D,D]    (PHYSICAL predictive covariance — FULL matrix)
      Hnext:        [M,L,B,H]
    """
    __constants__ = ['M','L','H']
    def __init__(self, members: List[_ExportStateless]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.M = len(members)
        if self.M < 1:
            self.L = 0; self.H = 0
        else:
            core0 = self.members[0].core
            self.L = int(core0.num_layers)
            self.H = int(core0.hidden)

    def forward(self, x_raw: torch.Tensor, Hprev: Optional[torch.Tensor]):
        if self.M < 1:
            raise RuntimeError("Ensemble has no members")
        B = x_raw.size(0); M = self.M; L = self.L; H = self.H

        if Hprev is None:
            Hprev = torch.zeros(M, L, B, H, device=x_raw.device)
        else:
            bad = (Hprev.dim()!=4) or (Hprev.size(0)!=M) or (Hprev.size(1)!=L) or (Hprev.size(2)!=B) or (Hprev.size(3)!=H) or (Hprev.device!=x_raw.device)
            if bad:
                Hprev = torch.zeros(M, L, B, H, device=x_raw.device)

        mus: List[torch.Tensor] = []
        logvars: List[torch.Tensor] = []
        Hnext_list: List[torch.Tensor] = []

        for m, member in enumerate(self.members):
            mu_phys_m, lv_phys_m, Hm_next = member(x_raw, Hprev[m])   # [B,D], [B,D], [L,B,H]
            mus.append(mu_phys_m)
            logvars.append(lv_phys_m)
            Hnext_list.append(Hm_next.unsqueeze(0))                   # [1,L,B,H]

        MU = torch.stack(mus, dim=0)                    # [M,B,D] physical means
        LV = torch.stack(logvars, dim=0)                # [M,B,D] physical log-variances
        VAR_m = torch.exp(LV)                           # [M,B,D] per-model variance (physical)

        mu_phys_mean = MU.mean(dim=0)                   # [B,D]

        # Aleatoric (diag) expected covariance E[Σ_m]
        e_sigma_diag = VAR_m.mean(dim=0)                # [B,D]
        cov_aleatoric = torch.diag_embed(e_sigma_diag)  # [B,D,D]

        # Epistemic covariance of means Cov_m[μ_m] (FULL matrix)
        centered = MU - mu_phys_mean.unsqueeze(0)       # [M,B,D]
        cov_epistemic = torch.einsum('mbi,mbj->bij', centered, centered) / float(M)  # [B,D,D]

        cov_phys = cov_aleatoric + cov_epistemic        # [B,D,D]

        Hnext = torch.cat(Hnext_list, dim=0)            # [M,L,B,H]
        return mu_phys_mean, cov_phys, Hnext


# ---------------- CLI / Main ---------------- #

def main():
    import os
    torch.backends.cudnn.benchmark = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (e.g., data/nn_dataset_v5/train.csv)")
    ap.add_argument("--val", default=None,   help="Path to val.csv (e.g., data/nn_dataset_v5/val.csv)")
    ap.add_argument("--out", default="data/nn_model_v5", help="Output folder")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=50, help="GRU unroll/window length")
    ap.add_argument("--lr", type=float, default=5e-4)  # stable default
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no_plot", action="store_true", help="Disable live plotting")
    # Ensemble settings
    ap.add_argument("--ensemble", type=int, default=1, help="Number of models to train in the ensemble")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bagging_frac", type=float, default=1.0,
                    help="Fraction of training windows per member (<=1.0) for bagging")
    # Loss schedule
    ap.add_argument("--switch_to_nll_ep", type=int, default=10,
                    help="Epoch at which to switch from MSE to NLL training")
    # Physics loss knobs
    ap.add_argument("--lambda_phys", type=float, default=1e-0,
                    help="Weight of physics model_loss during MSE warmup (set 0 to disable)")
    ap.add_argument("--phys_dt", type=float, default=0.0,
                    help="Sample time [s] for physics loss; if 0, auto from train.csv median Δt")
    # Normalization source
    ap.add_argument("--norm_json", default="",
                    help="Path to norm_stats.json (train-split). If empty, stats are computed from --train.")
    ap.add_argument("--in_std_floor",  type=float, default=1e-2, help="Min std for inputs")
    ap.add_argument("--out_std_floor", type=float, default=1e-2, help="Min std for targets")
    # Loader perf knobs
    ap.add_argument("--num_workers", type=int, default=min(8, (os.cpu_count() or 4)))
    ap.add_argument("--prefetch_factor", type=int, default=4)
    args = ap.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
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

    # Load or compute normalization (train-split only)
    if args.norm_json:
        norm_path = resolve(args.norm_json)
        if not norm_path.exists():
            raise SystemExit(f"--norm_json not found: {norm_path}")
        norm = NormStats.from_json(norm_path)
        print(f"[norm] loaded from {norm_path}")
    else:
        norm = NormStats.from_train_csv(train_path)
        print(f"[norm] computed from train.csv")

    # Apply std floors and persist exact stats used
    norm.in_std  = np.maximum(norm.in_std,  args.in_std_floor)
    norm.out_std = np.maximum(norm.out_std, args.out_std_floor)
    with open(out_dir / "norm_stats_used.json", "w") as f:
        json.dump(norm.to_json_dict(), f, indent=2)

    # Sanity: columns exist
    req_cols = ["run_id","t"] + INPUT_COLS + TARGET_COLS
    df_chk = pd.read_csv(train_path, nrows=1)
    missing = [c for c in req_cols if c not in df_chk.columns]
    if missing:
        raise SystemExit(f"Missing columns in train.csv: {missing}")

    # Determine dt for physics loss (if not provided)
    if args.phys_dt > 0:
        dt_used = float(args.phys_dt)
    else:
        df_dt = pd.read_csv(train_path, usecols=["run_id","t"])
        dts = []
        for _, g in df_dt.groupby("run_id"):
            tt = g["t"].to_numpy()
            if len(tt) >= 2:
                dts.append(np.median(np.diff(tt)))
        dt_used = float(np.median(dts)) if len(dts) else 0.02
    print(f"[phys] dt={dt_used:.6f}s, lambda_phys={args.lambda_phys}")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Validation loader
    dl_val = None
    if val_path:
        ds_val = WindowDataset(str(val_path), seq_len=args.seq, norm=norm)
        n_val = len(ds_val)
        if n_val == 0:
            raise SystemExit("No validation windows created. Reduce --seq or ensure val runs have ≥ seq rows.")
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=max(1024, args.batch), shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
            persistent_workers=True, prefetch_factor=args.prefetch_factor, drop_last=False
        )

    # Pre-create tensors for denorm in evaluate/train (on device)
    out_mu_t  = torch.from_numpy(norm.out_mu).to(device)
    out_std_t = torch.from_numpy(norm.out_std).to(device)

    # Ensemble training
    ens_dir = out_dir / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    member_weight_paths: List[Path] = []
    rng = np.random.default_rng(args.seed)

    for m in range(1, args.ensemble + 1):
        print(f"\n===== Training ensemble member {m}/{args.ensemble} =====")
        torch.manual_seed(int(rng.integers(0, 2**31-1)))

        ds_train = WindowDataset(str(train_path), seq_len=args.seq, norm=norm)
        n_train = len(ds_train)
        if n_train == 0:
            raise SystemExit("No training windows created. Reduce --seq or ensure train runs have ≥ seq rows.")
        frac = args.bagging_frac if (0.0 < args.bagging_frac <= 1.0) else 1.0

        if frac < 1.0:
            k = max(1, int(round(frac * n_train)))
            idx = rng.choice(n_train, size=k, replace=False)
            sampler = torch.utils.data.SubsetRandomSampler(idx.tolist())
            dl_train = torch.utils.data.DataLoader(
                ds_train, batch_size=args.batch, sampler=sampler,
                num_workers=args.num_workers, pin_memory=(device == "cuda"),
                persistent_workers=True, prefetch_factor=args.prefetch_factor,
                drop_last=True
            )
        else:
            dl_train = torch.utils.data.DataLoader(
                ds_train, batch_size=args.batch, shuffle=True,
                num_workers=args.num_workers, pin_memory=(device == "cuda"),
                persistent_workers=True, prefetch_factor=args.prefetch_factor,
                drop_last=True
            )

        print(f"Train windows (member {m}): {n_train} (seq={args.seq}, bagging_frac={frac})")
        if dl_val: print(f"Val windows:   {len(ds_val)} (seq={args.seq})")

        # Model / Optimizer
        model = GRUObserverV5().to(device)
        print(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Loss plots — MSE (log & linear) and NLL (linear)
        plot_mse_log = None if args.no_plot else LivePlot(out_dir, title=f"MSE Loss (log) — member {m}", yscale="log")
        plot_mse_lin = None if args.no_plot else LivePlot(out_dir, title=f"MSE Loss (linear) — member {m}", yscale="linear")
        plot_loss_nll = None if args.no_plot else LivePlot(out_dir, title=f"NLL Loss (member {m})", yscale="linear")

        # Per-output validation metrics: MSE (linear & log)
        plot_mse_ch_lin = None
        plot_mse_ch_log = None
        if (not args.no_plot) and (dl_val is not None):
            plot_mse_ch_lin = MetricPlot(out_dir, title=f"Validation MSE per output (linear) — member {m}",
                                         ylabel="MSE", labels=TARGET_COLS,
                                         png_name=f"val_mse_lin_member_{m}.png", yscale="linear")
            plot_mse_ch_log = MetricPlot(out_dir, title=f"Validation MSE per output (log) — member {m}",
                                         ylabel="MSE", labels=TARGET_COLS,
                                         png_name=f"val_mse_log_member_{m}.png", yscale="log")

        best_metric = float("inf")
        best_path = ens_dir / f"model_{m}_best.pt"

        for ep in range(1, args.epochs + 1):
            use_nll = (ep >= args.switch_to_nll_ep)
            train_loss, train_mse, train_nll = train_one_epoch(
                model, opt, dl_train, device, use_nll,
                out_mu_t, out_std_t,
                lambda_phys=float(args.lambda_phys), phys_dt=dt_used
            )

            if dl_val is not None:
                val_mse, val_nll, mse_ch = evaluate(model, dl_val, device, out_mu_t, out_std_t)
                active_val = val_nll if use_nll else val_mse

                msg = (f"[{ep:03d}] {'NLL' if use_nll else 'MSE'} | "
                       f"train={train_loss:.6f} (mse={train_mse:.6f}, nll={train_nll:.6f}) | "
                       f"val={active_val:.6f} (mse={val_mse:.6f}, nll={val_nll:.6f}) | "
                       f"MSE_ch[{', '.join(TARGET_COLS)}]=[{', '.join(f'{v:.4e}' for v in mse_ch)}]")
                print(msg)

                if plot_mse_log: plot_mse_log.update(ep, train_mse, val_mse, loss_name="MSE")
                if plot_mse_lin: plot_mse_lin.update(ep, train_mse, val_mse, loss_name="MSE")
                if plot_loss_nll: plot_loss_nll.update(ep, train_nll, val_nll, loss_name="NLL")
                if plot_mse_ch_lin: plot_mse_ch_lin.update(ep, mse_ch)
                if plot_mse_ch_log: plot_mse_ch_log.update(ep, mse_ch)

                if np.isfinite(active_val) and active_val < best_metric:
                    best_metric = active_val
                    torch.save(model.state_dict(), best_path)
            else:
                print(f"[{ep:03d}] {'NLL' if use_nll else 'MSE'} | train={train_loss:.6f}")
                if plot_mse_log: plot_mse_log.update(ep, train_mse, None, loss_name="MSE")
                if plot_mse_lin: plot_mse_lin.update(ep, train_mse, None, loss_name="MSE")
                if plot_loss_nll: plot_loss_nll.update(ep, train_nll, None, loss_name="NLL")

                active_train = train_nll if use_nll else train_mse
                if np.isfinite(active_train) and active_train < best_metric:
                    best_metric = active_train
                    torch.save(model.state_dict(), best_path)

        # Save final weights and export TS from **best** checkpoint (fallback to final if needed)
        weights_path = ens_dir / f"model_{m}.pt"
        torch.save(model.state_dict(), weights_path)
        export_src = best_path if best_path.exists() else weights_path

        state = torch.load(export_src, map_location="cpu", weights_only=True)
        model.load_state_dict(state); model.eval()
        export_torchscript_stateless(model, ens_dir / f"model_{m}_stateless.pt", norm)

        member_weight_paths.append(export_src)

        # Save plots
        if plot_mse_log: plot_mse_log.save_png(name=f"loss_mse_log_member_{m}.png")
        if plot_mse_lin: plot_mse_lin.save_png(name=f"loss_mse_lin_member_{m}.png")
        if plot_loss_nll: plot_loss_nll.save_png(name=f"loss_nll_member_{m}.png")
        if plot_mse_ch_lin: plot_mse_ch_lin.save_png()
        if plot_mse_ch_log: plot_mse_ch_log.save_png()

    # Export ensemble TorchScript if M>1 (PHYSICAL μ_mean + FULL predictive covariance)
    if len(member_weight_paths) > 1:
        wrapped_members: List[_ExportStateless] = []
        for pth in member_weight_paths:
            core = GRUObserverV5()
            sd = torch.load(pth, map_location="cpu", weights_only=True)
            core.load_state_dict(sd); core.eval()
            wrapped_members.append(_ExportStateless(
                core,
                torch.from_numpy(norm.in_mu),
                torch.from_numpy(norm.in_std),
                torch.from_numpy(norm.out_mu),
                torch.from_numpy(norm.out_std),
            ))
        ens = _EnsembleExportStateless(wrapped_members).eval()
        out_ens = out_dir / "ensemble_stateless.pt"
        tmp = out_ens.with_suffix(out_ens.suffix + ".tmp")
        scripted = torch.jit.script(ens)
        scripted.save(str(tmp))
        tmp.replace(out_ens)
        print(f"Saved TorchScript (stateless ENSEMBLE): {out_ens}")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "arch": {
                "in_dim": 8, "embed_dim": 80, "hidden": 80, "layers": 3, "dropout_p": 0.5,
                "heads": {"mu_hat": f"80->{len(TARGET_COLS)}", "logvar_hat": f"80->{len(TARGET_COLS)} (normalized)"}
            },
            "seq": args.seq,
            "lr": args.lr,
            "epochs": args.epochs,
            "inputs": INPUT_COLS,
            "targets": TARGET_COLS,
            "ensemble": args.ensemble,
            "bagging_frac": args.bagging_frac,
            "seed": args.seed,
            "switch_to_nll_ep": args.switch_to_nll_ep,
            "lambda_phys": args.lambda_phys,
            "phys_dt": dt_used,
            "norm_source": (str((resolve(args.norm_json) if args.norm_json else "")) or "computed_from_train_csv"),
            "in_std_floor": args.in_std_floor,
            "out_std_floor": args.out_std_floor,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

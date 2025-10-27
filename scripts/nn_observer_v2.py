# scripts/nn_observer_v2.py
# -------------------------------------------------------
# DeepVL-style GRU observer with ENSEMBLE covariance for EKF.
#
# Normalization (train-split stats):
#   - Dataset gives x̂ = (x-μ_in)/σ_in, ŷ = (y-μ_out)/σ_out to the model during TRAIN/VAL.
#   - EXPORTS (TorchScript) now take RAW IMU (physical) and:
#       * normalize inputs inside TS,
#       * denormalize mean outputs to PHYSICAL units,
#       * scale variances to PHYSICAL units.
#   -> C++ always passes raw IMU and receives physical μ and covariance.
#
# Architecture (fixed):
#   Input FC + ReLU: 6 → 60
#   GRU stack: 3 layers, hidden=60 (batch_first=True)
#   Dropout  : p=0.5 on the last GRU outputs
#   Heads    : fc_mu 60→6   (μ̂ = normalized means)
#              fc_lv 60→6   (log-variance in normalized space)
#
# Training:
#   Warmup with MSE for a few epochs, then switch to NLL (diag Gaussian).
#   (Use --switch_to_nll_ep to choose the switch epoch; default 10.)
#
# Ensemble:
#   Train M members (different seeds / optional bagging).
#   Export per-member stateless TorchScript (returns PHYSICAL μ, logvar) and
#   one ensemble TorchScript (returns PHYSICAL μ_mean and predictive cov diag).
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

# I/O columns
INPUT_COLS  = ["ax","ay","az","wx","wy","wz"]             # 6D IMU
TARGET_COLS = ["u","v","w","p","q","r"]                   # 6D linear+angular rates


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


# ---------------- Model: FC(6→60) + GRU×3(H=60) + Dropout(0.5) + Heads(60→6,60→6) ---------------- #

class VelRateGRUObserver(nn.Module):
    """
    Fixed architecture (cannot be changed via CLI/runtime):

        in_dim     = 6
        embed_dim  = 60
        hidden     = 60
        num_layers = 3
        dropout_p  = 0.5
        HEADS:  fc_mu: 60 -> 6     (μ̂ = normalized means)
                fc_lv: 60 -> 6     (log-variance in normalized space)

    Training path:
        x̂: [B,T,6] → FC: [B,T,60] → GRU×3: [B,T,60] → Dropout → (μ̂, logvar̂): [B,T,6] each
    """
    def __init__(self):
        super().__init__()
        self.in_dim: int = 6
        self.embed_dim: int = 60
        self.hidden: int = 60
        self.num_layers: int = 3
        self.dropout_p: float = 0.5

        self.input_fc = nn.Sequential(
            nn.Linear(self.in_dim, self.embed_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(self.embed_dim, self.hidden,
                          num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc_mu  = nn.Linear(self.hidden, 6)
        self.fc_lv  = nn.Linear(self.hidden, 6)  # log(var) in normalized space

    def _heads(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(z)
        mu = self.fc_mu(z)
        logvar = torch.clamp(self.fc_lv(z), -10.0, 5.0)
        return mu, logvar

    def forward_seq(self, x_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x̂ [B,T,6] -> (μ̂, logvar̂) each [B,T,6]
        """
        if x_hat.dim() != 3 or x_hat.size(-1) != self.in_dim:
            raise ValueError(f"forward_seq expects [B,T,{self.in_dim}]")
        x_emb = self.input_fc(x_hat)               # [B,T,60]
        out, _ = self.gru(x_emb)                   # [B,T,60]
        mu_hat, logvar_hat = self._heads(out)      # [B,T,6]
        return mu_hat, logvar_hat

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in_dim={self.in_dim}, embed_dim={self.embed_dim}, "
                f"hidden={self.hidden}, num_layers={self.num_layers}, dropout_p={self.dropout_p}, "
                f"HEADS='mu_hat:60→6, logvar_hat:60→6 (normalized)')")


# ---------------- Losses (computed in NORMALIZED space) ---------------- #

def mse_loss(mu_hat: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return torch.mean((mu_hat - y_hat) ** 2)

def nll_diag(mu_hat: torch.Tensor, logvar_hat: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Diagonal Gaussian NLL in normalized space (ignoring constant 0.5*log(2π)):
      0.5 * [ logvar̂ + (ŷ - μ̂)^2 / exp(logvar̂) ], sum over 6 dims, mean over batch*time
    """
    inv_var = torch.exp(-logvar_hat)
    nll = 0.5 * (logvar_hat + (y_hat - mu_hat)**2 * inv_var)
    return nll.sum(dim=-1).mean()


# ---------------- Live Plot (unchanged) ---------------- #

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
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            plt.ion(); plt.show()
        except Exception as e:
            print(f"[plot] disabled ({e})")
            self.enabled = False

    def update(self, ep: int, tr: float, va: Optional[float], loss_name: str):
        if not self.enabled:
            return
        self.epochs.append(ep)
        self.tr_vals.append(tr)
        self.va_vals.append(va if va is not None else math.nan)
        self.labels.append(loss_name)
        self.l_tr.set_data(self.epochs, self.tr_vals)
        self.l_va.set_data(self.epochs, self.va_vals)
        x_max = max(10, ep); self.ax.set_xlim(1, x_max)
        ys = [y for y in (self.tr_vals + self.va_vals) if math.isfinite(y)]
        if ys:
            ymin, ymax = min(ys), max(ys); ymax = ymin + 1e-6 if ymin == ymax else ymax
            pad = 0.05 * (ymax - ymin); self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.set_ylabel(loss_name)
        self.plt.tight_layout(); self.plt.pause(0.001)

    def save_png(self, name: str = "loss_curve.png"):
        if not self.enabled: return
        try:
            path = self.out_dir / name
            self.plt.savefig(path)
            print(f"[plot] saved: {path}")
        except Exception as e:
            print(f"[plot] save failed: {e}")


class MetricPlot6:
    """
    Live plot for 6-channel metrics (e.g., MAE or RMSE) over epochs.
    - Supports linear or log y-scale.
    - For log scale, values <= 0 are clamped to a tiny epsilon for plotting only.
    - Does NOT change the numeric values you print/save elsewhere.
    """
    def __init__(
        self,
        out_dir: Path,
        title: str,
        ylabel: str,
        labels: List[str],
        png_name: str,
        yscale: str = "linear",
    ):
        self.enabled = True
        self.out_dir = Path(out_dir)
        self.png_name = png_name
        self.labels = labels
        self.epochs: List[int] = []
        self.values = [[] for _ in range(len(labels))]  # per-channel history
        self._is_log = (yscale == "log")
        self._eps = 1e-12  # clamp for log plots

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

            # Seed safe limits so the first draw on a log axis works
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

        # Update series, clamping for log-scale visualization if needed
        for i, v in enumerate(vals):
            vv = float(v)
            if self._is_log:
                # On log axes, zero/negative/NaN/Inf cannot be drawn
                if not np.isfinite(vv) or vv <= 0.0:
                    vv = self._eps
                else:
                    vv = max(vv, self._eps)
            self.values[i].append(vv)
            self.lines[i].set_data(self.epochs, self.values[i])

        # Dynamic axes
        x_max = max(10, ep)
        self.ax.set_xlim(1, x_max)

        ys = [y for series in self.values for y in series if np.isfinite(y)]
        if ys:
            ymin, ymax = min(ys), max(ys)
            if self._is_log:
                # Ensure strictly positive bounds on log scale
                ymin = max(ymin, self._eps)
                # Avoid identical bounds; pad multiplicatively
                if ymax <= ymin:
                    ymax = ymin * 1.000001
                self.ax.set_ylim(ymin / 1.2, ymax * 1.2)
            else:
                if ymin == ymax:
                    ymax = ymin + 1e-6
                pad = 0.05 * (ymax - ymin)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        # Refresh
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
        Xwin = self.X_hat[start_idx:end_idx+1, :]   # [T,6] normalized
        Ywin = self.Y_hat[start_idx:end_idx+1, :]   # [T,6] normalized
        return torch.from_numpy(Xwin), torch.from_numpy(Ywin)


# ---------------- Train / Validate (normalized space; metrics in physical) ---------------- #

def train_one_epoch(model, opt, loader, device, use_nll: bool):
    model.train()
    total_loss, total_n = 0.0, 0
    total_mse, total_nll = 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        mu_hat, logvar_hat = model.forward_seq(xb)
        loss_mse = mse_loss(mu_hat, yb)
        loss_nll = nll_diag(mu_hat, logvar_hat, yb)
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
    sum_abs_err = torch.zeros(6, device=device, dtype=torch.float64)
    sum_sq_err  = torch.zeros(6, device=device, dtype=torch.float64)

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        mu_hat, logvar_hat = model.forward_seq(xb)      # normalized
        loss_mse = mse_loss(mu_hat, yb)
        loss_nll = nll_diag(mu_hat, logvar_hat, yb)
        total_mse += float(loss_mse.item()) * xb.size(0)
        total_nll += float(loss_nll.item()) * xb.size(0)
        total_n   += xb.size(0)

        # Denormalize to physical space for metrics
        mu_phys = (mu_hat * out_std + out_mu).reshape(-1, 6).to(torch.float64)
        y_phys  = (yb      * out_std + out_mu).reshape(-1, 6).to(torch.float64)
        err = (mu_phys - y_phys)
        sum_abs_err += err.abs().sum(dim=0)
        sum_sq_err  += (err**2).sum(dim=0)

    if total_n == 0:
        return float("nan"), float("nan"), [float("nan")]*6, [float("nan")]*6
    denom = float(total_n * loader.dataset.seq_len)
    mae  = (sum_abs_err / denom).tolist()
    rmse = torch.sqrt(sum_sq_err / denom).tolist()
    return total_mse / total_n, total_nll / total_n, mae, rmse


# ---------------- Export: single-model *stateless* TorchScript (PHYSICAL outputs) ---------------- #

class _ExportStateless(nn.Module):
    """
    forward(x_raw, h_prev) -> (mu_phys, logvar_phys, h_next)
      x_raw: [B,6]  (PHYSICAL IMU)
      h_prev: Optional[Tensor] [L,B,H] or None
      h_next: [L,B,H]

    Internally:
      x̂ = (x_raw - μ_in)/σ_in
      (μ̂, logvar̂) = core(x̂)
      μ_phys = μ̂*σ_out + μ_out
      logvar_phys = logvar̂ + 2*log(σ_out)
    """
    __constants__ = ['IN_DIM']
    def __init__(self, core: VelRateGRUObserver,
                 in_mu: torch.Tensor, in_std: torch.Tensor,
                 out_mu: torch.Tensor, out_std: torch.Tensor):
        super().__init__()
        self.core = core
        self.IN_DIM = 6
        # register buffers for stats (shape [1,6] for broadcast)
        self.register_buffer("in_mu",  in_mu.view(1, 6))
        self.register_buffer("in_std", in_std.view(1, 6))
        self.register_buffer("out_mu",  out_mu.view(1, 6))
        self.register_buffer("out_std", out_std.view(1, 6))
        self.register_buffer("log_out_std2", (2.0 * torch.log(out_std.view(1, 6))))

    def forward(self, x_raw: torch.Tensor, h_prev: Optional[torch.Tensor]):
        if not (x_raw.dim() == 2 and x_raw.size(-1) == self.IN_DIM):
            raise RuntimeError("expected x_raw of shape [B,6]")
        B = x_raw.size(0)
        L = int(self.core.gru.num_layers)
        H = int(self.core.gru.hidden_size)

        # normalize input
        x_hat = (x_raw - self.in_mu) / self.in_std  # [B,6]
        # handle hidden
        if h_prev is None:
            h_prev = torch.zeros(L, B, H, device=x_raw.device)
        else:
            bad = (h_prev.dim()!=3) or (h_prev.size(0)!=L) or (h_prev.size(1)!=B) or (h_prev.size(2)!=H) or (h_prev.device!=x_raw.device)
            if bad:
                h_prev = torch.zeros(L, B, H, device=x_raw.device)

        # run core on normalized input
        x_emb = self.core.input_fc(x_hat)                         # [B,60]
        out, h_next = self.core.gru(x_emb.unsqueeze(1), h_prev)   # [B,1,60], [L,B,60]
        z = self.core.dropout(out[:, -1, :])                      # [B,60]
        mu_hat = self.core.fc_mu(z)                               # [B,6] (normalized mean)
        logvar_hat = torch.clamp(self.core.fc_lv(z), -10.0, 5.0)  # [B,6] (normalized log-var)

        # denormalize outputs to PHYSICAL
        mu_phys = mu_hat * self.out_std + self.out_mu             # [B,6]
        logvar_phys = logvar_hat + self.log_out_std2              # [B,6]

        return mu_phys, logvar_phys, h_next


def export_torchscript_stateless(model: VelRateGRUObserver, out_path: Path, norm: NormStats):
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
        Σ* = E_m[Σ_m] + Var_m[μ_m]   (diagonal per axis), with Σ_m = diag(exp(logvar_phys_m))

    forward(x_raw, Hprev) -> (mu_phys_mean, cov_diag_phys, Hnext)
      x_raw:        [B,6]      (PHYSICAL inputs)
      Hprev:        [M,L,B,H]  or None
      mu_phys_mean: [B,6]      (PHYSICAL mean)
      cov_diag_phys:[B,6]      (PHYSICAL predictive covariance diag)
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
            g = self.members[0].core.gru
            self.L = int(g.num_layers)
            self.H = int(g.hidden_size)

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
            mu_phys_m, lv_phys_m, Hm_next = member(x_raw, Hprev[m])   # [B,6], [B,6], [L,B,H]
            mus.append(mu_phys_m)
            logvars.append(lv_phys_m)
            Hnext_list.append(Hm_next.unsqueeze(0))                   # [1,L,B,H]

        MU = torch.stack(mus, dim=0)                    # [M,B,6] physical means
        LV = torch.stack(logvars, dim=0)                # [M,B,6] physical log-variances
        VAR_m = torch.exp(LV)                           # [M,B,6] per-model variance (physical)

        mu_phys_mean = MU.mean(dim=0)                   # [B,6]
        e_sigma = VAR_m.mean(dim=0)                     # [B,6]
        e_y2    = (MU * MU).mean(dim=0)                 # [B,6]
        cov_diag_phys = torch.clamp(e_sigma + e_y2 - mu_phys_mean*mu_phys_mean, min=1e-12)  # [B,6]

        Hnext = torch.cat(Hnext_list, dim=0)            # [M,L,B,H]
        return mu_phys_mean, cov_diag_phys, Hnext


def export_ensemble_torchscript(member_paths: List[Path], out_path: Path, out_std: np.ndarray):
    """
    Load per-member weights, wrap into stateless modules, and save ensemble TorchScript.
    Covariance is scaled to PHYSICAL units using σ_out^2.
    """
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    members: List[_ExportStateless] = []
    for p in member_paths:
        core = VelRateGRUObserver()
        # Support both new and old PyTorch load signatures
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(p, map_location="cpu")
        core.load_state_dict(sd)
        core.eval()
        members.append(_ExportStateless(core))

    out_std_sq = torch.from_numpy((out_std.astype(np.float32) ** 2))  # [6]
    ens = _EnsembleExportStateless(members, out_std_sq).eval()
    scripted = torch.jit.script(ens)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    scripted.save(str(tmp))
    tmp.replace(out_path)
    print(f"Saved TorchScript (stateless ENSEMBLE): {out_path}")


# ---------------- CLI / Main ---------------- #

def main():
    import os
    torch.backends.cudnn.benchmark = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (e.g., data/nn_dataset/train.csv)")
    ap.add_argument("--val", default=None,   help="Path to val.csv (e.g., data/nn_dataset/val.csv)")
    ap.add_argument("--out", default="data/nn_model_v2", help="Output folder")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=50, help="GRU unroll/window length")
    ap.add_argument("--lr", type=float, default=1e-3)
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
    # Normalization source
    ap.add_argument("--norm_json", default="",
                    help="Path to norm_stats.json (train-split). If empty, stats are computed from --train.")
    ap.add_argument("--in_std_floor",  type=float, default=1e-2, help="Min std for inputs")
    ap.add_argument("--out_std_floor", type=float, default=1e-2, help="Min std for targets") 
    # Loader perf knobs
    ap.add_argument("--num_workers", type=int, default= min(8, (os.cpu_count() or 4)))
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

    # Pre-create tensors for denorm in evaluate (on device)
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
        model = VelRateGRUObserver().to(device)
        print(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Plots — MSE (log), NLL (linear), metrics (log)
        plot_loss_mse = None if args.no_plot else LivePlot(out_dir, title=f"MSE Loss (member {m})", yscale="log")
        plot_loss_nll = None if args.no_plot else LivePlot(out_dir, title=f"NLL Loss (member {m})", yscale="linear")
        plot_mae  = None
        plot_rmse = None
        if (not args.no_plot) and (dl_val is not None):
            plot_mae  = MetricPlot6(out_dir, title=f"Validation MAE (member {m})",
                                    ylabel="MAE", labels=TARGET_COLS,
                                    png_name=f"val_mae_member_{m}.png", yscale="log")
            plot_rmse = MetricPlot6(out_dir, title=f"Validation RMSE (member {m})",
                                    ylabel="RMSE", labels=TARGET_COLS,
                                    png_name=f"val_rmse_member_{m}.png", yscale="log")

        best_metric = float("inf")
        best_path = ens_dir / f"model_{m}_best.pt"

        for ep in range(1, args.epochs + 1):
            use_nll = (ep >= args.switch_to_nll_ep)
            train_loss, train_mse, train_nll = train_one_epoch(model, opt, dl_train, device, use_nll)
            if dl_val is not None:
                val_mse, val_nll, mae, rmse = evaluate(model, dl_val, device, out_mu_t, out_std_t)
                active_val = val_nll if use_nll else val_mse
                msg = (f"[{ep:03d}] {'NLL' if use_nll else 'MSE'} | "
                       f"train={train_loss:.6f} (mse={train_mse:.6f}, nll={train_nll:.6f}) | "
                       f"val={active_val:.6f} (mse={val_mse:.6f}, nll={val_nll:.6f}) | "
                       f"MAE[{', '.join(TARGET_COLS)}]=[{', '.join(f'{v:.4f}' for v in mae)}] | "
                       f"RMSE=[{', '.join(f'{v:.4f}' for v in rmse)}]")
                print(msg)
                if plot_loss_mse: plot_loss_mse.update(ep, train_mse, val_mse, loss_name="MSE")
                if plot_loss_nll: plot_loss_nll.update(ep, train_nll, val_nll, loss_name="NLL")
                if plot_mae:      plot_mae.update(ep, mae)
                if plot_rmse:     plot_rmse.update(ep, rmse)
                if np.isfinite(active_val) and active_val < best_metric:
                    best_metric = active_val
                    torch.save(model.state_dict(), best_path)
            else:
                print(f"[{ep:03d}] {'NLL' if use_nll else 'MSE'} | train={train_loss:.6f}")
                if plot_loss_mse: plot_loss_mse.update(ep, train_mse, None, loss_name="MSE")
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

        if plot_loss_mse: plot_loss_mse.save_png(name=f"loss_mse_member_{m}.png")
        if plot_loss_nll: plot_loss_nll.save_png(name=f"loss_nll_member_{m}.png")
        if plot_mae:      plot_mae.save_png()
        if plot_rmse:     plot_rmse.save_png()

    # Export ensemble TorchScript if M>1 (PHYSICAL μ_mean + predictive cov)
    if len(member_weight_paths) > 1:
        # Load best members and wrap them with stats-aware _ExportStateless so they accept RAW x and output PHYSICAL
        wrapped_members: List[_ExportStateless] = []
        for pth in member_weight_paths:
            core = VelRateGRUObserver()
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
                "in_dim": 6, "embed_dim": 60, "hidden": 60, "layers": 3, "dropout_p": 0.5,
                "heads": {"mu_hat": "60->6", "logvar_hat": "60->6 (normalized)"}
            },
            "seq": args.seq,
            "lr": args.lr,
            "epochs": args.epochs,
            "targets": TARGET_COLS,
            "ensemble": args.ensemble,
            "bagging_frac": args.bagging_frac,
            "seed": args.seed,
            "switch_to_nll_ep": args.switch_to_nll_ep,
            "norm_source": (str((resolve(args.norm_json) if args.norm_json else "")) or "computed_from_train_csv"),
            "in_std_floor": args.in_std_floor,
            "out_std_floor": args.out_std_floor,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

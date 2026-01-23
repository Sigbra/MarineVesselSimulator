#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_observer_v11.py — Chunked GPU training (one-step export) — NO physics loss
-----------------------------------------------------------------------------

Velocity-only NN (END frame) trained purely with supervised MSE.
This version removes all RAN/physics loss dependencies.

Key features
------------
• Fast GPU utilization via chunked training:
    - Train on mini-batches of windows shaped [B, T, 10] → [B, T, 3]
    - Streaming validation (strict T=1 step-by-step) to mimic runtime use
• Mixed precision (new AMP API): --amp {off,fp16,bf16}
• Optional TF32 math on CUDA: --tf32
• torch.compile acceleration: --compile
• Linear LR warmup: --warmup_epochs, --warmup_init_factor
• TorchScript export (trace-only) of:
    - Stateless one-step: forward(x:[1,1,10]) → y:[1,1,3]
    - Stateful one-step:  forward(x:[1,1,10], h:[L,1,H]) → (y:[1,1,3], h_next)

Inputs (10):  [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]
Outputs (3):  [vE, vN, vD]

Examples
-------
python3 -u scripts/nn_v11/nn_observer_v11.py \
  --train data/dataset03_v11_seq256_no_accel_bias_hem/train.csv \
  --val   data/dataset03_v11_seq256_no_accel_bias_hem/val.csv \
  --test  data/dataset03_v11_seq256_no_accel_bias_hem/test.csv \
  --out   data/model03_v11_ens4_h001_wd4_lr5_qw64_seq256_d3_hem \
  --epochs 600 --gpu --ensemble 4 \
  --norm_json data/dataset03_v11_seq256_no_accel_bias_hem/norm_stats.json

  python3 -u scripts/nn_v11/nn_observer_v11.py \
  --train data/dataset03_v11_seq256_no_accel_bias_sign/train.csv \
  --val   data/dataset03_v11_seq256_no_accel_bias_sign/val.csv \
  --test  data/dataset03_v11_seq256_no_accel_bias_sign/test.csv \
  --out   data/model03_v11_ens4_h001_wd4_lr5_qw64_seq256_d3_sign \
  --epochs 600 --gpu --ensemble 4 \
  --norm_json data/dataset03_v11_seq256_no_accel_bias_sign/norm_stats.json
"""

import os
import json
import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shutil
import multiprocessing as mp

# ==================== Columns ====================

IN_COLS     = ["ax","ay","az","qw","qx","qy","qz","tau_x","tau_y","tau_n"]
OUT_COLS    = ["vE","vN","vD"]
AUX_W_COLS  = ["w_est_x","w_est_y","w_est_z"]  # kept for compatibility; unused here


# ==================== Utilities ====================

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Normalizer:
    """Standardize inputs; standardize velocity outputs (if y_mean/std provided)."""
    def __init__(self, x_mean, x_std, y_mean=None, y_std=None):
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std  = np.asarray(x_std,  dtype=np.float32)
        self.y_mean = None if y_mean is None else np.asarray(y_mean, dtype=np.float32)
        self.y_std  = None if y_std  is None else np.asarray(y_std,  dtype=np.float32)
        self.x_std[self.x_std == 0] = 1.0
        if self.y_std is not None:
            self.y_std[self.y_std == 0] = 1.0
    def x(self, x_np: np.ndarray) -> np.ndarray:
        return (x_np - self.x_mean) / self.x_std
    def y(self, y_np: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            return y_np
        return (y_np - self.y_mean) / self.y_std
    def y_inv(self, y_norm_np: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            return y_norm_np
        return (y_norm_np * self.y_std) + self.y_mean

def load_norm(norm_json_path: str, x_train: np.ndarray = None, y_train: np.ndarray = None) -> Normalizer:
    if norm_json_path and os.path.isfile(norm_json_path):
        with open(norm_json_path, "r") as f:
            d = json.load(f)
        return Normalizer(d["x_mean"], d["x_std"], d.get("y_mean"), d.get("y_std"))
    if x_train is None:
        raise ValueError("norm_json not found and no training data provided to compute norms.")
    x_mean = x_train.mean(axis=0).tolist()
    x_std  = x_train.std(axis=0).tolist()
    y_mean = y_train.mean(axis=0).tolist() if y_train is not None else None
    y_std  = y_train.std(axis=0).tolist()  if y_train is not None else None
    return Normalizer(x_mean, x_std, y_mean, y_std)

def read_csv_columns(path: str,
                     in_cols=IN_COLS,
                     out_cols=OUT_COLS):
    df = pd.read_csv(path)
    miss_in  = [c for c in in_cols  if c not in df.columns]
    miss_out = [c for c in out_cols if c not in df.columns]
    if miss_in or miss_out:
        raise ValueError(f"{path}: missing columns. Missing inputs={miss_in}, missing outputs={miss_out}")

    x = df[in_cols].to_numpy(np.float32)
    y = df[out_cols].to_numpy(np.float32)

    # Normalize quaternion block for safety (unit quaternion)
    q = x[:, 3:7]
    qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    x[:, 3:7] = q / qn

    return x, y


# ==================== Datasets ====================

class ChunkedDataset(torch.utils.data.Dataset):
    """Windows of length T sampled non-overlapping across the whole sequence."""
    def __init__(self, x_nt10: np.ndarray, y_nt3: np.ndarray, T: int):
        assert x_nt10.ndim == 2 and y_nt3.ndim == 2 and x_nt10.shape[0] == y_nt3.shape[0]
        self.T = int(T)
        N = x_nt10.shape[0]
        W = N // self.T  # non-overlapping windows
        K = W * self.T
        self.x = torch.from_numpy(x_nt10[:K].reshape(W, self.T, -1))  # [W,T,10]
        self.y = torch.from_numpy(y_nt3[:K].reshape(W, self.T, -1))   # [W,T,3]
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class StreamingDataset(torch.utils.data.Dataset):
    """Return (x_t, y_t) per *time step* (order preserved)."""
    def __init__(self, x_nt10: np.ndarray, y_nt3: np.ndarray):
        assert x_nt10.ndim == 2 and y_nt3.ndim == 2 and x_nt10.shape[0] == y_nt3.shape[0]
        self.x = torch.from_numpy(x_nt10.astype(np.float32))  # [N,10]
        self.y = torch.from_numpy(y_nt3.astype(np.float32))   # [N,3]
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def build_chunked_loader(csv_path: str,
                         norm: Normalizer,
                         T: int,
                         batch: int,
                         workers: int,
                         prefetch: int,
                         device_is_cuda: bool):
    x, y = read_csv_columns(csv_path)
    x = norm.x(x); y = norm.y(y)
    ds = ChunkedDataset(x, y, T=T)
    pin = device_is_cuda
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=pin,
        prefetch_factor=prefetch if workers > 0 else None,
        persistent_workers=(workers > 0)
    )
    return loader, len(ds)

def build_streaming_loader(csv_path: str, norm: Normalizer, device_is_cuda: bool):
    x, y = read_csv_columns(csv_path)
    x = norm.x(x); y = norm.y(y)
    ds = StreamingDataset(x, y)
    pin = device_is_cuda
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=pin
    )
    return loader


# ==================== Model ====================

class VelNetV11(nn.Module):
    """
    GRU over input x[t] = [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n] (10-D)
    Output per step: [vE, vN, vD]

    forward accepts [B,T,10] or [B,1,10]  → returns (y:[B,T,3], h_next)
    """
    def __init__(self,
                 hidden: int = 256,
                 num_layers: int = 3,
                 dropout_p: float = 0.1,
                 use_layernorm: bool = True):
        super().__init__()
        self.input_dim   = 10
        self.hidden_GRU  = hidden
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_GRU,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.post_gru_norm = nn.LayerNorm(self.hidden_GRU) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(self.hidden_GRU, 3)

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B,T,10] or [B,1,10]; convert [B,1,10] to [B,1,10] (no-op), reject others softly
        if x.dim() == 3 and x.size(-1) == self.input_dim:
            return x
        if x.dim() == 4 and x.size(1) == 1 and x.size(-1) == self.input_dim:  # [B,1,T,10] → [B,T,10]
            return x.squeeze(1)
        # For export tracing safety, avoid raising here; best-effort reshape if [B,10]
        if x.dim() == 2 and x.size(-1) == self.input_dim:
            return x.unsqueeze(1)
        # Fall back (no crash during trace)
        return x

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        x = self._normalize_x(x)               # [B,T,10]
        if h is not None and h.dim() != 3:
            # For runtime clarity; not hit during trace
            raise ValueError(f"VelNetV11.forward: hidden must be [L,B,H], got {tuple(h.shape)}")

        z, h_next = self.gru(x, h)             # [B,T,H]
        #z = self.post_gru_norm(z) 
        z = self.dropout(z)
        y = self.out(z)                         # [B,T,3]
        return y, h_next


# ==================== Loss (MSE only) ====================

@dataclass
class LossWeights:
    mse: float = 1.0

def total_loss(pred_bT13: torch.Tensor,
               gt_bT13:   torch.Tensor,
               w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Shapes here are flattened (B*T, 1, 3) to reuse one-step loss;
    Physics loss removed: pure supervised MSE on [vE,vN,vD].
    """
    L_mse = F.mse_loss(pred_bT13, gt_bT13)
    L = w.mse * L_mse
    return L, {
        "total": float(L.detach().cpu()),
        "mse_v": float(L_mse.detach().cpu()),
        "phys_trans": 0.0,
    }


# ==================== Training (chunked, GPU) ====================

@dataclass
class TrainCfg:
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0.0
    dt: float = 0.02
    dropout_p: float = 0.1
    qwidth: int = 256
    device: str = "cuda"
    loss_w: "LossWeights" = field(default_factory=LossWeights)
    x_mean: torch.Tensor = None  # kept for API parity; not used
    x_std:  torch.Tensor = None  # kept for API parity; not used
    print_period: int = 5
    tbptt: int = 256            # not used in chunked path; kept for API parity
    amp_dtype: torch.dtype = torch.bfloat16
    use_compile: bool = False
    progress_every: int = 5
    warmup_epochs: int = 0
    warmup_init_factor: float = 0.1

def train_one_model_chunked(model: nn.Module,
                            train_loader,
                            val_loader_stream,
                            cfg: TrainCfg,
                            save_best_path: str | None = None,
                            save_last_path: str | None = None) -> Dict[str, List[float]]:
    """
    Train a single model with chunked batches and streaming validation.

    New behavior:
      - Tracks best validation loss (val_total).
      - If save_best_path is provided, saves CPU-cloned state_dict at each improvement.
      - If save_last_path is provided, saves the final-epoch CPU-cloned state_dict.
      - Attaches attributes to `model`:
          model.best_val_total_ (float)
          model.best_epoch_     (int, 1-based)
          model.best_state_dict_ (Dict[str, Tensor] on CPU)
    Returns:
      hist dict (unchanged keys), with extra 'best_val_total' and 'best_epoch'.
    """
    device = cfg.device
    model.to(device)

    if cfg.use_compile:
        try:
            model = torch.compile(model, mode="max-autotune", dynamic=False)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] failed, continuing without compile: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- Linear warmup (epoch-based) ---
    base_lr = cfg.lr
    def _lr_factor(epoch_idx: int) -> float:
        if cfg.warmup_epochs <= 0:
            return 1.0
        if cfg.warmup_epochs == 1:
            return 1.0 if epoch_idx >= 1 else cfg.warmup_init_factor
        if epoch_idx < cfg.warmup_epochs:
            f0 = cfg.warmup_init_factor
            t = epoch_idx / (cfg.warmup_epochs - 1)
            return f0 + (1.0 - f0) * t
        return 1.0

    # AMP (new API)
    use_fp16_scaler = (device == "cuda" and cfg.amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_fp16_scaler else None

    hist = {
        "train_total": [],
        "val_total": [],
        "val_mse_v": [],
        "val_by_output": [[] for _ in range(3)],
        "val_total_by_output": [[] for _ in range(3)],
    }

    best_val = float("inf")
    best_epoch = -1
    best_state_cpu = None  # CPU-cloned state_dict snapshot

    for epoch in range(cfg.epochs):
        # Set LR with warmup
        current_lr = base_lr * _lr_factor(epoch)
        for g in opt.param_groups: g["lr"] = current_lr

        model.train()
        train_sum = 0.0
        n_batches = 0

        print(f"[epoch {epoch+1}/{cfg.epochs}] starting… (lr={current_lr:.3e})")
        for b_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)  # [B,T,10]
            yb = yb.to(device, non_blocking=True)  # [B,T,3]

            with torch.amp.autocast(device_type="cuda",
                                    dtype=cfg.amp_dtype,
                                    enabled=(device=="cuda" and cfg.amp_dtype != torch.float32)):
                y_hat, _ = model(xb, None)       # [B,T,3]
                B, T = y_hat.shape[0], y_hat.shape[1]
                pred_bT13 = y_hat.reshape(B*T, 1, 3)
                gt_bT13   = yb.reshape(B*T, 1, 3)
                L, _ = total_loss(pred_bT13, gt_bT13, cfg.loss_w)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(L).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            train_sum += float(L.detach().cpu())
            n_batches += 1

            if (b_idx + 1) % cfg.progress_every == 0:
                print(f"  [epoch {epoch+1}] batch {b_idx+1}/{len(train_loader)}  loss={L.detach().item():.6g}")

        train_total = train_sum / max(1, n_batches)
        hist["train_total"].append(train_total)

        # ---- Validation (strict streaming, MSE only) ----
        model.eval()
        with torch.no_grad():
            tot_sum = 0.0
            mse_v_sum = 0.0
            per_out_mse_sums = [0.0] * 3
            per_out_total_sums = [0.0] * 3
            batches = 0

            h_val = None
            for x_t, y_t in val_loader_stream:
                x_t = x_t.to(device, non_blocking=True).unsqueeze(1)  # [1,1,10]
                y_t = y_t.to(device, non_blocking=True).unsqueeze(1)  # [1,1,3]
                y_hat, h_val = model(x_t, h_val)
                # loss per step
                L_mse = F.mse_loss(y_hat, y_t)
                tot_sum += float(L_mse.detach().cpu())
                mse_v_sum += float(L_mse.detach().cpu())

                mse_vec = torch.mean((y_hat - y_t).pow(2), dim=(0, 1))  # [3]
                per_total = mse_vec.cpu().numpy()
                for i in range(3):
                    per_out_mse_sums[i] += float(mse_vec[i].cpu())
                    per_out_total_sums[i] += float(per_total[i])
                batches += 1

            val_total = tot_sum / max(1, batches)
            hist["val_total"].append(val_total)
            hist["val_mse_v"].append(mse_v_sum / max(1, batches))
            for i in range(3):
                hist["val_by_output"][i].append(per_out_mse_sums[i] / max(1, batches))
                hist["val_total_by_output"][i].append(per_out_total_sums[i] / max(1, batches))

        # ---- Best checkpoint tracking/saving ----
        if val_total < best_val:
            best_val = val_total
            best_epoch = epoch + 1  # 1-based
            # snapshot CPU weights so caller can reuse without re-reading from disk
            best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_best_path is not None:
                torch.save({"state_dict": best_state_cpu}, save_best_path)
                print(f"[ckpt] ✓ new best @ epoch {best_epoch}: val_total={best_val:.6g}  → {save_best_path}")

        if (epoch + 1) % cfg.print_period == 0:
            print(f"[{epoch+1:03d}/{cfg.epochs}] "
                  f"train_total={train_total:.6g} (mse={train_total:.6g}, phys=0)  "
                  f"val_total={val_total:.6g} (mse={hist['val_mse_v'][-1]:.6g}, phys=0)")

    # ---- Save LAST checkpoint if requested ----
    if save_last_path is not None:
        last_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": last_state_cpu}, save_last_path)
        print(f"[ckpt] saved last epoch weights → {save_last_path}")

    # annotate model and hist with best info
    model.best_val_total_ = float(best_val)
    model.best_epoch_ = int(best_epoch)
    model.best_state_dict_ = best_state_cpu
    hist["best_val_total"] = float(best_val)
    hist["best_epoch"] = int(best_epoch)

    return hist


# ==================== Plotting & save helpers ====================

def plot_training_curves(hist: Dict[str, List[float]], out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(hist["train_total"]) + 1)

    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.xlabel("epoch"); plt.ylabel("total loss"); plt.title("Total Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_linear.svg"); plt.close()

    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("total loss (log)"); plt.title("Total Loss (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_log.svg"); plt.close()

    names = ["vE", "vN", "vD"]
    plt.figure()
    for i in range(3):
        plt.plot(epochs, hist["val_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val MSE per output"); plt.title("Validation MSE per output (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_mse_per_output.svg"); plt.close()

    plt.figure()
    for i in range(3):
        plt.plot(epochs, hist["val_total_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val TOTAL per output"); plt.title("Validation TOTAL per output (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_total_per_output.svg"); plt.close()


def compute_metrics(y_pred: np.ndarray, y_gt: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_gt
    mse = np.mean(err**2, axis=0)         # [3]
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)
    out = {
        "MSE_vE": float(mse[0]), "MSE_vN": float(mse[1]), "MSE_vD": float(mse[2]),
        "RMSE_vE": float(rmse[0]), "RMSE_vN": float(rmse[1]), "RMSE_vD": float(rmse[2]),
        "MAE_vE": float(mae[0]), "MAE_vN": float(mae[1]), "MAE_vD": float(mae[2]),
        "MSE_mean": float(np.mean(mse)),
        "RMSE_mean": float(np.sqrt(np.mean(mse))),
        "MAE_mean": float(np.mean(mae)),
        "MSE_total":  float(np.sum(mse)),
        "RMSE_total": float(np.sqrt(np.sum(mse))),
        "MAE_total":  float(np.sum(mae)),
    }
    return out


# ==================== TorchScript export (trace-only) ====================

class StatelessOneStep(nn.Module):
    """Wrap VelNetV11 but ignore hidden state (one-step). forward expects [B,1,10]."""
    def __init__(self, core: VelNetV11):
        super().__init__()
        self.core = core
    def forward(self, x_b1d: torch.Tensor):
        # Normalize to [B,1,10]
        if x_b1d.dim() == 2 and x_b1d.size(-1) == 10:
            x_b1d = x_b1d.unsqueeze(1)
        y, _ = self.core(x_b1d, None)   # [B,1,3]
        return y

class StatefulOneStep(torch.nn.Module):
    """TorchScript-friendly stateful wrapper for runtime.

    Methods:
      - forward(x:[1,1,10], h:[L,1,H]) -> (y:[1,1,3], h_next)
      - init_state(batch:int=1) -> zeros [L,B,H]
    """
    def __init__(self, core: VelNetV11):
        super().__init__()
        self.core = core
        self.num_layers = core.gru.num_layers
        self.hidden_size = core.gru.hidden_size

    @torch.jit.export
    def init_state(self, batch: int = 1):
        return torch.zeros(self.num_layers, batch, self.hidden_size, dtype=torch.float32)

    def forward(self, x_b1d: torch.Tensor, h_in: torch.Tensor):
        y, h_next = self.core(x_b1d, h_in)
        return y, h_next

def _export_torchscript_stateless_one_step(model: VelNetV11, out_path: str):
    model_cpu = model.eval().to("cpu")
    wrapper = StatelessOneStep(model_cpu).eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Trace only (avoid scripting shape checks)
    example = torch.zeros(1, 1, 10, dtype=torch.float32)
    scripted = torch.jit.trace(wrapper, example, strict=False)
    scripted.save(out_path)
    print(f"[export] wrote TorchScript (one-step stateless): {out_path}")

def _export_torchscript_stateful_one_step(model: VelNetV11, out_path: str):
    core = model.eval().to("cpu")
    wrapper = StatefulOneStep(core).eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    example_x = torch.zeros(1, 1, 10, dtype=torch.float32)
    example_h = torch.zeros(core.gru.num_layers, 1, core.gru.hidden_size, dtype=torch.float32)
    scripted = torch.jit.trace(wrapper, (example_x, example_h), strict=False)
    scripted.save(out_path)
    print(f"[export] wrote TorchScript (one-step STATEFUL): {out_path}")


# ==================== Streaming evaluation ====================

@torch.no_grad()
def evaluate_on_loader_streaming(model: VelNetV11, loader, device: str = "cpu"):
    model.eval()
    preds, gts = [], []
    h = None
    for x_t, y_t in loader:
        x_t = x_t.to(device).unsqueeze(1)  # [1,1,10]
        y_t = y_t.to(device).unsqueeze(1)  # [1,1,3]
        y_hat, h = model(x_t, h)
        preds.append(y_hat.cpu().numpy())
        gts.append(y_t.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, 3)
    gts   = np.concatenate(gts,   axis=0).reshape(-1, 3)
    return preds, gts

def save_test_outputs(out_dir: str, pred_phy: np.ndarray, gt_phy: np.ndarray, metrics: Dict[str, float]):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "vE_gt": gt_phy[:,0], "vN_gt": gt_phy[:,1], "vD_gt": gt_phy[:,2],
        "vE_pred": pred_phy[:,0], "vN_pred": pred_phy[:,1], "vD_pred": pred_phy[:,2],
    })
    csv_path = os.path.join(out_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[test] wrote predictions: {csv_path}")
    print(f"[test] wrote metrics:     {os.path.join(out_dir, 'test_metrics.json')}")


# ==================== CLI / main ====================
def main():
    parser = argparse.ArgumentParser(description="Train velocity-only NN (MSE only) with fast GPU chunked GRU.")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val",   required=True, help="Path to validation CSV")
    parser.add_argument("--test",  required=True, help="Path to test CSV")
    parser.add_argument("--out",   required=True, help="Output directory (models, plots)")

    # Training shape / chunking
    parser.add_argument("--chunk_len",   type=int, default=256, help="Window length T") # same as --seq in make_nn_dataset 
    parser.add_argument("--chunk_batch", type=int, default=64,  help="Windows per batch B")
    parser.add_argument("--tbptt",       type=int, default=256, help="Kept for parity; unused in chunked mode")

    parser.add_argument("--epochs",      type=int, default=1000, help="Training epochs")
    parser.add_argument("--lr",          type=float, default=5e-4, help="Learning rate") #1e-3
    parser.add_argument("--wd",          type=float, default=1e-4, help="Weight decay") #1e-4
    parser.add_argument("--qwidth",      type=int, default=64, help="GRU hidden size") #64
    parser.add_argument("--dropout",     type=float, default=0.3, help="GRU dropout p") #0.3

    # Device / perf
    parser.add_argument("--gpu",         action="store_true", help="Use CUDA if available")
    parser.add_argument("--workers",     type=int, default=4, help="DataLoader workers") #4
    parser.add_argument("--prefetch",    type=int, default=4, help="DataLoader prefetch factor (per worker)") #4
    parser.add_argument("--amp",         type=str, default="bf16", choices=["off","fp16","bf16"], help="AMP dtype")
    parser.add_argument("--tf32",        action="store_true", help="Enable TF32 on CUDA")
    parser.add_argument("--compile",     action="store_true", help="Enable torch.compile")
    parser.add_argument("--progress_every", type=int, default=160, help="Print every N batches") #1

    # Ensemble/normalization
    parser.add_argument("--ensemble",    type=int, default=4, help="Number of models to train")
    parser.add_argument("--norm_json",   type=str, default="", help="Normalization JSON path")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed (member 0). Members use seed+i")

    # Warmup
    parser.add_argument("--warmup_epochs",       type=int, default=0,
                        help="Number of warmup epochs (linear ramp). 0 disables warmup.")
    parser.add_argument("--warmup_init_factor",  type=float, default=0.1,
                        help="LR multiplier at first warmup epoch (e.g., 0.1 → 10% of --lr).")

    args = parser.parse_args()

    # Multiprocessing start method (for workers>0)
    try:
        mp.set_start_method("spawn", force=True)
        print("[mp] start method set to 'spawn'")
    except RuntimeError:
        pass

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        if args.tf32:
            try:
                torch.backends.cuda.matmul.fp32_precision = "tf32"
            except Exception:
                pass
            try:
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            except Exception:
                pass
    else:
        print("Device: CPU")

    amp_map = {"off": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = amp_map[args.amp]

    # Load once to compute/fallback norms for I/O
    def _read_cols(path):
        x, y = read_csv_columns(path)
        return x, y
    x_tmp, y_tmp = _read_cols(args.train)
    norm = load_norm(args.norm_json, x_tmp, y_tmp)
    with open(os.path.join(args.out, "norm_used.json"), "w") as f:
        json.dump({
            "x_mean": norm.x_mean.tolist(),
            "x_std":  norm.x_std.tolist(),
            "y_mean": norm.y_mean.tolist() if norm.y_mean is not None else None,
            "y_std":  norm.y_std.tolist()  if norm.y_std  is not None else None
        }, f, indent=2)

    # Data loaders
    train_loader, n_windows = build_chunked_loader(
        args.train, norm,
        T=args.chunk_len, batch=args.chunk_batch,
        workers=args.workers, prefetch=args.prefetch,
        device_is_cuda=(device=="cuda")
    )
    val_loader_stream = build_streaming_loader(args.val, norm, device_is_cuda=(device=="cuda"))
    test_loader_stream = build_streaming_loader(args.test, norm, device_is_cuda=(device=="cuda"))

    batches_per_epoch = len(train_loader)
    print("\n" + "="*90)
    print(f"Training config: B={args.chunk_batch}, T={args.chunk_len}, amp={args.amp}, "
          f"tf32={args.tf32}, workers={args.workers}, prefetch={args.prefetch}, compile={args.compile}")
    print(f"Dataset windows: {n_windows}  (each T={args.chunk_len})  → batches/epoch ≈ {batches_per_epoch}")
    print("="*90 + "\n")

    # Train ensemble
    ts_collect_dir = os.path.join(args.out, "ts")
    os.makedirs(ts_collect_dir, exist_ok=True)

    # ---------------- NEW: collect per-member test predictions for ensemble scoring ----------------
    preds_phy_members: List[np.ndarray] = []
    gts_phy_ref: np.ndarray | None = None
    # ------------------------------------------------------------------------------------------------

    for m in range(args.ensemble):
        print("="*90)
        print(f"Ensemble member {m+1}/{args.ensemble}")
        print("="*90)
        set_seed(args.seed + m)

        model = VelNetV11(hidden=args.qwidth, num_layers=3, dropout_p=args.dropout).to(device)
        cfg = TrainCfg(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
                       dt=0.01, dropout_p=args.dropout, qwidth=args.qwidth,
                       device=device, loss_w=LossWeights(mse=1.0),
                       x_mean=None, x_std=None,
                       print_period=max(1, min(args.epochs, 5)),
                       tbptt=args.tbptt,
                       amp_dtype=amp_dtype,
                       use_compile=args.compile,
                       progress_every=args.progress_every,
                       warmup_epochs=args.warmup_epochs,
                       warmup_init_factor=args.warmup_init_factor)

        member_dir = os.path.join(args.out, f"member_{m:02d}")
        os.makedirs(member_dir, exist_ok=True)
        with open(os.path.join(member_dir, "config.json"), "w") as f:
            json.dump({
                "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                "dropout_p": cfg.dropout_p, "hidden": cfg.qwidth,
                "device": cfg.device, "seed": args.seed + m,
                "amp": args.amp, "tf32": args.tf32, "compile": args.compile,
                "chunk_len": args.chunk_len, "chunk_batch": args.chunk_batch,
                "warmup_epochs": cfg.warmup_epochs, "warmup_init_factor": cfg.warmup_init_factor
            }, f, indent=2)

        best_path = os.path.join(member_dir, "model_best.pth")
        last_path = os.path.join(member_dir, "model_last.pth")

        hist = train_one_model_chunked(
            model, train_loader, val_loader_stream, cfg,
            save_best_path=best_path, save_last_path=last_path
        )

        with open(os.path.join(member_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        plot_training_curves(hist, out_dir=member_dir)

        # ---- Load BEST before exporting & testing ----
        ckpt = torch.load(best_path, map_location=device)
        state = ckpt["state_dict"]
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

        # ---- Export TorchScript ----
        stateless_path = os.path.join(member_dir, "ts", f"member_{m:02d}_onestep.pt")
        stateful_path  = os.path.join(member_dir, "ts", f"member_{m:02d}_onestep_stateful.pt")
        _export_torchscript_stateless_one_step(model, stateless_path)
        _export_torchscript_stateful_one_step(model,  stateful_path)
        shutil.copyfile(stateless_path, os.path.join(ts_collect_dir, f"member_{m:02d}_onestep.pt"))
        shutil.copyfile(stateful_path,  os.path.join(ts_collect_dir, f"member_{m:02d}_onestep_stateful.pt"))

        # ==== Per-member TEST evaluation (STREAMING) ====
        print("[test] Evaluating member on test set…")
        preds_norm_m, gts_norm_m = evaluate_on_loader_streaming(model.eval().to(device), test_loader_stream, device=device)

        if norm.y_std is None:
            preds_phy_m = preds_norm_m
            gts_phy_m   = gts_norm_m
        else:
            preds_phy_m = preds_norm_m * norm.y_std.reshape(1,3) + norm.y_mean.reshape(1,3)
            gts_phy_m   = gts_norm_m   * norm.y_std.reshape(1,3) + norm.y_mean.reshape(1,3)

        metrics_m = compute_metrics(preds_phy_m, gts_phy_m)
        save_test_outputs(member_dir, preds_phy_m, gts_phy_m, metrics_m)

        # ---------------- NEW: stash for ensemble evaluation ----------------
        preds_phy_members.append(preds_phy_m)
        if gts_phy_ref is None:
            gts_phy_ref = gts_phy_m
        else:
            # ensure alignment (same ordering/length); tolerate float noise
            if gts_phy_ref.shape != gts_phy_m.shape or not np.allclose(gts_phy_ref, gts_phy_m, atol=1e-6, rtol=1e-6):
                raise RuntimeError("Test ground-truth mismatch across members; check test loader determinism/order.")
        # --------------------------------------------------------------------

    # ================= NEW: Evaluate the FULL ENSEMBLE on the test set =================
    print("\n" + "="*90)
    print("[test] Evaluating COMPLETE ENSEMBLE on test set (mean across members)…")
    print("="*90)

    if len(preds_phy_members) == 0:
        raise RuntimeError("No member predictions collected; ensemble evaluation cannot run.")
    if gts_phy_ref is None:
        raise RuntimeError("No ground truth collected; ensemble evaluation cannot run.")

    preds_stack = np.stack(preds_phy_members, axis=0)   # [M, N, 3]
    preds_ens_mean = np.mean(preds_stack, axis=0)       # [N, 3]
    preds_ens_std  = np.std(preds_stack, axis=0)        # [N, 3] (optional uncertainty proxy)

    metrics_ens = compute_metrics(preds_ens_mean, gts_phy_ref)

    ensemble_dir = os.path.join(args.out, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)

    # Save mean-pred performance in same format as members
    save_test_outputs(ensemble_dir, preds_ens_mean, gts_phy_ref, metrics_ens)

    # Also save std (uncertainty proxy) alongside mean predictions
    df_ens = pd.DataFrame({
        "vE_gt": gts_phy_ref[:,0], "vN_gt": gts_phy_ref[:,1], "vD_gt": gts_phy_ref[:,2],
        "vE_pred_mean": preds_ens_mean[:,0], "vN_pred_mean": preds_ens_mean[:,1], "vD_pred_mean": preds_ens_mean[:,2],
        "vE_pred_std":  preds_ens_std[:,0],  "vN_pred_std":  preds_ens_std[:,1],  "vD_pred_std":  preds_ens_std[:,2],
    })
    df_ens_path = os.path.join(ensemble_dir, "test_predictions_with_std.csv")
    df_ens.to_csv(df_ens_path, index=False)
    print(f"[test] wrote ensemble predictions w/ std: {df_ens_path}")

    print("\nAll ensemble members trained, exported (from BEST), and evaluated (including full-ensemble test metrics).\n")

# def main():
#     parser = argparse.ArgumentParser(description="Train velocity-only NN (MSE only) with fast GPU chunked GRU.")
#     parser.add_argument("--train", required=True, help="Path to training CSV")
#     parser.add_argument("--val",   required=True, help="Path to validation CSV")
#     parser.add_argument("--test",  required=True, help="Path to test CSV")
#     parser.add_argument("--out",   required=True, help="Output directory (models, plots)")

#     # Training shape / chunking
#     parser.add_argument("--chunk_len",   type=int, default=256, help="Window length T")
#     parser.add_argument("--chunk_batch", type=int, default=128,  help="Windows per batch B")
#     parser.add_argument("--tbptt",       type=int, default=256, help="Kept for parity; unused in chunked mode")

#     parser.add_argument("--epochs",      type=int, default=1000, help="Training epochs")
#     parser.add_argument("--lr",          type=float, default=1e-3, help="Learning rate")
#     parser.add_argument("--wd",          type=float, default=1e-3, help="Weight decay")
#     parser.add_argument("--qwidth",      type=int, default=64, help="GRU hidden size") #128 -> final2, 256 -> final3, 64 -> final 4
#     parser.add_argument("--dropout",     type=float, default=0.05, help="GRU dropout p")

#     # Device / perf
#     parser.add_argument("--gpu",         action="store_true", help="Use CUDA if available")
#     parser.add_argument("--workers",     type=int, default=4, help="DataLoader workers")
#     parser.add_argument("--prefetch",    type=int, default=4, help="DataLoader prefetch factor (per worker)")
#     parser.add_argument("--amp",         type=str, default="bf16", choices=["off","fp16","bf16"], help="AMP dtype")
#     parser.add_argument("--tf32",        action="store_true", help="Enable TF32 on CUDA")
#     parser.add_argument("--compile",     action="store_true", help="Enable torch.compile")
#     parser.add_argument("--progress_every", type=int, default=1, help="Print every N batches")

#     # Ensemble/normalization
#     parser.add_argument("--ensemble",    type=int, default=4, help="Number of models to train")
#     parser.add_argument("--norm_json",   type=str, default="", help="Normalization JSON path")
#     parser.add_argument("--seed",        type=int, default=42, help="Random seed (member 0). Members use seed+i")

#     # Warmup
#     parser.add_argument("--warmup_epochs",       type=int, default=0,
#                         help="Number of warmup epochs (linear ramp). 0 disables warmup.")
#     parser.add_argument("--warmup_init_factor",  type=float, default=0.1,
#                         help="LR multiplier at first warmup epoch (e.g., 0.1 → 10% of --lr).")

#     args = parser.parse_args()

#     # Multiprocessing start method (for workers>0)
#     try:
#         mp.set_start_method("spawn", force=True)
#         print("[mp] start method set to 'spawn'")
#     except RuntimeError:
#         pass

#     os.makedirs(args.out, exist_ok=True)
#     set_seed(args.seed)

#     device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
#     if device == "cuda":
#         torch.backends.cudnn.benchmark = True
#         print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
#         # New TF32 controls
#         if args.tf32:
#             try:
#                 torch.backends.cuda.matmul.fp32_precision = "tf32"
#             except Exception:
#                 pass
#             try:
#                 torch.backends.cudnn.conv.fp32_precision = "tf32"
#             except Exception:
#                 pass
#     else:
#         print("Device: CPU")

#     # AMP dtype
#     amp_map = {"off": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
#     amp_dtype = amp_map[args.amp]

#     # Load once to compute/fallback norms for I/O (x_mean/std always used; y optional)
#     def _read_cols(path):
#         x, y = read_csv_columns(path)
#         return x, y
#     x_tmp, y_tmp = _read_cols(args.train)
#     norm = load_norm(args.norm_json, x_tmp, y_tmp)
#     with open(os.path.join(args.out, "norm_used.json"), "w") as f:
#         json.dump({
#             "x_mean": norm.x_mean.tolist(),
#             "x_std":  norm.x_std.tolist(),
#             "y_mean": norm.y_mean.tolist() if norm.y_mean is not None else None,
#             "y_std":  norm.y_std.tolist()  if norm.y_std  is not None else None
#         }, f, indent=2)

#     # Data loaders
#     train_loader, n_windows = build_chunked_loader(
#         args.train, norm,
#         T=args.chunk_len, batch=args.chunk_batch,
#         workers=args.workers, prefetch=args.prefetch,
#         device_is_cuda=(device=="cuda")
#     )
#     val_loader_stream = build_streaming_loader(args.val, norm, device_is_cuda=(device=="cuda"))
#     test_loader_stream = build_streaming_loader(args.test, norm, device_is_cuda=(device=="cuda"))

#     batches_per_epoch = len(train_loader)
#     print("\n" + "="*90)
#     print(f"Training config: B={args.chunk_batch}, T={args.chunk_len}, amp={args.amp}, "
#           f"tf32={args.tf32}, workers={args.workers}, prefetch={args.prefetch}, compile={args.compile}")
#     print(f"Dataset windows: {n_windows}  (each T={args.chunk_len})  → batches/epoch ≈ {batches_per_epoch}")
#     print("="*90 + "\n")

#     # Train ensemble
#     ts_collect_dir = os.path.join(args.out, "ts")
#     os.makedirs(ts_collect_dir, exist_ok=True)

#     for m in range(args.ensemble):
#         print("="*90)
#         print(f"Ensemble member {m+1}/{args.ensemble}")
#         print("="*90)
#         set_seed(args.seed + m)

#         model = VelNetV11(hidden=args.qwidth, num_layers=2, dropout_p=args.dropout).to(device)
#         cfg = TrainCfg(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
#                        dt=0.05, dropout_p=args.dropout, qwidth=args.qwidth,
#                        device=device, loss_w=LossWeights(mse=1.0),
#                        x_mean=None, x_std=None,
#                        print_period=max(1, min(args.epochs, 5)),
#                        tbptt=args.tbptt,
#                        amp_dtype=amp_dtype,
#                        use_compile=args.compile,
#                        progress_every=args.progress_every,
#                        warmup_epochs=args.warmup_epochs,
#                        warmup_init_factor=args.warmup_init_factor)

#         member_dir = os.path.join(args.out, f"member_{m:02d}")
#         os.makedirs(member_dir, exist_ok=True)
#         with open(os.path.join(member_dir, "config.json"), "w") as f:
#             json.dump({
#                 "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
#                 "dropout_p": cfg.dropout_p, "hidden": cfg.qwidth,
#                 "device": cfg.device, "seed": args.seed + m,
#                 "amp": args.amp, "tf32": args.tf32, "compile": args.compile,
#                 "chunk_len": args.chunk_len, "chunk_batch": args.chunk_batch,
#                 "warmup_epochs": cfg.warmup_epochs, "warmup_init_factor": cfg.warmup_init_factor
#             }, f, indent=2)

#         # ===== Train with best/last checkpointing =====
#         best_path = os.path.join(member_dir, "model_best.pth")
#         last_path = os.path.join(member_dir, "model_last.pth")

#         hist = train_one_model_chunked(
#             model, train_loader, val_loader_stream, cfg,
#             save_best_path=best_path, save_last_path=last_path
#         )

#         with open(os.path.join(member_dir, "history.json"), "w") as f:
#             json.dump(hist, f, indent=2)
#         plot_training_curves(hist, out_dir=member_dir)

#         # ---- Load BEST before exporting & testing ----
#         ckpt = torch.load(best_path, map_location=device)
#         state = ckpt["state_dict"]
#         # strip torch.compile prefix if present
#         if any(k.startswith("_orig_mod.") for k in state.keys()):
#             state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
#         model.load_state_dict(state)


#         # ---- Export TorchScript (trace only; stateless & stateful) ----
#         stateless_path = os.path.join(member_dir, "ts", f"member_{m:02d}_onestep.pt")
#         stateful_path  = os.path.join(member_dir, "ts", f"member_{m:02d}_onestep_stateful.pt")
#         _export_torchscript_stateless_one_step(model, stateless_path)
#         _export_torchscript_stateful_one_step(model,  stateful_path)
#         # Collect copies
#         shutil.copyfile(stateless_path, os.path.join(ts_collect_dir, f"member_{m:02d}_onestep.pt"))
#         shutil.copyfile(stateful_path,  os.path.join(ts_collect_dir, f"member_{m:02d}_onestep_stateful.pt"))

#         # ==== Per-member TEST evaluation (STREAMING) ====
#         print("[test] Evaluating member on test set…")
#         preds_norm_m, gts_norm_m = evaluate_on_loader_streaming(model.eval().to(device), test_loader_stream, device=device)
#         # Denormalize to physical units (if y stats provided)
#         if norm.y_std is None:
#             preds_phy_m = preds_norm_m
#             gts_phy_m   = gts_norm_m
#         else:
#             preds_phy_m = preds_norm_m * norm.y_std.reshape(1,3) + norm.y_mean.reshape(1,3)
#             gts_phy_m   = gts_norm_m   * norm.y_std.reshape(1,3) + norm.y_mean.reshape(1,3)
#         metrics_m = compute_metrics(preds_phy_m, gts_phy_m)
#         save_test_outputs(member_dir, preds_phy_m, gts_phy_m, metrics_m)

#     print("\nAll ensemble members trained, exported (from BEST), and evaluated.\n")


if __name__ == "__main__":
    main()
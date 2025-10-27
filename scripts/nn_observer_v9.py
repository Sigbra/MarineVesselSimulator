#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_observer_v9.py
Velocity-only NN (END frame) with RAN-like physics-informed translational loss.
Architecture: regular GRU over full input vector (10 channels), ω_b is passed only to the loss.

NOW WITH TEST EVALUATION (minimal changes):
- New CLI arg --test (required): path to test CSV
- After **each member** finishes training, evaluate that member on the test set (and save outputs under its folder)
- After all members, evaluate the final **ensemble** on the test set
- Save per-timestep predictions vs ground truth in physical units
- Save test metrics (MSE/MAE/RMSE per output and overall)

Inputs (10):  [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]   (no gyro)
Aux (3):      [w_est_x, w_est_y, w_est_z]  (BODY rates from q-Obs; used ONLY in physics loss)
Outputs (3):  [vE, vN, vD]

IMPORTANT:
- Training uses *normalized* inputs, but the physics loss uses **de-normalized**
  accel/quat/tau signals. We pass x_mean/x_std into the loss to de-normalize
  each batch, and pass ω_b (w_est) separately as raw.

Run example:
python3 scripts/nn_observer_v9.py \
  --train data/nn_dataset_v9_X_C0/train.csv \
  --val   data/nn_dataset_v9_X_C0/val.csv \
  --test  data/nn_dataset_v9_X_C0/test.csv \
  --out   data/nn_model_v9_ens4 \
  --seq 300 --epochs 200 --gpu --ensemble 4 \
  --norm_json data/nn_dataset_v9_X_C0/norm_stats.json
"""

import os
import math
import json
import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.functional as F

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shutil

# ---- RAN physics residual (uses custom rotation; τ in BODY; ω_b from w_est) ----
from ran_model_loss import ran_translational_loss_v9


# ==================== GRU model (inputs 10 -> outputs 3) ====================

class VelNetV9(nn.Module):
    """
    Regular GRU over full input:
      x[t] = [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]  (10-D)
    Output per step: [vE, vN, vD]
    """
    def __init__(self,
                 hidden: int = 64,
                 num_layers: int = 3,
                 dropout_p: float = 0.2,
                 use_layernorm: bool = True,
                 leaky: float = 0.01):
        super().__init__()
        self.input_dim   = 10
        self.hidden_GRU  = hidden
        self.hidden_head = (3 * hidden) // 4
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_GRU,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.post_gru_norm = nn.LayerNorm(self.hidden_GRU) if use_layernorm else nn.Identity()
        self.act = nn.LeakyReLU(leaky, inplace=True) if leaky > 0.0 else nn.ReLU(inplace=True)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_GRU, self.hidden_head),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_head, 3),
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(GRU in=10, hidden_GRU={self.hidden_GRU}, "
                f"hidden_head={self.hidden_head}, layers={self.num_layers}) -> 3")

    def forward(self, x_bt10: torch.Tensor) -> torch.Tensor:
        z, _ = self.gru(x_bt10)      # [B,T,H]
        z = self.post_gru_norm(z)
        z = self.act(z)
        y = self.head(z)             # [B,T,3] -> [vE,vN,vD]
        return y


# ==================== Losses ====================

@dataclass
class LossWeights:
    mse: float = 1.0
    trans: float = 1.0

def total_loss(pred_bt3: torch.Tensor,
               gt_bt3: torch.Tensor,
               x_norm_bt10: torch.Tensor,
               x_mean_t: torch.Tensor,
               x_std_t: torch.Tensor,
               w_est_bt3: torch.Tensor,
               dt: float,
               w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined supervised + physics loss.
    - pred_bt3: [B,T,3] model outputs in END frame
    - gt_bt3  : [B,T,3] ground-truth END velocity
    - x_norm_bt10: [B,T,10] normalized inputs
    - x_mean_t/x_std_t: [1,1,10] tensors (on device) for de-normalization
    - w_est_bt3: [B,T,3] BODY rates from q-Obs (raw, not normalized)
    """
    L_mse = F.mse_loss(pred_bt3, gt_bt3)

    # de-normalize inputs for physics loss
    x_raw = x_norm_bt10 * x_std_t + x_mean_t  # [B,T,10] real units

    # physics term (τ in BODY, custom R_nb, uses ω_b)
    L_phys = ran_translational_loss_v9(pred_bt3, x_raw, dt, w_est_bt3)

    L = w.mse * L_mse + w.trans * L_phys
    return L, {
        "total": float(L.detach().cpu()),
        "mse_v": float(L_mse.detach().cpu()),
        "phys_trans": float(L_phys.detach().cpu()),
    }


# ==================== Training scaffold ====================

@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    dt: float = 0.01
    dropout_p: float = 0.2
    qwidth: int = 64
    device: str = "cuda"
    loss_w: "LossWeights" = field(default_factory=LossWeights)
    x_mean: torch.Tensor = None  # [1,1,10]
    x_std:  torch.Tensor = None  # [1,1,10]

class SeqDataset(torch.utils.data.Dataset):
    """Return (x_bt10, y_bt3, w_bt3) per item."""
    def __init__(self, x_bt10: torch.Tensor, y_bt3: torch.Tensor, w_bt3: torch.Tensor):
        assert x_bt10.shape[:2] == y_bt3.shape[:2] == w_bt3.shape[:2]
        self.x = x_bt10; self.y = y_bt3; self.w = w_bt3
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx], self.w[idx]

def train_one_model(model: nn.Module,
                    train_loader,
                    val_loader,
                    cfg: TrainCfg) -> Dict[str, List[float]]:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist = {
        "train_total": [],
        "val_total": [],
        "val_mse_v": [],
        "val_by_output": [[] for _ in range(3)],
        "val_total_by_output": [[] for _ in range(3)],
    }

    best_val = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        # ---------------- Train ----------------
        model.train()
        train_sum = 0.0
        for xb, yb, wb in train_loader:
            xb = xb.to(cfg.device)         # [B,T,10] normalized
            yb = yb.to(cfg.device)         # [B,T,3]
            wb = wb.to(cfg.device)         # [B,T,3] raw ω_b
            pred = model(xb)               # [B,T,3]
            L, _ = total_loss(pred, yb, xb, cfg.x_mean.to(cfg.device), cfg.x_std.to(cfg.device),
                              wb, cfg.dt, cfg.loss_w)
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            train_sum += float(L.detach().cpu())
        train_total = train_sum / max(1, len(train_loader))
        hist["train_total"].append(train_total)

        # ---------------- Validation ----------------
        model.eval()
        with torch.no_grad():
            tot_sum = 0.0
            mse_v_sum = 0.0
            per_out_mse_sums = [0.0] * 3
            per_out_total_sums = [0.0] * 3
            batches = 0

            for xb, yb, wb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                wb = wb.to(cfg.device)
                pred = model(xb)
                L, logs = total_loss(pred, yb, xb, cfg.x_mean.to(cfg.device), cfg.x_std.to(cfg.device),
                                     wb, cfg.dt, cfg.loss_w)
                tot_sum += logs["total"]
                mse_v_sum += logs["mse_v"]

                mse_vec = torch.mean((pred - yb).pow(2), dim=(0, 1))  # [3]
                add_v = cfg.loss_w.trans * logs["phys_trans"]
                per_total = (cfg.loss_w.mse * mse_vec).cpu().numpy()
                per_total[0:3] += add_v

                for i in range(3):
                    per_out_mse_sums[i] += float(mse_vec[i].cpu())
                    per_out_total_sums[i] += float(per_total[i])
                batches += 1

            val_total = tot_sum / max(1, len(val_loader))
            hist["val_total"].append(val_total)
            hist["val_mse_v"].append(mse_v_sum / max(1, len(val_loader)))
            for i in range(3):
                hist["val_by_output"][i].append(per_out_mse_sums[i] / max(1, batches))
                hist["val_total_by_output"][i].append(per_out_total_sums[i] / max(1, batches))

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{epoch+1:03d}/{cfg.epochs}] train_total={train_total:.6g}  "
              f"val_total={val_total:.6g}  (val MSE v={hist['val_mse_v'][-1]:.6g})")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return hist


# ==================== Plotting ====================

def plot_training_curves(hist: Dict[str, List[float]], out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(hist["train_total"]) + 1)

    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.xlabel("epoch"); plt.ylabel("total loss"); plt.title("Total Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_linear.png"); plt.close()

    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("total loss (log)"); plt.title("Total Loss (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_log.png"); plt.close()

    names = ["vE", "vN", "vD"]
    plt.figure()
    for i in range(3):
        plt.plot(epochs, hist["val_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val MSE per output"); plt.title("Validation MSE per output (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_mse_per_output.png"); plt.close()

    plt.figure()
    for i in range(3):
        plt.plot(epochs, hist["val_total_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val TOTAL per output"); plt.title("Validation TOTAL per output (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_total_per_output.png"); plt.close()


def save_history_csv(hist: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = {
        "epoch": list(range(1, len(hist["train_total"]) + 1)),
        "train_total": hist["train_total"],
        "val_total":   hist["val_total"],
        "val_mse_v":   hist["val_mse_v"],
    }
    for i, name in enumerate(["vE", "vN", "vD"]):
        cols[f"val_mse_{name}"] = hist["val_by_output"][i]
        cols[f"val_total_{name}"] = hist["val_total_by_output"][i]
    pd.DataFrame(cols).to_csv(out_path, index=False)


# ==================== Data I/O / normalization ====================

IN_COLS     = ["ax","ay","az","qw","qx","qy","qz","tau_x","tau_y","tau_n"]
OUT_COLS    = ["vE","vN","vD"]
AUX_W_COLS  = ["w_est_x","w_est_y","w_est_z"]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Normalizer:
    """Standardize inputs; standardize velocity outputs."""
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
                     out_cols=OUT_COLS,
                     aux_w_cols=AUX_W_COLS):
    df = pd.read_csv(path)
    miss_in  = [c for c in in_cols  if c not in df.columns]
    miss_out = [c for c in out_cols if c not in df.columns]
    if miss_in or miss_out:
        raise ValueError(f"{path}: missing columns. Missing inputs={miss_in}, missing outputs={miss_out}")

    x = df[in_cols].to_numpy(np.float32)
    y = df[out_cols].to_numpy(np.float32)

    # normalize quaternion input for safety
    q = x[:, 3:7]
    qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    x[:, 3:7] = q / qn

    # auxiliary ω_b from q-observer (raw, NOT normalized)
    if all(c in df.columns for c in aux_w_cols):
        w = df[aux_w_cols].to_numpy(np.float32)
    else:
        w = np.zeros((x.shape[0], 3), dtype=np.float32)

    return x, y, w

def make_sequences3(x: np.ndarray, y: np.ndarray, w: np.ndarray, T: int):
    N = x.shape[0]; B = N // T
    if B == 0:
        raise ValueError(f"Sequence length {T} longer than data ({N}).")
    x = x[:B*T].reshape(B, T, -1)
    y = y[:B*T].reshape(B, T, -1)
    w = w[:B*T].reshape(B, T, -1)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

def build_loaders(train_csv: str, val_csv: str, norm: Normalizer, seq_len: int,
                  batch_size: int = 64):
    xtr, ytr, wtr = read_csv_columns(train_csv)
    xva, yva, wva = read_csv_columns(val_csv)
    xtr = norm.x(xtr); xva = norm.x(xva)
    ytr = norm.y(ytr); yva = norm.y(yva)
    xtr_t, ytr_t, wtr_t = make_sequences3(xtr, ytr, wtr, seq_len)
    xva_t, yva_t, wva_t = make_sequences3(xva, yva, wva, seq_len)
    tr_ds = SeqDataset(xtr_t, ytr_t, wtr_t)
    va_ds = SeqDataset(xva_t, yva_t, wva_t)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return tr_dl, va_dl


# ==================== Export helpers ====================

def _export_torchscript(model, out_path: str, seq_len: int = 300):
    model_cpu = model.eval().to("cpu")
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception as e:
        print(f"[export] script() failed → tracing instead: {e}")
        example = torch.zeros(1, seq_len, 10, dtype=torch.float32)
        scripted = torch.jit.trace(model_cpu, example, strict=False)
    scripted.save(out_path)
    print(f"[export] wrote TorchScript: {out_path}")

def export_ensemble_torchscript(ModelClass, member_pths: list, out_path: str, seq_len: int):
    class Ensemble(nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = nn.ModuleList(members)
        def forward(self, x):
            ys = [m(x) for m in self.members]   # each [B,T,3]
            y  = torch.stack(ys, dim=0).mean(0) # [B,T,3]
            return y
    members = []
    for pth in member_pths:
        m = ModelClass()
        ckpt = torch.load(pth, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        m.load_state_dict(state, strict=True)
        m.eval().to("cpu")
        members.append(m)
    ens = Ensemble(members).eval().to("cpu")
    try:
        scripted = torch.jit.script(ens)
    except Exception as e:
        print(f"[export ensemble] script() failed → tracing instead: {e}")
        example = torch.zeros(1, seq_len, 10, dtype=torch.float32)
        scripted = torch.jit.trace(ens, example, strict=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"[export ensemble] wrote TorchScript: {out_path}")


# ==================== Test evaluation helpers (NEW) ====================

def build_test_loader(test_csv: str, norm: Normalizer, seq_len: int, batch_size: int = 64):
    xt, yt, wt = read_csv_columns(test_csv)
    xt = norm.x(xt)
    yt = norm.y(yt)
    xt_t, yt_t, wt_t = make_sequences3(xt, yt, wt, seq_len)
    ds = SeqDataset(xt_t, yt_t, wt_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return dl

@torch.no_grad()
def evaluate_on_loader(model: nn.Module, loader, device: str = "cpu"):
    model.eval()
    preds = []
    gts = []
    for xb, yb, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        preds.append(yhat.cpu().numpy())
        gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # [B,T,3]
    gts   = np.concatenate(gts,   axis=0)  # [B,T,3]
    preds = preds.reshape(-1, 3)
    gts   = gts.reshape(-1, 3)
    return preds, gts

def evaluate_members_on_loader(members: List[nn.Module], loader, device: str = "cpu"):
    """Return stacked predictions from each member on the same loader.
    Outputs:
      preds: [M, N, 3] where N=B*T flattened; gts: [N,3]
    """
    preds_list = []
    gts_ref = None
    for i, m in enumerate(members):
        m = m.eval().to(device)
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb, wb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = m(xb)
                preds.append(yhat.cpu().numpy())
                if i == 0:
                    gts.append(yb.cpu().numpy())
        preds = np.concatenate(preds, axis=0).reshape(-1, 3)
        preds_list.append(preds)
        if i == 0:
            gts_ref = np.concatenate(gts, axis=0).reshape(-1, 3)
    preds_stack = np.stack(preds_list, axis=0)  # [M,N,3]
    return preds_stack, gts_ref

def compute_metrics(y_pred: np.ndarray, y_gt: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_gt
    mse = np.mean(err**2, axis=0)         # [3]
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)
    out = {
        "MSE_vE": float(mse[0]), "MSE_vN": float(mse[1]), "MSE_vD": float(mse[2]),
        "RMSE_vE": float(rmse[0]), "RMSE_vN": float(rmse[1]), "RMSE_vD": float(rmse[2]),
        "MAE_vE": float(mae[0]), "MAE_vN": float(mae[1]), "MAE_vD": float(mae[2]),
        "MSE_total": float(np.mean(mse)),
        "RMSE_total": float(np.sqrt(np.mean(mse))),
        "MAE_total": float(np.mean(mae)),
    }
    return out


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

def save_test_outputs_with_uncertainty(out_dir: str,
                                       pred_mean_phy: np.ndarray,
                                       gt_phy: np.ndarray,
                                       metrics: Dict[str, float],
                                       pred_std_phy: np.ndarray,
                                       ci95_phy: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "vE_gt": gt_phy[:,0], "vN_gt": gt_phy[:,1], "vD_gt": gt_phy[:,2],
        "vE_pred": pred_mean_phy[:,0], "vN_pred": pred_mean_phy[:,1], "vD_pred": pred_mean_phy[:,2],
        "vE_std": pred_std_phy[:,0], "vN_std": pred_std_phy[:,1], "vD_std": pred_std_phy[:,2],
        "vE_ci95": ci95_phy[:,0], "vN_ci95": ci95_phy[:,1], "vD_ci95": ci95_phy[:,2],
    })
    csv_path = os.path.join(out_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[test] wrote predictions (with std & 95% CI): {csv_path}")
    print(f"[test] wrote metrics:                         {os.path.join(out_dir, 'test_metrics.json')}")


# ==================== CLI / main ====================

def main():
    parser = argparse.ArgumentParser(description="Train velocity-only NN (no gyro) with RAN-based physics-informed loss (GRU).")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val",   required=True, help="Path to validation CSV")
    parser.add_argument("--test",  required=True, help="Path to test CSV")  # NEW
    parser.add_argument("--out",   required=True, help="Output directory (models, plots)")
    parser.add_argument("--seq",   type=int, default=400, help="Sequence length T")
    parser.add_argument("--epochs",type=int, default=10000, help="Training epochs")
    parser.add_argument("--batch", type=int, default=256,  help="Batch size")
    parser.add_argument("--lr",    type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--wd",    type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gpu",   action="store_true", help="Use CUDA if available")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of models to train")
    parser.add_argument("--dropout",  type=float, default=0.1, help="GRU inter-layer dropout p")
    parser.add_argument("--qwidth",   type=int, default=128, help="GRU hidden size")
    parser.add_argument("--dt",       type=float, default=0.01, help="Sample period (s)")
    parser.add_argument("--norm_json", type=str, default="", help="Normalization JSON path")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed (member 0). Members use seed+i")
    parser.add_argument("--w_phys",   type=float, default=1.0, help="Physics loss weight")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: CPU")

    # Load once to compute/fallback norms
    def _read_cols(path):
        df = pd.read_csv(path)
        x = df[IN_COLS].to_numpy(np.float32)
        y = df[OUT_COLS].to_numpy(np.float32)
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

    # Prepare de-normalization tensors for physics loss
    x_mean_t = torch.tensor(norm.x_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    x_std_t  = torch.tensor(norm.x_std,  dtype=torch.float32, device=device).view(1, 1, -1)

    # Data loaders (with w_est aux)
    def build_loaders_local(train_csv, val_csv, seq_len, batch_size):
        xtr, ytr, wtr = read_csv_columns(train_csv)
        xva, yva, wva = read_csv_columns(val_csv)
        xtr = norm.x(xtr); xva = norm.x(xva)
        ytr = norm.y(ytr); yva = norm.y(yva)
        xtr_t, ytr_t, wtr_t = make_sequences3(xtr, ytr, wtr, seq_len)
        xva_t, yva_t, wva_t = make_sequences3(xva, yva, wva, seq_len)
        tr = torch.utils.data.DataLoader(SeqDataset(xtr_t, ytr_t, wtr_t), batch_size=batch_size, shuffle=True)
        va = torch.utils.data.DataLoader(SeqDataset(xva_t, yva_t, wva_t), batch_size=batch_size, shuffle=False)
        return tr, va

    train_loader, val_loader = build_loaders_local(args.train, args.val, args.seq, args.batch)

    # Build test loader (normalized identically)
    test_loader = build_test_loader(args.test, norm, args.seq, args.batch)

    # Train ensemble
    ts_collect_dir = os.path.join(args.out, "ts")
    os.makedirs(ts_collect_dir, exist_ok=True)
    all_hists, member_ckpts = [], []

    for m in range(args.ensemble):
        print("\n" + "="*90)
        print(f"Ensemble member {m+1}/{args.ensemble}")
        print("="*90)
        set_seed(args.seed + m)

        model = VelNetV9(hidden=args.qwidth, num_layers=3, dropout_p=args.dropout).to(device)
        cfg = TrainCfg(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
                       dt=args.dt, dropout_p=args.dropout, qwidth=args.qwidth,
                       device=device, loss_w=LossWeights(mse=1.0, trans=args.w_phys),
                       x_mean=x_mean_t, x_std=x_std_t)

        member_dir = os.path.join(args.out, f"member_{m:02d}")
        os.makedirs(member_dir, exist_ok=True)
        with open(os.path.join(member_dir, "config.json"), "w") as f:
            json.dump({
                "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                "dt": cfg.dt, "dropout_p": cfg.dropout_p, "hidden": cfg.qwidth,
                "device": cfg.device, "seed": args.seed + m, "w_phys": args.w_phys
            }, f, indent=2)

        hist = train_one_model(model, train_loader, val_loader, cfg)
        all_hists.append(hist)

        best_pth = os.path.join(member_dir, "model.pth")
        torch.save({"state_dict": model.state_dict()}, best_pth)
        member_ckpts.append(best_pth)

        with open(os.path.join(member_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        plot_training_curves(hist, out_dir=member_dir)

        # Export TorchScript for member
        model_cpu = model.eval().to("cpu")
        member_ts_dir = os.path.join(member_dir, "ts")
        os.makedirs(member_ts_dir, exist_ok=True)
        member_pt_path = os.path.join(member_ts_dir, f"member_{m:02d}.pt")
        example = torch.zeros(1, args.seq, 10, dtype=torch.float32)
        with torch.no_grad():
            scripted = None
            try:
                scripted = torch.jit.script(model_cpu)
                try:
                    scripted.save(member_pt_path)
                    print(f"[export] wrote TorchScript (script): {member_pt_path}")
                except Exception as e_save:
                    print(f"[export] script.save() failed → fallback to trace: {e_save}")
                    scripted = None
            except Exception as e_script:
                print(f"[export] torch.jit.script failed → trace: {e_script}")
            if scripted is None:
                traced = torch.jit.trace(model_cpu, example, strict=False)
                traced.save(member_pt_path)
                print(f"[export] wrote TorchScript (trace):  {member_pt_path}")
        shutil.copyfile(member_pt_path, os.path.join(ts_collect_dir, f"member_{m:02d}.pt"))

        # ==== Per-member TEST evaluation (NEW) ====
        print("[test] Evaluating member on test set…")
        model_eval = model.eval().to(device)
        preds_norm_m, gts_norm_m = evaluate_on_loader(model_eval, test_loader, device=device)
        preds_phy_m = norm.y_inv(preds_norm_m)
        gts_phy_m   = norm.y_inv(gts_norm_m)
        metrics_m = compute_metrics(preds_phy_m, gts_phy_m)
        save_test_outputs(member_dir, preds_phy_m, gts_phy_m, metrics_m)

    # Build Python ensemble for evaluation/export
    class Ensemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)
        def forward(self, x):
            ys = [m(x) for m in self.members]           # each [B,T,3]
            y  = torch.stack(ys, dim=0).mean(0)         # [B,T,3]
            return y

    members = []
    for pth in member_ckpts:
        if not os.path.isfile(pth):
            print(f"[warn] missing checkpoint: {pth} (skipping)")
            continue
        m = VelNetV9(hidden=args.qwidth, num_layers=3, dropout_p=args.dropout).eval().to("cpu")
        ckpt = torch.load(pth, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        m.load_state_dict(state, strict=True)
        members.append(m)

    if len(members) >= 1:
        # Export ensemble TorchScript
        ens = Ensemble(members).eval().to("cpu")
        ens_pt_path = os.path.join(args.out, "ensemble_stateless.pt")
        example = torch.zeros(1, args.seq, 10, dtype=torch.float32)
        with torch.no_grad():
            scripted = None
            try:
                scripted = torch.jit.script(ens)
                try:
                    scripted.save(ens_pt_path)
                    print(f"[export ensemble] wrote (script): {ens_pt_path}")
                except Exception as e_save:
                    print(f"[export ensemble] script.save() failed → trace: {e_save}")
                    scripted = None
            except Exception as e_script:
                print(f"[export ensemble] script failed → trace: {e_script}")
            if scripted is None:
                traced = torch.jit.trace(ens, example, strict=False)
                traced.save(ens_pt_path)
                print(f"[export ensemble] wrote (trace):  {ens_pt_path}")

        # ============ TEST EVALUATION ============
        # Evaluate ensemble on test set with per-timestep uncertainty (std and 95% CI across members)
        print("\n[test] Evaluating ensemble on test set…")
        device_eval = device

        # Collect per-member predictions
        preds_norm_stack, gts_norm = evaluate_members_on_loader(members, test_loader, device=device_eval)  # [M,N,3], [N,3]

        # Denormalize to physical units
        if (norm.y_mean is not None) and (norm.y_std is not None):
            y_mean = norm.y_mean.reshape(1, 1, 3)
            y_std  = norm.y_std.reshape(1, 1, 3)
            preds_phy_stack = preds_norm_stack * y_std + y_mean   # [M,N,3]
            gts_phy = gts_norm * norm.y_std + norm.y_mean         # [N,3]
        else:
            preds_phy_stack = preds_norm_stack
            gts_phy = gts_norm

        # Ensemble mean & std (per timestep)
        preds_phy_mean = preds_phy_stack.mean(axis=0)           # [N,3]
        preds_phy_std  = preds_phy_stack.std(axis=0, ddof=0)    # [N,3]
        ci95 = 1.96 * preds_phy_std

        metrics = compute_metrics(preds_phy_mean, gts_phy)
        # Summary uncertainty in metrics (averaged over time)
        metrics.update({
            "STD_vE_mean": float(np.mean(preds_phy_std[:,0])),
            "STD_vN_mean": float(np.mean(preds_phy_std[:,1])),
            "STD_vD_mean": float(np.mean(preds_phy_std[:,2])),
        })

        save_test_outputs_with_uncertainty(
            args.out,
            pred_mean_phy=preds_phy_mean,
            gt_phy=gts_phy,
            metrics=metrics,
            pred_std_phy=preds_phy_std,
            ci95_phy=ci95,
        )


if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# nn_observer_v9.py
# Velocity-only NN (END frame) with RAN-like physics-informed translational loss.
# Architecture: regular GRU over full input vector (10 channels), ω_b is passed only to the loss.

# Inputs (10):  [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]   (no gyro)
# Aux (3):      [w_est_x, w_est_y, w_est_z]  (BODY rates from q-Obs; used ONLY in physics loss)
# Outputs (3):  [vE, vN, vD]

# IMPORTANT:
# - Training uses *normalized* inputs, but the physics loss uses **de-normalized**
#   accel/quat/tau signals. We pass x_mean/x_std into the loss to de-normalize
#   each batch, and pass ω_b (w_est) separately as raw.

# Run example:
# python3 scripts/nn_observer_v9.py \
#   --train data/nn_dataset_v9_X_C0/train.csv \
#   --val   data/nn_dataset_v9_X_C0/val.csv \
#   --out   data/nn_model_v9_ens4 \
#   --seq 300 --epochs 200 --gpu --ensemble 4 \
#   --norm_json data/nn_dataset_v9_X_C0/norm_stats.json
# """

# import os
# import math
# import json
# import argparse
# import random
# from dataclasses import dataclass, field
# from typing import Dict, Tuple, List

# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # headless plotting
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# import shutil

# # ---- RAN physics residual (uses custom rotation; τ in BODY; ω_b from w_est) ----
# from ran_model_loss import ran_translational_loss_v9


# # ==================== GRU model (inputs 10 -> outputs 3) ====================

# class VelNetV9(nn.Module):
#     """
#     Regular GRU over full input:
#       x[t] = [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]  (10-D)
#     Output per step: [vE, vN, vD]
#     """
#     def __init__(self,
#                  hidden: int = 64,
#                  num_layers: int = 3,
#                  dropout_p: float = 0.2,
#                  use_layernorm: bool = True,
#                  leaky: float = 0.01):
#         super().__init__()
#         self.input_dim   = 10
#         self.hidden_GRU  = hidden
#         self.hidden_head = (3 * hidden) // 4
#         self.num_layers  = num_layers

#         self.gru = nn.GRU(
#             input_size=self.input_dim,
#             hidden_size=self.hidden_GRU,
#             num_layers=self.num_layers,
#             batch_first=True
#             #dropout=(dropout_p if self.num_layers > 1 else 0.0),
#         )

#         self.post_gru_norm = nn.LayerNorm(self.hidden_GRU) if use_layernorm else nn.Identity()
#         self.act = nn.LeakyReLU(leaky, inplace=True) if leaky > 0.0 else nn.ReLU(inplace=True)

#         self.head = nn.Sequential(
#             nn.Linear(self.hidden_GRU, self.hidden_head),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(self.hidden_head, 3),
#         )

#     def __repr__(self) -> str:
#         return (f"{self.__class__.__name__}(GRU in=10, hidden_GRU={self.hidden_GRU}, "
#                 f"hidden_head={self.hidden_head}, layers={self.num_layers}) -> 3")

#     def forward(self, x_bt10: torch.Tensor) -> torch.Tensor:
#         z, _ = self.gru(x_bt10)      # [B,T,H]
#         z = self.post_gru_norm(z)
#         z = self.act(z)
#         y = self.head(z)             # [B,T,3] -> [vE,vN,vD]
#         return y


# # ==================== Losses ====================

# @dataclass
# class LossWeights:
#     mse: float = 1.0
#     trans: float = 1.0

# def total_loss(pred_bt3: torch.Tensor,
#                gt_bt3: torch.Tensor,
#                x_norm_bt10: torch.Tensor,
#                x_mean_t: torch.Tensor,
#                x_std_t: torch.Tensor,
#                w_est_bt3: torch.Tensor,
#                dt: float,
#                w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
#     """
#     Combined supervised + physics loss.
#     - pred_bt3: [B,T,3] model outputs in END frame
#     - gt_bt3  : [B,T,3] ground-truth END velocity
#     - x_norm_bt10: [B,T,10] normalized inputs
#     - x_mean_t/x_std_t: [1,1,10] tensors (on device) for de-normalization
#     - w_est_bt3: [B,T,3] BODY rates from q-Obs (raw, not normalized)
#     """
#     L_mse = F.mse_loss(pred_bt3, gt_bt3)

#     # de-normalize inputs for physics loss
#     x_raw = x_norm_bt10 * x_std_t + x_mean_t  # [B,T,10] real units

#     # physics term (τ in BODY, custom R_nb, uses ω_b)
#     L_phys = ran_translational_loss_v9(pred_bt3, x_raw, dt, w_est_bt3)

#     L = w.mse * L_mse + w.trans * L_phys
#     return L, {
#         "total": float(L.detach().cpu()),
#         "mse_v": float(L_mse.detach().cpu()),
#         "phys_trans": float(L_phys.detach().cpu()),
#     }


# # ==================== Training scaffold ====================

# @dataclass
# class TrainCfg:
#     epochs: int = 50
#     lr: float = 1e-3
#     weight_decay: float = 0.0
#     dt: float = 0.01
#     dropout_p: float = 0.2
#     qwidth: int = 64
#     device: str = "cuda"
#     loss_w: "LossWeights" = field(default_factory=LossWeights)
#     x_mean: torch.Tensor = None  # [1,1,10]
#     x_std:  torch.Tensor = None  # [1,1,10]

# class SeqDataset(torch.utils.data.Dataset):
#     """Return (x_bt10, y_bt3, w_bt3) per item."""
#     def __init__(self, x_bt10: torch.Tensor, y_bt3: torch.Tensor, w_bt3: torch.Tensor):
#         assert x_bt10.shape[:2] == y_bt3.shape[:2] == w_bt3.shape[:2]
#         self.x = x_bt10; self.y = y_bt3; self.w = w_bt3
#     def __len__(self): return self.x.shape[0]
#     def __getitem__(self, idx): return self.x[idx], self.y[idx], self.w[idx]

# def train_one_model(model: nn.Module,
#                     train_loader,
#                     val_loader,
#                     cfg: TrainCfg) -> Dict[str, List[float]]:
#     model.to(cfg.device)
#     opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

#     hist = {
#         "train_total": [],
#         "val_total": [],
#         "val_mse_v": [],
#         "val_by_output": [[] for _ in range(3)],
#         "val_total_by_output": [[] for _ in range(3)],
#     }

#     best_val = float("inf")
#     best_state = None

#     for epoch in range(cfg.epochs):
#         # ---------------- Train ----------------
#         model.train()
#         train_sum = 0.0
#         for xb, yb, wb in train_loader:
#             xb = xb.to(cfg.device)         # [B,T,10] normalized
#             yb = yb.to(cfg.device)         # [B,T,3]
#             wb = wb.to(cfg.device)         # [B,T,3] raw ω_b
#             pred = model(xb)               # [B,T,3]
#             L, _ = total_loss(pred, yb, xb, cfg.x_mean.to(cfg.device), cfg.x_std.to(cfg.device),
#                               wb, cfg.dt, cfg.loss_w)
#             opt.zero_grad(set_to_none=True)
#             L.backward()
#             opt.step()
#             train_sum += float(L.detach().cpu())
#         train_total = train_sum / max(1, len(train_loader))
#         hist["train_total"].append(train_total)

#         # ---------------- Validation ----------------
#         model.eval()
#         with torch.no_grad():
#             tot_sum = 0.0
#             mse_v_sum = 0.0
#             per_out_mse_sums = [0.0] * 3
#             per_out_total_sums = [0.0] * 3
#             batches = 0

#             for xb, yb, wb in val_loader:
#                 xb = xb.to(cfg.device)
#                 yb = yb.to(cfg.device)
#                 wb = wb.to(cfg.device)
#                 pred = model(xb)
#                 L, logs = total_loss(pred, yb, xb, cfg.x_mean.to(cfg.device), cfg.x_std.to(cfg.device),
#                                      wb, cfg.dt, cfg.loss_w)
#                 tot_sum += logs["total"]
#                 mse_v_sum += logs["mse_v"]

#                 mse_vec = torch.mean((pred - yb).pow(2), dim=(0, 1))  # [3]
#                 add_v = cfg.loss_w.trans * logs["phys_trans"]
#                 per_total = (cfg.loss_w.mse * mse_vec).cpu().numpy()
#                 per_total[0:3] += add_v

#                 for i in range(3):
#                     per_out_mse_sums[i] += float(mse_vec[i].cpu())
#                     per_out_total_sums[i] += float(per_total[i])
#                 batches += 1

#             val_total = tot_sum / max(1, len(val_loader))
#             hist["val_total"].append(val_total)
#             hist["val_mse_v"].append(mse_v_sum / max(1, len(val_loader)))
#             for i in range(3):
#                 hist["val_by_output"][i].append(per_out_mse_sums[i] / max(1, batches))
#                 hist["val_total_by_output"][i].append(per_out_total_sums[i] / max(1, batches))

#         if val_total < best_val:
#             best_val = val_total
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

#         print(f"[{epoch+1:03d}/{cfg.epochs}] train_total={train_total:.6g}  "
#               f"val_total={val_total:.6g}  (val MSE v={hist['val_mse_v'][-1]:.6g})")

#     if best_state is not None:
#         model.load_state_dict(best_state, strict=True)

#     return hist


# # ==================== Plotting ====================

# def plot_training_curves(hist: Dict[str, List[float]], out_dir: str = "."):
#     os.makedirs(out_dir, exist_ok=True)
#     epochs = range(1, len(hist["train_total"]) + 1)

#     plt.figure()
#     plt.plot(epochs, hist["train_total"], label="train total")
#     plt.plot(epochs, hist["val_total"], label="val total")
#     plt.xlabel("epoch"); plt.ylabel("total loss"); plt.title("Total Loss")
#     plt.legend(); plt.grid(True, alpha=0.3)
#     plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_linear.png"); plt.close()

#     plt.figure()
#     plt.plot(epochs, hist["train_total"], label="train total")
#     plt.plot(epochs, hist["val_total"], label="val total")
#     plt.yscale("log")
#     plt.xlabel("epoch"); plt.ylabel("total loss (log)"); plt.title("Total Loss (log)")
#     plt.legend(); plt.grid(True, which="both", alpha=0.3)
#     plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_log.png"); plt.close()

#     names = ["vE", "vN", "vD"]
#     plt.figure()
#     for i in range(3):
#         plt.plot(epochs, hist["val_by_output"][i], label=names[i])
#     plt.yscale("log")
#     plt.xlabel("epoch"); plt.ylabel("val MSE per output"); plt.title("Validation MSE per output (log)")
#     plt.legend(); plt.grid(True, which="both", alpha=0.3)
#     plt.tight_layout(); plt.savefig(f"{out_dir}/val_mse_per_output.png"); plt.close()

#     plt.figure()
#     for i in range(3):
#         plt.plot(epochs, hist["val_total_by_output"][i], label=names[i])
#     plt.yscale("log")
#     plt.xlabel("epoch"); plt.ylabel("val TOTAL per output"); plt.title("Validation TOTAL per output (log)")
#     plt.legend(); plt.grid(True, which="both", alpha=0.3)
#     plt.tight_layout(); plt.savefig(f"{out_dir}/val_total_per_output.png"); plt.close()


# def save_history_csv(hist: dict, out_path: str):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     cols = {
#         "epoch": list(range(1, len(hist["train_total"]) + 1)),
#         "train_total": hist["train_total"],
#         "val_total":   hist["val_total"],
#         "val_mse_v":   hist["val_mse_v"],
#     }
#     for i, name in enumerate(["vE", "vN", "vD"]):
#         cols[f"val_mse_{name}"] = hist["val_by_output"][i]
#         cols[f"val_total_{name}"] = hist["val_total_by_output"][i]
#     pd.DataFrame(cols).to_csv(out_path, index=False)


# # ==================== Data I/O / normalization ====================

# IN_COLS     = ["ax","ay","az","qw","qx","qy","qz","tau_x","tau_y","tau_n"]
# OUT_COLS    = ["vE","vN","vD"]
# AUX_W_COLS  = ["w_est_x","w_est_y","w_est_z"]

# def set_seed(seed: int = 42):
#     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# class Normalizer:
#     """Standardize inputs; standardize velocity outputs."""
#     def __init__(self, x_mean, x_std, y_mean=None, y_std=None):
#         self.x_mean = np.asarray(x_mean, dtype=np.float32)
#         self.x_std  = np.asarray(x_std,  dtype=np.float32)
#         self.y_mean = None if y_mean is None else np.asarray(y_mean, dtype=np.float32)
#         self.y_std  = None if y_std  is None else np.asarray(y_std,  dtype=np.float32)
#         self.x_std[self.x_std == 0] = 1.0
#         if self.y_std is not None:
#             self.y_std[self.y_std == 0] = 1.0
#     def x(self, x_np: np.ndarray) -> np.ndarray:
#         return (x_np - self.x_mean) / self.x_std
#     def y(self, y_np: np.ndarray) -> np.ndarray:
#         if self.y_mean is None or self.y_std is None:
#             return y_np
#         return (y_np - self.y_mean) / self.y_std

# def load_norm(norm_json_path: str, x_train: np.ndarray = None, y_train: np.ndarray = None) -> Normalizer:
#     if norm_json_path and os.path.isfile(norm_json_path):
#         with open(norm_json_path, "r") as f:
#             d = json.load(f)
#         return Normalizer(d["x_mean"], d["x_std"], d.get("y_mean"), d.get("y_std"))
#     if x_train is None:
#         raise ValueError("norm_json not found and no training data provided to compute norms.")
#     x_mean = x_train.mean(axis=0).tolist()
#     x_std  = x_train.std(axis=0).tolist()
#     y_mean = y_train.mean(axis=0).tolist() if y_train is not None else None
#     y_std  = y_train.std(axis=0).tolist()  if y_train is not None else None
#     return Normalizer(x_mean, x_std, y_mean, y_std)

# def read_csv_columns(path: str,
#                      in_cols=IN_COLS,
#                      out_cols=OUT_COLS,
#                      aux_w_cols=AUX_W_COLS):
#     df = pd.read_csv(path)
#     miss_in  = [c for c in in_cols  if c not in df.columns]
#     miss_out = [c for c in out_cols if c not in df.columns]
#     if miss_in or miss_out:
#         raise ValueError(f"{path}: missing columns. Missing inputs={miss_in}, missing outputs={miss_out}")

#     x = df[in_cols].to_numpy(np.float32)
#     y = df[out_cols].to_numpy(np.float32)

#     # normalize quaternion input for safety
#     q = x[:, 3:7]
#     qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
#     x[:, 3:7] = q / qn

#     # auxiliary ω_b from q-observer (raw, NOT normalized)
#     if all(c in df.columns for c in aux_w_cols):
#         w = df[aux_w_cols].to_numpy(np.float32)
#     else:
#         w = np.zeros((x.shape[0], 3), dtype=np.float32)

#     return x, y, w

# def make_sequences3(x: np.ndarray, y: np.ndarray, w: np.ndarray, T: int):
#     N = x.shape[0]; B = N // T
#     if B == 0:
#         raise ValueError(f"Sequence length {T} longer than data ({N}).")
#     x = x[:B*T].reshape(B, T, -1)
#     y = y[:B*T].reshape(B, T, -1)
#     w = w[:B*T].reshape(B, T, -1)
#     return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

# def build_loaders(train_csv: str, val_csv: str, norm: Normalizer, seq_len: int,
#                   batch_size: int = 64):
#     xtr, ytr, wtr = read_csv_columns(train_csv)
#     xva, yva, wva = read_csv_columns(val_csv)
#     xtr = norm.x(xtr); xva = norm.x(xva)
#     ytr = norm.y(ytr); yva = norm.y(yva)
#     xtr_t, ytr_t, wtr_t = make_sequences3(xtr, ytr, wtr, seq_len)
#     xva_t, yva_t, wva_t = make_sequences3(xva, yva, wva, seq_len)
#     tr_ds = SeqDataset(xtr_t, ytr_t, wtr_t)
#     va_ds = SeqDataset(xva_t, yva_t, wva_t)
#     tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
#     va_dl = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)
#     return tr_dl, va_dl


# # ==================== Export helpers ====================

# def _export_torchscript(model, out_path: str, seq_len: int = 300):
#     model_cpu = model.eval().to("cpu")
#     try:
#         scripted = torch.jit.script(model_cpu)
#     except Exception as e:
#         print(f"[export] script() failed → tracing instead: {e}")
#         example = torch.zeros(1, seq_len, 10, dtype=torch.float32)
#         scripted = torch.jit.trace(model_cpu, example, strict=False)
#     scripted.save(out_path)
#     print(f"[export] wrote TorchScript: {out_path}")

# def export_ensemble_torchscript(ModelClass, member_pths: list, out_path: str, seq_len: int):
#     class Ensemble(nn.Module):
#         def __init__(self, members):
#             super().__init__()
#             self.members = nn.ModuleList(members)
#         def forward(self, x):
#             ys = [m(x) for m in self.members]   # each [B,T,3]
#             y  = torch.stack(ys, dim=0).mean(0) # [B,T,3]
#             return y
#     members = []
#     for pth in member_pths:
#         m = ModelClass()
#         ckpt = torch.load(pth, map_location="cpu")
#         state = ckpt.get("state_dict", ckpt)
#         m.load_state_dict(state, strict=True)
#         m.eval().to("cpu")
#         members.append(m)
#     ens = Ensemble(members).eval().to("cpu")
#     try:
#         scripted = torch.jit.script(ens)
#     except Exception as e:
#         print(f"[export ensemble] script() failed → tracing instead: {e}")
#         example = torch.zeros(1, seq_len, 10, dtype=torch.float32)
#         scripted = torch.jit.trace(ens, example, strict=False)
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     scripted.save(out_path)
#     print(f"[export ensemble] wrote TorchScript: {out_path}")


# # ==================== CLI / main ====================

# def main():
#     parser = argparse.ArgumentParser(description="Train velocity-only NN (no gyro) with RAN-based physics-informed loss (GRU).")
#     parser.add_argument("--train", required=True, help="Path to training CSV")
#     parser.add_argument("--val",   required=True, help="Path to validation CSV")
#     parser.add_argument("--out",   required=True, help="Output directory (models, plots)")
#     parser.add_argument("--seq",   type=int, default=400, help="Sequence length T")
#     parser.add_argument("--epochs",type=int, default=10000, help="Training epochs")
#     parser.add_argument("--batch", type=int, default=256,  help="Batch size")
#     parser.add_argument("--lr",    type=float, default=1e-4, help="Learning rate") #Try 1e-5
#     parser.add_argument("--wd",    type=float, default=0.0, help="Weight decay")
#     parser.add_argument("--gpu",   action="store_true", help="Use CUDA if available")
#     parser.add_argument("--ensemble", type=int, default=1, help="Number of models to train")
#     parser.add_argument("--dropout",  type=float, default=0.1, help="GRU inter-layer dropout p")
#     parser.add_argument("--qwidth",   type=int, default=128, help="GRU hidden size")
#     parser.add_argument("--dt",       type=float, default=0.01, help="Sample period (s)")
#     parser.add_argument("--norm_json", type=str, default="", help="Normalization JSON path")
#     parser.add_argument("--seed",     type=int, default=42, help="Random seed (member 0). Members use seed+i")
#     parser.add_argument("--w_phys",   type=float, default=1.0, help="Physics loss weight")
#     args = parser.parse_args()

#     os.makedirs(args.out, exist_ok=True)
#     set_seed(args.seed)

#     device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
#     if device == "cuda":
#         torch.backends.cudnn.benchmark = True
#         print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
#     else:
#         print("Device: CPU")

#     # Load once to compute/fallback norms
#     def _read_cols(path):
#         df = pd.read_csv(path)
#         x = df[IN_COLS].to_numpy(np.float32)
#         y = df[OUT_COLS].to_numpy(np.float32)
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

#     # Prepare de-normalization tensors for physics loss
#     x_mean_t = torch.tensor(norm.x_mean, dtype=torch.float32, device=device).view(1, 1, -1)
#     x_std_t  = torch.tensor(norm.x_std,  dtype=torch.float32, device=device).view(1, 1, -1)

#     # Data loaders (with w_est aux)
#     def build_loaders_local(train_csv, val_csv, seq_len, batch_size):
#         xtr, ytr, wtr = read_csv_columns(train_csv)
#         xva, yva, wva = read_csv_columns(val_csv)
#         xtr = norm.x(xtr); xva = norm.x(xva)
#         ytr = norm.y(ytr); yva = norm.y(yva)
#         xtr_t, ytr_t, wtr_t = make_sequences3(xtr, ytr, wtr, seq_len)
#         xva_t, yva_t, wva_t = make_sequences3(xva, yva, wva, seq_len)
#         tr = torch.utils.data.DataLoader(SeqDataset(xtr_t, ytr_t, wtr_t), batch_size=batch_size, shuffle=True)
#         va = torch.utils.data.DataLoader(SeqDataset(xva_t, yva_t, wva_t), batch_size=batch_size, shuffle=False)
#         return tr, va

#     train_loader, val_loader = build_loaders_local(args.train, args.val, args.seq, args.batch)

#     # Train ensemble
#     ts_collect_dir = os.path.join(args.out, "ts")
#     os.makedirs(ts_collect_dir, exist_ok=True)
#     all_hists, member_ckpts = [], []

#     for m in range(args.ensemble):
#         print("\n" + "="*90)
#         print(f"Ensemble member {m+1}/{args.ensemble}")
#         print("="*90)
#         set_seed(args.seed + m)

#         model = VelNetV9(hidden=args.qwidth, num_layers=3, dropout_p=args.dropout).to(device)
#         cfg = TrainCfg(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
#                        dt=args.dt, dropout_p=args.dropout, qwidth=args.qwidth,
#                        device=device, loss_w=LossWeights(mse=1.0, trans=args.w_phys),
#                        x_mean=x_mean_t, x_std=x_std_t)

#         member_dir = os.path.join(args.out, f"member_{m:02d}")
#         os.makedirs(member_dir, exist_ok=True)
#         with open(os.path.join(member_dir, "config.json"), "w") as f:
#             json.dump({
#                 "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
#                 "dt": cfg.dt, "dropout_p": cfg.dropout_p, "hidden": cfg.qwidth,
#                 "device": cfg.device, "seed": args.seed + m, "w_phys": args.w_phys
#             }, f, indent=2)

#         hist = train_one_model(model, train_loader, val_loader, cfg)
#         all_hists.append(hist)

#         best_pth = os.path.join(member_dir, "model.pth")
#         torch.save({"state_dict": model.state_dict()}, best_pth)
#         member_ckpts.append(best_pth)

#         with open(os.path.join(member_dir, "history.json"), "w") as f:
#             json.dump(hist, f, indent=2)
#         plot_training_curves(hist, out_dir=member_dir)

#         # Export TorchScript for member
#         model_cpu = model.eval().to("cpu")
#         member_ts_dir = os.path.join(member_dir, "ts")
#         os.makedirs(member_ts_dir, exist_ok=True)
#         member_pt_path = os.path.join(member_ts_dir, f"member_{m:02d}.pt")
#         example = torch.zeros(1, args.seq, 10, dtype=torch.float32)
#         with torch.no_grad():
#             scripted = None
#             try:
#                 scripted = torch.jit.script(model_cpu)
#                 try:
#                     scripted.save(member_pt_path)
#                     print(f"[export] wrote TorchScript (script): {member_pt_path}")
#                 except Exception as e_save:
#                     print(f"[export] script.save() failed → fallback to trace: {e_save}")
#                     scripted = None
#             except Exception as e_script:
#                 print(f"[export] torch.jit.script failed → trace: {e_script}")
#             if scripted is None:
#                 traced = torch.jit.trace(model_cpu, example, strict=False)
#                 traced.save(member_pt_path)
#                 print(f"[export] wrote TorchScript (trace):  {member_pt_path}")
#         shutil.copyfile(member_pt_path, os.path.join(ts_collect_dir, f"member_{m:02d}.pt"))

#     # Optional: export ensemble-mean TorchScript
#     class Ensemble(torch.nn.Module):
#         def __init__(self, members):
#             super().__init__()
#             self.members = torch.nn.ModuleList(members)
#         def forward(self, x):
#             ys = [m(x) for m in self.members]           # each [B,T,3]
#             y  = torch.stack(ys, dim=0).mean(0)         # [B,T,3]
#             return y

#     members = []
#     for pth in member_ckpts:
#         if not os.path.isfile(pth):
#             print(f"[warn] missing checkpoint: {pth} (skipping)")
#             continue
#         m = VelNetV9(hidden=args.qwidth, num_layers=3, dropout_p=args.dropout).eval().to("cpu")
#         ckpt = torch.load(pth, map_location="cpu")
#         state = ckpt.get("state_dict", ckpt)
#         m.load_state_dict(state, strict=True)
#         members.append(m)

#     if len(members) >= 1:
#         ens = Ensemble(members).eval().to("cpu")
#         ens_pt_path = os.path.join(args.out, "ensemble_stateless.pt")
#         example = torch.zeros(1, args.seq, 10, dtype=torch.float32)
#         with torch.no_grad():
#             scripted = None
#             try:
#                 scripted = torch.jit.script(ens)
#                 try:
#                     scripted.save(ens_pt_path)
#                     print(f"[export ensemble] wrote (script): {ens_pt_path}")
#                 except Exception as e_save:
#                     print(f"[export ensemble] script.save() failed → trace: {e_save}")
#                     scripted = None
#             except Exception as e_script:
#                 print(f"[export ensemble] script failed → trace: {e_script}")
#             if scripted is None:
#                 traced = torch.jit.trace(ens, example, strict=False)
#                 traced.save(ens_pt_path)
#                 print(f"[export ensemble] wrote (trace):  {ens_pt_path}")

# if __name__ == "__main__":
#     main()

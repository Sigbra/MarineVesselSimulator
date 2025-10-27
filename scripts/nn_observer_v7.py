#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_observer_v7.py
Velocity-only NN (END frame) with physics-informed translational loss.
Inputs:  [ax, ay, az, qw, qx, qy, qz]         (no gyro)
Outputs: [vE, vN, vD]

Run example:
python3 scripts/nn_observer_v7.py \
  --train data/nn_dataset_v7_X_C0/train.csv \
  --val   data/nn_dataset_v7_X_C0/val.csv \
  --out   data/nn_model_v7_ens4 \
  --seq 200 --epochs 200 --gpu --ensemble 4 \
  --norm_json data/nn_dataset_v7_X_C0/norm_stats.json
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
import torch.nn.functional as F

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shutil

# ==================== Quaternion utilities ====================

def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=eps)

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    w,x,y,z = q.unbind(-1)
    return torch.stack([w,-x,-y,-z], dim=-1)

def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w1,x1,y1,z1 = q.unbind(-1)
    w2,x2,y2,z2 = r.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w,x,y,z], dim=-1)

def quat_rotate(q: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
    """Rotate Body vector v3 -> END via q (active). Shapes [...,4] and [...,3] -> [...,3]."""
    q = quat_normalize(q)
    qv = torch.cat([torch.zeros_like(v3[..., :1]), v3], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]

# ==================== Quaternion Conv1D blocks ====================

class QuaternionConv1d(nn.Module):
    """Quaternion 1D convolution. Input [B, 4*Cin, T] -> Output [B, 4*Cout, T]."""
    def __init__(self, in_qc: int, out_qc: int, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.ic, self.oc = in_qc, out_qc
        k = kernel_size
        self.Wr = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.Wi = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.Wj = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.Wk = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.bias = nn.Parameter(torch.zeros(out_qc)) if bias else None
        for W in (self.Wr, self.Wi, self.Wj, self.Wk):
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        self.stride, self.padding = stride, padding
    def _conv(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, W, bias=None, stride=self.stride, padding=self.padding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C % 4 == 0, "Channels must be multiple of 4 (r,i,j,k)."
        Cq = C // 4
        a, b, c, d = torch.split(x, Cq, dim=1)  # r,i,j,k
        ar = self._conv(a, self.Wr); ai = self._conv(a, self.Wi); aj = self._conv(a, self.Wj); ak = self._conv(a, self.Wk)
        br = self._conv(b, self.Wr); bi = self._conv(b, self.Wi); bj = self._conv(b, self.Wj); bk = self._conv(b, self.Wk)
        cr = self._conv(c, self.Wr); ci = self._conv(c, self.Wi); cj = self._conv(c, self.Wj); ck = self._conv(c, self.Wk)
        dr = self._conv(d, self.Wr); di = self._conv(d, self.Wi); dj = self._conv(d, self.Wj); dk = self._conv(d, self.Wk)
        o_r =  ar - bi - cj - dk
        o_i =  ai + br + ck - dj
        o_j =  aj - bk + cr + di
        o_k =  ak + bj - ci + dr
        if self.bias is not None:
            o_r = o_r + self.bias[:, None]
            o_i = o_i + self.bias[:, None]
            o_j = o_j + self.bias[:, None]
            o_k = o_k + self.bias[:, None]
        return torch.cat([o_r, o_i, o_j, o_k], dim=1)

class QBN1d(nn.Module):
    def __init__(self, qc: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(4 * qc)
    def forward(self, x): return self.bn(x)

class QAct(nn.Module):
    def __init__(self, act=nn.ReLU()):
        super().__init__()
        self.act = act
    def forward(self, x): return self.act(x)

class QuatUnitLockedDropout(nn.Module):
    """Drops whole quaternion units together, locked across time."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = float(p)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0: return x
        B, C, T = x.shape
        assert C % 4 == 0
        Cq = C // 4
        keep = 1.0 - self.p
        m = (torch.rand(B, Cq, 1, device=x.device, dtype=x.dtype) < keep) / keep
        m = m.repeat_interleave(4, dim=1)
        return x * m

class QuatCovNN(nn.Module):
    """Conv block with BN, activation. (No FiLM here; attitude enters as its own quat channel.)"""
    def __init__(self, in_qc: int, out_qc: int, kernel_size=3, padding=1):
        super().__init__()
        self.qconv = QuaternionConv1d(in_qc, out_qc, kernel_size=kernel_size, padding=padding)
        self.qbn = QBN1d(out_qc)
        self.qact = QAct(nn.ReLU())
    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        return self.qact(self.qbn(self.qconv(x_q)))

# ==================== Model (inputs 7 -> outputs 3) ====================

class VelNetV7(nn.Module):
    """
    Input:  [B,T,7] = [ax,ay,az, qw,qx,qy,qz]
      Internally packs to two quaternion channels:
        q_acc = (0, ax, ay, az), q_att = (qw,qx,qy,qz) -> [B, 4*2, T]
    Output: [B,T,3] = [vE, vN, vD]
    """
    def __init__(self, qwidth: int = 32, dropout_p: float = 0.3):
        super().__init__()
        self.in_qc = 2
        self.qw = qwidth
        self.block1 = QuatCovNN(self.in_qc, self.qw, kernel_size=5, padding=2)
        self.drop1  = QuatUnitLockedDropout(dropout_p)
        self.block2 = QuatCovNN(self.qw,     self.qw, kernel_size=3, padding=1)
        self.drop2  = QuatUnitLockedDropout(dropout_p)
        self.block3 = QuatCovNN(self.qw,     self.qw, kernel_size=3, padding=1)
        self.drop3  = QuatUnitLockedDropout(dropout_p)
        self.head = nn.Linear(4*self.qw, 3)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in=7 -> pack to 2 quat chans, "
                f"blocks=3×QuatCovNN(qwidth={self.qw}), head→3 v)")

    @staticmethod
    def _pack_inputs_to_quat_channels(x_bt7: torch.Tensor) -> torch.Tensor:
        ax, ay, az = x_bt7[..., 0], x_bt7[..., 1], x_bt7[..., 2]
        qw, qx, qy, qz = x_bt7[..., 3], x_bt7[..., 4], x_bt7[..., 5], x_bt7[..., 6]
        zeros = torch.zeros_like(ax)
        q_acc = torch.stack([zeros, ax, ay, az], dim=-1)            # [B,T,4]
        q_att = torch.stack([qw, qx, qy, qz], dim=-1)               # [B,T,4]
        q_att = quat_normalize(q_att)
        x_quat = torch.cat([q_acc, q_att], dim=-1).transpose(1, 2)  # [B,8,T]
        return x_quat

    def forward(self, x_bt7: torch.Tensor) -> torch.Tensor:
        x_q = self._pack_inputs_to_quat_channels(x_bt7)
        z = self.block1(x_q); z = self.drop1(z)
        z = self.block2(z);   z = self.drop2(z)
        z = self.block3(z);   z = self.drop3(z)
        z = z.transpose(1, 2)              # [B,T,4*qw]
        y = self.head(z)                   # [B,T,3]
        return y

# ==================== Losses ====================

@dataclass
class LossWeights:
    mse: float = 1.0
    trans: float = 1.0

def mse_vel(pred_bt3: torch.Tensor, gt_bt3: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_bt3, gt_bt3)

def loss_translational_END(vn_bt3: torch.Tensor,
                           q_in_bt4: torch.Tensor,
                           f_b_bt3: torch.Tensor,
                           dt: float,
                           g: float = 9.81) -> torch.Tensor:
    """
    Strapdown translational kinematics in END (using **input quaternion**):
    dv/dt ≈ R(q_in) (f_b) + g^n, with g^n=[0,0,+g] (Down positive).
    Trapezoidal rotation of f_b for stability.
    """
    if vn_bt3.size(1) < 2: return vn_bt3.new_tensor(0.0)
    q_unit = quat_normalize(q_in_bt4)
    f_n_t   = quat_rotate(q_unit[:, :-1], f_b_bt3[:, :-1])
    f_n_tp1 = quat_rotate(q_unit[:,  1:], f_b_bt3[:,  1:])
    f_n_trap = 0.5 * (f_n_t + f_n_tp1)
    dv_dt = (vn_bt3[:, 1:] - vn_bt3[:, :-1]) / dt
    g_n = vn_bt3.new_tensor([0.0, 0.0, g])
    model = f_n_trap + g_n
    return (dv_dt - model).pow(2).mean()

def total_loss(pred_bt3: torch.Tensor,
               gt_bt3: torch.Tensor,
               x_bt7: torch.Tensor,
               dt: float,
               w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    x_bt7 channels: [ax,ay,az, qw,qx,qy,qz]
    """
    L_mse = mse_vel(pred_bt3, gt_bt3)
    f_b = x_bt7[..., 0:3]
    q_in = x_bt7[..., 3:7]
    L_trans = loss_translational_END(pred_bt3, q_in, f_b, dt)
    L = w.mse*L_mse + w.trans*L_trans
    return L, {
        "total": float(L.detach().cpu()),
        "mse_v": float(L_mse.detach().cpu()),
        "phys_trans": float(L_trans.detach().cpu()),
    }

# ==================== Training scaffold ====================

@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    dt: float = 0.05
    dropout_p: float = 0.3
    qwidth: int = 32
    device: str = "cuda"
    loss_w: "LossWeights" = field(default_factory=LossWeights)

class SeqDataset(torch.utils.data.Dataset):
    """Holds tensors: x:[N,T,7], y:[N,T,3]"""
    def __init__(self, x_bt7: torch.Tensor, y_bt3: torch.Tensor):
        assert x_bt7.shape[:2] == y_bt3.shape[:2]
        self.x = x_bt7; self.y = y_bt3
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

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
        "val_by_output": [[] for _ in range(3)],          # per-output MSE (vE,vN,vD)
        "val_total_by_output": [[] for _ in range(3)],     # per-output TOTAL (MSE + attributed phys)
    }

    for epoch in range(cfg.epochs):
        # ---------------- Train ----------------
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)  # [B,T,7]
            yb = yb.to(cfg.device)  # [B,T,3]
            pred = model(xb)        # [B,T,3]
            L, _ = total_loss(pred, yb, xb, cfg.dt, cfg.loss_w)
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
            per_out_mse_sums = [0.0]*3
            per_out_total_sums = [0.0]*3
            batches = 0

            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb)
                L, logs = total_loss(pred, yb, xb, cfg.dt, cfg.loss_w)
                tot_sum += logs["total"]
                mse_v_sum += logs["mse_v"]

                # Per-output MSE
                mse_vec = torch.mean((pred - yb).pow(2), dim=(0,1))  # [3]

                # Attribute physics term equally across vE,vN,vD
                add_v = cfg.loss_w.trans * logs["phys_trans"]
                per_total = (cfg.loss_w.mse * mse_vec).cpu().numpy()
                per_total[0:3] += add_v

                for i in range(3):
                    per_out_mse_sums[i]   += float(mse_vec[i].cpu())
                    per_out_total_sums[i] += float(per_total[i])

                batches += 1

            val_total = tot_sum / max(1, len(val_loader))
            hist["val_total"].append(val_total)
            hist["val_mse_v"].append(mse_v_sum / max(1, len(val_loader)))
            for i in range(3):
                hist["val_by_output"][i].append(per_out_mse_sums[i] / max(1, batches))
                hist["val_total_by_output"][i].append(per_out_total_sums[i] / max(1, batches))

        print(f"[{epoch+1:03d}/{cfg.epochs}] train_total={train_total:.6g}  "
              f"val_total={val_total:.6g}  (val MSE v={hist['val_mse_v'][-1]:.6g})")

    return hist

# ==================== Plotting ====================

def plot_training_curves(hist: Dict[str, List[float]], out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(hist["train_total"])+1)

    # Total loss (linear)
    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.xlabel("epoch"); plt.ylabel("total loss"); plt.title("Total Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_linear.png"); plt.close()

    # Total loss (log)
    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("total loss (log)"); plt.title("Total Loss (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_log.png"); plt.close()

    # Validation per-output MSE (3 outputs)
    names = ["vE","vN","vD"]
    plt.figure()
    for i in range(3):
        plt.plot(epochs, hist["val_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val MSE per output"); plt.title("Validation MSE per output (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_mse_per_output.png"); plt.close()

    # Validation TOTAL per output (log)
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
        "epoch": list(range(1, len(hist["train_total"])+1)),
        "train_total": hist["train_total"],
        "val_total":   hist["val_total"],
        "val_mse_v":   hist["val_mse_v"],
    }
    for i, name in enumerate(["vE","vN","vD"]):
        cols[f"val_mse_{name}"] = hist["val_by_output"][i]
        cols[f"val_total_{name}"] = hist["val_total_by_output"][i]
    pd.DataFrame(cols).to_csv(out_path, index=False)

# ==================== Data I/O / normalization ====================

IN_COLS  = ["ax","ay","az","qw","qx","qy","qz"]
OUT_COLS = ["vE","vN","vD"]

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
        y_scaled = (y_np - self.y_mean) / self.y_std
        return y_scaled

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

def read_csv_columns(path: str, in_cols=IN_COLS, out_cols=OUT_COLS) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    miss_in  = [c for c in in_cols  if c not in df.columns]
    miss_out = [c for c in out_cols if c not in df.columns]
    if miss_in or miss_out:
        raise ValueError(f"{path}: missing columns. Missing inputs={miss_in}, missing outputs={miss_out}")
    x = df[in_cols].to_numpy(dtype=np.float32)
    y = df[out_cols].to_numpy(dtype=np.float32)
    # ensure quaternion inputs are unit (for safety)
    q = x[:, 3:7]
    qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    x[:, 3:7] = q / qn
    return x, y

def make_sequences(x: np.ndarray, y: np.ndarray, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N = x.shape[0]; B = N // T
    if B == 0:
        raise ValueError(f"Sequence length {T} longer than data ({N}).")
    x = x[:B*T].reshape(B, T, -1)
    y = y[:B*T].reshape(B, T, -1)
    return torch.from_numpy(x), torch.from_numpy(y)

def build_loaders(train_csv: str, val_csv: str, norm: Normalizer, seq_len: int,
                  batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    xtr, ytr = read_csv_columns(train_csv)
    xva, yva = read_csv_columns(val_csv)
    xtr = norm.x(xtr); xva = norm.x(xva)
    ytr = norm.y(ytr); yva = norm.y(yva)
    xtr_t, ytr_t = make_sequences(xtr, ytr, seq_len)
    xva_t, yva_t = make_sequences(xva, yva, seq_len)
    tr_ds = SeqDataset(xtr_t, ytr_t)
    va_ds = SeqDataset(xva_t, yva_t)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return tr_dl, va_dl

def save_hist_json(hist: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(hist, f, indent=2)

def average_histories(hists: List[dict]) -> dict:
    out = {}
    keys = ["train_total","val_total","val_mse_v"]
    for k in keys:
        out[k] = list(np.mean([h[k] for h in hists], axis=0))
    out["val_by_output"] = []
    for i in range(3):
        per_i = [h["val_by_output"][i] for h in hists]
        out["val_by_output"].append(list(np.mean(per_i, axis=0)))
    out["val_total_by_output"] = []
    for i in range(3):
        out["val_total_by_output"].append(list(np.mean([h["val_total_by_output"][i] for h in hists], axis=0)))
    return out

# ==================== Export helpers ====================

def _export_torchscript(model, out_path: str, seq_len: int = 200):
    model_cpu = model.eval().to("cpu")
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception as e:
        print(f"[export] script() failed → tracing instead: {e}")
        example = torch.zeros(1, seq_len, 7, dtype=torch.float32)
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
        example = torch.zeros(1, seq_len, 7, dtype=torch.float32)
        scripted = torch.jit.trace(ens, example, strict=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"[export ensemble] wrote TorchScript: {out_path}")

# ==================== CLI / main ====================

def main():
    parser = argparse.ArgumentParser(description="Train velocity-only NN (no gyro) with physics-informed loss.")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val",   required=True, help="Path to validation CSV")
    parser.add_argument("--out",   required=True, help="Output directory (models, plots)")
    parser.add_argument("--seq",   type=int, default=200, help="Sequence length T")
    parser.add_argument("--epochs",type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64,  help="Batch size")
    parser.add_argument("--lr",    type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd",    type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gpu",   action="store_true", help="Use CUDA if available")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of models to train")
    parser.add_argument("--dropout",  type=float, default=0.3, help="Locked dropout p")
    parser.add_argument("--qwidth",   type=int, default=32, help="Quaternion conv width")
    parser.add_argument("--dt",       type=float, default=0.01, help="Sample period (s)")
    parser.add_argument("--norm_json", type=str, default="", help="Normalization JSON path")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed (member 0). Members use seed+i")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

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

    # Data loaders
    def build_loaders_local(train_csv, val_csv, seq_len, batch_size):
        xtr, ytr = _read_cols(train_csv); xva, yva = _read_cols(val_csv)
        xtr = norm.x(xtr); xva = norm.x(xva)
        ytr = norm.y(ytr); yva = norm.y(yva)
        xtr_t, ytr_t = make_sequences(xtr, ytr, seq_len)
        xva_t, yva_t = make_sequences(xva, yva, seq_len)
        tr = torch.utils.data.DataLoader(SeqDataset(xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
        va = torch.utils.data.DataLoader(SeqDataset(xva_t, yva_t), batch_size=batch_size, shuffle=False)
        return tr, va

    train_loader, val_loader = build_loaders_local(args.train, args.val, args.seq, args.batch)

    # Where to collect TorchScript members for easy loading
    ts_collect_dir = os.path.join(args.out, "ts")
    os.makedirs(ts_collect_dir, exist_ok=True)

    # Train ensemble
    all_hists = []
    member_ckpts = []
    for m in range(args.ensemble):
        print("\n" + "="*90)
        print(f"Ensemble member {m+1}/{args.ensemble}")
        print("="*90)
        set_seed(args.seed + m)

        model = VelNetV7(qwidth=args.qwidth, dropout_p=args.dropout).to(device)
        cfg = TrainCfg(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
                       dt=args.dt, dropout_p=args.dropout, qwidth=args.qwidth,
                       device=device, loss_w=LossWeights(mse=1.0, trans=1.0))

        member_dir = os.path.join(args.out, f"member_{m:02d}")
        os.makedirs(member_dir, exist_ok=True)
        with open(os.path.join(member_dir, "config.json"), "w") as f:
            json.dump({
                "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                "dt": cfg.dt, "dropout_p": cfg.dropout_p, "qwidth": cfg.qwidth,
                "device": cfg.device, "seed": args.seed + m
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
        example = torch.zeros(1, args.seq, 7, dtype=torch.float32)
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

    # Ensemble-mean plots
    if args.ensemble > 1:
        ens_hist = average_histories(all_hists)
        with open(os.path.join(args.out, "ensemble_history_mean.json"), "w") as f:
            json.dump(ens_hist, f, indent=2)
        plot_training_curves(ens_hist, out_dir=args.out)
        print(f"\nEnsemble mean curves saved to {args.out}")

    # Export a single ensemble_stateless.pt (mean of members)
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
        m = VelNetV7(qwidth=args.qwidth, dropout_p=args.dropout).eval().to("cpu")
        ckpt = torch.load(pth, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        m.load_state_dict(state, strict=True)
        members.append(m)

    if len(members) >= 1:
        ens = Ensemble(members).eval().to("cpu")
        ens_pt_path = os.path.join(args.out, "ensemble_stateless.pt")
        example = torch.zeros(1, args.seq, 7, dtype=torch.float32)
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
        shutil.copyfile(ens_pt_path, os.path.join(ts_collect_dir, "ensemble_stateless.pt"))
    else:
        print("[export ensemble] No members available to export an ensemble.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_model_v6.py
Quaternion + Velocity NN with physics-informed losses, ensemble training, and plotting.

Run example:
python3 scripts/nn_model_v6.py \
  --train data/nn_dataset_v6_X_C0/train.csv \
  --val   data/nn_dataset_v6_X_C0/val.csv \
  --out   data/nn_model_v6_ens4 \
  --seq 200 --epochs 400 --gpu --ensemble 4 \
  --norm_json data/nn_dataset_v6_X_C0/norm_stats.json
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

def _export_torchscript(model, out_path: str, seq_len: int = 200):
    """
    Export a model whose forward expects [B,T,8] and returns [B,T,7].
    We prefer scripting; fall back to trace with a dummy example.
    Always exports a CPU module.
    """
    model_cpu = model.eval().to("cpu")
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception as e:
        print(f"[export] script() failed → tracing instead: {e}")
        example = torch.zeros(1, seq_len, 8, dtype=torch.float32)
        scripted = torch.jit.trace(model_cpu, example, strict=False)
    scripted.save(out_path)
    print(f"[export] wrote TorchScript: {out_path}")

def export_member_torchscript(model, member_idx: int, run_dir: str, collect_dir: str, seq_len: int):
    """
    Save TorchScript for one member in:
      {run_dir}/ts/member_XX.pt      (member-local)
    and also copy to a flat collection directory:
      {collect_dir}/member_XX.pt     (for MEKF to load all members easily)
    """
    os.makedirs(os.path.join(run_dir, "ts"), exist_ok=True)
    os.makedirs(collect_dir, exist_ok=True)
    local_pt = os.path.join(run_dir, "ts", f"member_{member_idx:02d}.pt")
    _export_torchscript(model, local_pt, seq_len=seq_len)
    shutil.copyfile(local_pt, os.path.join(collect_dir, f"member_{member_idx:02d}.pt"))

def export_ensemble_torchscript(ModelClass, member_pths: list, out_path: str, seq_len: int):
    """
    Load all member state_dicts, wrap in a simple averaging Ensemble, and export once.
    Keeps scripting first; falls back to tracing.
    """
    class Ensemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)
        def forward(self, x):
            ys = [m(x) for m in self.members]   # each [B,T,7]
            y  = torch.stack(ys, dim=0).mean(0) # [B,T,7]
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
        example = torch.zeros(1, seq_len, 8, dtype=torch.float32)
        scripted = torch.jit.trace(ens, example, strict=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"[export ensemble] wrote TorchScript: {out_path}")

# ==================== Quaternion utilities (Body→END active rotation) ====================

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
    qv = torch.cat([torch.zeros_like(v3[..., :1]), v3], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]

def quat_rel(q_prev: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
    """Relative quaternion: q_rel = q_prev*^{-1} ⊗ q_next. Both unit."""
    return quat_mul(quat_conj(q_prev), q_next)

def rotvec_from_quat(q_rel: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """q_rel unit quaternion -> rotation vector (axis * angle)."""
    w, x, y, z = q_rel.unbind(-1)
    v = torch.stack([x, y, z], dim=-1)
    v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)
    angle = 2.0 * torch.atan2(v_norm.squeeze(-1), torch.clamp(w, -1.0, 1.0))
    axis = v / v_norm
    return axis * angle.unsqueeze(-1)  # [...,3]

# ==================== Quaternion Conv1D building blocks ====================

class QuaternionConv1d(nn.Module):
    """
    Quaternion 1D convolution over time.
    Input:  [B, 4*Cin, T] with (r,i,j,k) stacked along channels.
    Output: [B, 4*Cout, T].
    """
    def __init__(self, in_qc: int, out_qc: int, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.ic, self.oc = in_qc, out_qc
        k = kernel_size
        # Four real kernels composed as Hamilton product
        self.Wr = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.Wi = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.Wj = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.Wk = nn.Parameter(torch.empty(out_qc, in_qc, k))
        self.bias = nn.Parameter(torch.zeros(out_qc)) if bias else None

        for W in (self.Wr, self.Wi, self.Wj, self.Wk):
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

        self.stride, self.padding = stride, padding

    def _conv(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # x: [B,Cin,T], W: [Cout,Cin,K] -> [B,Cout,T]
        return F.conv1d(x, W, bias=None, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C % 4 == 0, "Channels must be multiple of 4 (r,i,j,k)."
        Cq = C // 4
        a, b, c, d = torch.split(x, Cq, dim=1)  # r,i,j,k components

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
    """
    Time-locked dropout that drops whole quaternion units together (r,i,j,k share one mask),
    shared across time. Input: [B, 4*Cq, T]
    """
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

class QuatFiLM(nn.Module):
    """
    FiLM modulation conditioned on a small real feature (here yaw inputs cpsi,spsi).
    Produces per-quaternion-channel gamma/beta and repeats over r,i,j,k.
    cond: [B, T, cond_dim]  -> gamma/beta: [B, 4*Cq, T]
    """
    def __init__(self, cq: int, cond_dim: int):
        super().__init__()
        self.cq = cq
        self.gamma = nn.Conv1d(cond_dim, cq, kernel_size=1)
        self.beta  = nn.Conv1d(cond_dim, cq, kernel_size=1)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)
    def forward(self, x_q: torch.Tensor, cond_bt2: torch.Tensor) -> torch.Tensor:
        # x_q: [B, 4*Cq, T]; cond_bt2: [B, T, cond_dim]
        cond = cond_bt2.transpose(1, 2)  # [B, cond_dim, T]
        g = self.gamma(cond)  # [B, Cq, T]
        b = self.beta(cond)   # [B, Cq, T]
        g = g.repeat_interleave(4, dim=1)
        b = b.repeat_interleave(4, dim=1)
        return x_q * (1.0 + g) + b

class QuatCovNN(nn.Module):
    """ One quaternion convolution block with BN, activation, and optional FiLM conditioning. """
    def __init__(self, in_qc: int, out_qc: int, kernel_size=3, padding=1, cond_dim: int = 2):
        super().__init__()
        self.qconv = QuaternionConv1d(in_qc, out_qc, kernel_size=kernel_size, padding=padding)
        self.qbn = QBN1d(out_qc)
        self.qact = QAct(nn.ReLU())
        self.film = QuatFiLM(out_qc, cond_dim) if cond_dim is not None and cond_dim > 0 else None
    def forward(self, x_q: torch.Tensor, cond_bt2: torch.Tensor = None) -> torch.Tensor:
        z = self.qact(self.qbn(self.qconv(x_q)))
        if self.film is not None and cond_bt2 is not None:
            z = self.film(z, cond_bt2)
        return z

# ==================== Model: 3×(QuatCovNN + locked dropout) -> head(7) ====================

class QuatVelAttNet(nn.Module):
    """
    Input:  [B,T,8] with channels [p,q,r, ax,ay,az, cpsi,spsi]
    Backbone: QuatCovNN × 3 with time-locked quaternion dropout after each
    Output: [B,T,7] -> [vE,vN,vD, w,x,y,z]
    """
    def __init__(self, qwidth: int = 32, dropout_p: float = 0.3):
        super().__init__()
        self.in_qc = 2            # (0,p,q,r) and (0,ax,ay,az) -> 2 quaternion channels -> 8 real
        self.qw = qwidth

        self.block1 = QuatCovNN(self.in_qc, self.qw, kernel_size=5, padding=2, cond_dim=2)
        self.drop1  = QuatUnitLockedDropout(dropout_p)
        self.block2 = QuatCovNN(self.qw,     self.qw, kernel_size=3, padding=1, cond_dim=2)
        self.drop2  = QuatUnitLockedDropout(dropout_p)
        self.block3 = QuatCovNN(self.qw,     self.qw, kernel_size=3, padding=1, cond_dim=2)
        self.drop3  = QuatUnitLockedDropout(dropout_p)

        # Head: collapse quaternion channels to real features then Linear->7
        self.head = nn.Linear(4 * self.qw, 7)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in=8, blocks=3×QuatCovNN(qwidth={self.qw}), "
                f"dropout='QuatUnitLocked', head=Linear(4*{self.qw}→7) -> [vE,vN,vD,w,x,y,z])")

    @staticmethod
    def _pack_inputs_to_quat_channels(x_bt8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,T,8] -> (x_quat [B,8,T], cond [B,T,2])
        Channels assumed: [p,q,r, ax,ay,az, cpsi,spsi]
        """
        p,q,r = x_bt8[..., 0], x_bt8[..., 1], x_bt8[..., 2]
        ax,ay,az = x_bt8[..., 3], x_bt8[..., 4], x_bt8[..., 5]
        cpsi, spsi = x_bt8[..., 6], x_bt8[..., 7]

        zeros = torch.zeros_like(p)
        q_gyro = torch.stack([zeros, p, q, r], dim=-1)   # [B,T,4]
        q_acc  = torch.stack([zeros, ax, ay, az], dim=-1)# [B,T,4]
        x_quat = torch.cat([q_gyro, q_acc], dim=-1)      # [B,T,8] -> [B, 4*2]
        x_quat = x_quat.transpose(1, 2)                  # [B,8,T] = [B,4*Cin,T]
        cond_bt2 = torch.stack([cpsi, spsi], dim=-1)     # [B,T,2]
        return x_quat, cond_bt2

    def forward(self, x_bt8: torch.Tensor) -> torch.Tensor:
        # x_bt8: [B,T,8]
        x_q, cond = self._pack_inputs_to_quat_channels(x_bt8)   # [B,8,T], [B,T,2]
        z = self.block1(x_q, cond); z = self.drop1(z)
        z = self.block2(z, cond);  z = self.drop2(z)
        z = self.block3(z, cond);  z = self.drop3(z)
        z = z.transpose(1, 2)                      # [B,T,4*qw]
        y = self.head(z)                           # [B,T,7]
        return y

# ==================== Losses ====================

@dataclass
class LossWeights:
    mse: float = 1.0
    trans: float = 1.0
    relrot: float = 1.0
    unit: float = 0.01

def mse_outputs(pred_bt7: torch.Tensor, gt_bt7: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    MSE over velocities and quaternion (sign-safe, normalized).
    pred/gt: [B,T,7] = [vE,vN,vD, w,x,y,z]
    """
    v_pred = pred_bt7[..., :3]
    q_pred_raw = pred_bt7[..., 3:]
    v_gt = gt_bt7[..., :3]
    q_gt_raw = gt_bt7[..., 3:]

    # Normalize & sign-align quaternion vs GT (double-cover)
    q_pred = quat_normalize(q_pred_raw)
    q_gt   = quat_normalize(q_gt_raw)
    dot = (q_pred * q_gt).sum(dim=-1, keepdim=True)
    q_pred_aligned = torch.where(dot < 0, -q_pred, q_pred)

    mse_v = F.mse_loss(v_pred, v_gt)
    mse_q = F.mse_loss(q_pred_aligned, q_gt)

    return mse_v + mse_q, {"mse_v": mse_v.detach(), "mse_q": mse_q.detach()}

def loss_translational_END(vn_bt3: torch.Tensor,
                           q_bt4: torch.Tensor,
                           f_b_bt3: torch.Tensor,
                           dt: float,
                           g: float = 9.81) -> torch.Tensor:
    """
    Strapdown translational kinematics in END:
    dv/dt ≈ R(q) (f_b) + g^n, with g^n=[0,0,+g] (Down positive).
    Uses trapezoidal rotation of f_b for stability.
    """
    if vn_bt3.size(1) < 2: return vn_bt3.new_tensor(0.0)

    q_unit = quat_normalize(q_bt4)
    f_n_t   = quat_rotate(q_unit[:, :-1], f_b_bt3[:, :-1])  # [B,T-1,3]
    f_n_tp1 = quat_rotate(q_unit[:,  1:], f_b_bt3[:,  1:])  # [B,T-1,3]
    f_n_trap = 0.5 * (f_n_t + f_n_tp1)

    dv_dt = (vn_bt3[:, 1:] - vn_bt3[:, :-1]) / dt
    g_n = vn_bt3.new_tensor([0.0, 0.0, g])  # [3]
    model = f_n_trap + g_n

    return (dv_dt - model).pow(2).mean()

def loss_relative_rotation(q_bt4: torch.Tensor,
                           omega_bt3: torch.Tensor,
                           dt: float) -> torch.Tensor:
    """
    Relative quaternion vs gyro: q_rel ≈ exp(0.5*[0, ω]*dt) → rotvec ≈ ω*dt
    Implemented via rotvec from q_rel: rotvec/dt ≈ ω
    """
    if q_bt4.size(1) < 2: return q_bt4.new_tensor(0.0)
    q = quat_normalize(q_bt4)
    # Sign continuity across time (helps stability)
    q_prev = q[:, :-1]
    q_next = q[:, 1:]
    same_sign = torch.sign((q_prev * q_next).sum(dim=-1, keepdim=True) + 1e-8)
    q_next = torch.where(same_sign < 0, -q_next, q_next)

    qrel = quat_rel(q_prev, q_next)          # [B,T-1,4]
    rotvec = rotvec_from_quat(qrel)          # [B,T-1,3]
    omega_fd = rotvec / dt
    return (omega_fd - omega_bt3[:, :-1]).pow(2).mean()

def loss_unit_sphere(q_bt4: torch.Tensor) -> torch.Tensor:
    q = q_bt4
    return (q.pow(2).sum(dim=-1) - 1.0).pow(2).mean()

def total_loss(pred_bt7: torch.Tensor,
               gt_bt7: torch.Tensor,
               x_bt8: torch.Tensor,
               dt: float,
               w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    x_bt8: inputs [p,q,r, ax,ay,az, cpsi,spsi] used for physics terms (omega, f_b)
    """
    # Base MSE (sign-safe quat)
    mse_all, mse_dict = mse_outputs(pred_bt7, gt_bt7)

    # Physics terms
    omega = x_bt8[..., 0:3]
    f_b   = x_bt8[..., 3:6]
    q_pred = pred_bt7[..., 3:]
    v_pred = pred_bt7[..., :3]

    L_trans = loss_translational_END(v_pred, q_pred, f_b, dt)
    L_rel   = loss_relative_rotation(q_pred, omega, dt)
    L_unit  = loss_unit_sphere(q_pred)

    L = w.mse*mse_all + w.trans*L_trans + w.relrot*L_rel + w.unit*L_unit

    logs = {
        "total": float(L.detach().cpu()),
        "mse_total": float(mse_all.detach().cpu()),
        "mse_v": float(mse_dict["mse_v"].cpu()),
        "mse_q": float(mse_dict["mse_q"].cpu()),
        "phys_trans": float(L_trans.detach().cpu()),
        "phys_relrot": float(L_rel.detach().cpu()),
        "phys_unit": float(L_unit.detach().cpu()),
    }
    return L, logs

# ==================== Training scaffold ====================

@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    dt: float = 0.05          # seconds between timesteps
    dropout_p: float = 0.3
    qwidth: int = 32
    device: str = "cuda"
    loss_w: "LossWeights" = field(default_factory=LossWeights)

class SeqDataset(torch.utils.data.Dataset):
    """
    Holds tensors: x:[N,T,8], y:[N,T,7]
    x channels: [p,q,r, ax,ay,az, cpsi,spsi]
    y channels: [vE,vN,vD, w,x,y,z]  (quaternion Body→END)
    """
    def __init__(self, x_bt8: torch.Tensor, y_bt7: torch.Tensor):
        assert x_bt8.shape[0] == y_bt7.shape[0] and x_bt8.shape[1] == y_bt7.shape[1]
        self.x = x_bt8; self.y = y_bt7
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
        "val_mse_q": [],
        "val_by_output": [[] for _ in range(7)],          # per-output MSE (vE,vN,vD,qw,qx,qy,qz)
        "val_total_by_output": [[] for _ in range(7)],     # per-output TOTAL (MSE + attributed physics)
    }

    for epoch in range(cfg.epochs):
        # ---------------- Train ----------------
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)  # [B,T,8]
            yb = yb.to(cfg.device)  # [B,T,7]
            pred = model(xb)        # [B,T,7]

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
            mse_q_sum = 0.0
            per_out_mse_sums = [0.0] * 7
            per_out_total_sums = [0.0] * 7
            batches = 0

            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb)

                L, logs = total_loss(pred, yb, xb, cfg.dt, cfg.loss_w)
                tot_sum += logs["total"]
                mse_v_sum += logs["mse_v"]
                mse_q_sum += logs["mse_q"]

                # ----- Per-output plain MSE (no physics) -----
                v_pred = pred[..., :3]
                v_gt   = yb  [..., :3]
                q_pred = quat_normalize(pred[..., 3:])
                q_gt   = quat_normalize(yb  [..., 3:])
                # sign-align quaternion prediction to GT
                dot = (q_pred * q_gt).sum(dim=-1, keepdim=True)
                q_pred = torch.where(dot < 0, -q_pred, q_pred)

                # mean over batch and time -> [7]
                mse_vec = torch.mean(
                    (torch.cat([v_pred, q_pred], dim=-1) - torch.cat([v_gt, q_gt], dim=-1)).pow(2),
                    dim=(0, 1)
                )  # tensor shape [7]

                # ----- Per-output TOTAL = weighted MSE + attributed physics -----
                # Physics terms were logged *unweighted*; attribute to heads and apply weights now.
                add_v = cfg.loss_w.trans * logs["phys_trans"]  # equally across vE,vN,vD
                add_q = (cfg.loss_w.relrot * logs["phys_relrot"] + cfg.loss_w.unit * logs["phys_unit"]) / 4.0  # across qw..qz

                per_total = (cfg.loss_w.mse * mse_vec).cpu().numpy()
                per_total[0:3] += add_v
                per_total[3:7] += add_q

                # Accumulate
                for i in range(7):
                    per_out_mse_sums[i]   += float(mse_vec[i].cpu())
                    per_out_total_sums[i] += float(per_total[i])

                batches += 1

            val_total = tot_sum / max(1, len(val_loader))
            hist["val_total"].append(val_total)
            hist["val_mse_v"].append(mse_v_sum / max(1, len(val_loader)))
            hist["val_mse_q"].append(mse_q_sum / max(1, len(val_loader)))

            for i in range(7):
                hist["val_by_output"][i].append(per_out_mse_sums[i] / max(1, batches))
                hist["val_total_by_output"][i].append(per_out_total_sums[i] / max(1, batches))

        print(f"[{epoch+1:03d}/{cfg.epochs}] "
              f"train_total={train_total:.6g}  "
              f"val_total={val_total:.6g}  "
              f"(val MSE v={hist['val_mse_v'][-1]:.6g}, q={hist['val_mse_q'][-1]:.6g})")

    return hist


# ==================== Plotting ====================

def plot_training_curves(hist: Dict[str, List[float]], out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(hist["train_total"])+1)

    # Total loss: linear scale
    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.xlabel("epoch"); plt.ylabel("total loss"); plt.title("Total Loss (linear)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_linear.png"); plt.close()

    # Total loss: log scale
    plt.figure()
    plt.plot(epochs, hist["train_total"], label="train total")
    plt.plot(epochs, hist["val_total"], label="val total")
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("total loss (log)"); plt.title("Total Loss (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/loss_total_log.png"); plt.close()

    # Validation per-output MSE (7 outputs)
    names = ["vE","vN","vD","qw","qx","qy","qz"]
    plt.figure()
    for i in range(7):
        plt.plot(epochs, hist["val_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val MSE per output"); plt.title("Validation MSE per output (log scale)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_mse_per_output.png"); plt.close()

    # NEW: Validation total loss per output (log scale, includes physics attribution)
    names = ["vE","vN","vD","qw","qx","qy","qz"]
    plt.figure()
    for i in range(7):
        plt.plot(range(1, len(hist["train_total"])+1),
                 hist["val_total_by_output"][i], label=names[i])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("val TOTAL per output")
    plt.title("Validation TOTAL loss per output (log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/val_total_per_output.png"); plt.close()

def save_history_csv(hist: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Flatten per-output into columns
    cols = {
        "epoch": list(range(1, len(hist["train_total"])+1)),
        "train_total": hist["train_total"],
        "val_total":   hist["val_total"],
        "val_mse_v":   hist["val_mse_v"],
        "val_mse_q":   hist["val_mse_q"],
    }
    for i, name in enumerate(["vE","vN","vD","qw","qx","qy","qz"]):
        cols[f"val_mse_{name}"] = hist["val_by_output"][i]
        cols[f"val_total_{name}"] = hist["val_total_by_output"][i]
    pd.DataFrame(cols).to_csv(out_path, index=False)

# ==================== Data I/O, normalization, helpers ====================

IN_COLS  = ["p","q","r","ax","ay","az","cpsi","spsi"]
OUT_COLS = ["vE","vN","vD","qw","qx","qy","qz"]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Normalizer:
    """Standardize inputs (always). Optionally standardize only velocity outputs (first 3)."""
    def __init__(self, x_mean, x_std, y_mean=None, y_std=None):
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std  = np.asarray(x_std,  dtype=np.float32)
        self.y_mean = None if y_mean is None else np.asarray(y_mean, dtype=np.float32)
        self.y_std  = None if y_std  is None else np.asarray(y_std,  dtype=np.float32)
        # Safety
        self.x_std[self.x_std == 0] = 1.0
        if self.y_std is not None:
            self.y_std[self.y_std == 0] = 1.0

    def x(self, x_np: np.ndarray) -> np.ndarray:
        return (x_np - self.x_mean) / self.x_std

    def y(self, y_np: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            return y_np
        # Only scale first 3 (vE,vN,vD). Leave quaternion untouched.
        y_scaled = y_np.copy()
        y_scaled[..., :3] = (y_np[..., :3] - self.y_mean) / self.y_std
        return y_scaled

def load_norm(norm_json_path: str, x_train: np.ndarray = None, y_train: np.ndarray = None) -> Normalizer:
    if norm_json_path and os.path.isfile(norm_json_path):
        with open(norm_json_path, "r") as f:
            d = json.load(f)
        x_mean, x_std = d["x_mean"], d["x_std"]
        y_mean = d.get("y_mean", None)
        y_std  = d.get("y_std",  None)
        return Normalizer(x_mean, x_std, y_mean, y_std)
    # Fallback: compute from train
    if x_train is None:
        raise ValueError("norm_json not found and no training data provided to compute norms.")
    x_mean = x_train.mean(axis=0).tolist()
    x_std  = x_train.std(axis=0).tolist()
    y_mean = None; y_std = None
    if y_train is not None:
        y_mean = y_train[:, :3].mean(axis=0).tolist()
        y_std  = y_train[:, :3].std(axis=0).tolist()
    return Normalizer(x_mean, x_std, y_mean, y_std)

def read_csv_columns(path: str, in_cols=IN_COLS, out_cols=OUT_COLS) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing_in  = [c for c in in_cols  if c not in df.columns]
    missing_out = [c for c in out_cols if c not in df.columns]
    if missing_in or missing_out:
        raise ValueError(f"{path}: missing columns. Missing inputs={missing_in}, missing outputs={missing_out}")
    x = df[in_cols].to_numpy(dtype=np.float32)
    y = df[out_cols].to_numpy(dtype=np.float32)
    # ensure quaternion rows are unit (just in case dataset is slightly off)
    q = y[:, 3:7]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    y[:, 3:7] = q / q_norm
    return x, y

def make_sequences(x: np.ndarray, y: np.ndarray, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs flat timeseries [N,D] into non-overlapping sequences [B,T,D].
    Remainder is dropped if N % T != 0. Adjust if you prefer sliding windows.
    """
    N = x.shape[0]
    B = N // T
    if B == 0:
        raise ValueError(f"Sequence length {T} longer than data ({N}).")
    x = x[:B*T].reshape(B, T, -1)
    y = y[:B*T].reshape(B, T, -1)
    return torch.from_numpy(x), torch.from_numpy(y)

def build_loaders(train_csv: str, val_csv: str, norm: Normalizer, seq_len: int,
                  batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    xtr, ytr = read_csv_columns(train_csv)
    xva, yva = read_csv_columns(val_csv)

    # Normalize
    xtr = norm.x(xtr); xva = norm.x(xva)
    ytr = norm.y(ytr); yva = norm.y(yva)

    # Pack to sequences
    xtr_t, ytr_t = make_sequences(xtr, ytr, seq_len)
    xva_t, yva_t = make_sequences(xva, yva, seq_len)

    # Datasets
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
    """Averages scalar lists across ensemble members for plotting an ensemble-mean curve."""
    out = {}
    keys_scalar_lists = ["train_total","val_total","val_mse_v","val_mse_q"]
    for k in keys_scalar_lists:
        out[k] = list(np.mean([h[k] for h in hists], axis=0))
    # per-output
    out["val_by_output"] = []
    for i in range(7):
        per_i = [h["val_by_output"][i] for h in hists]  # list of lists (members) -> [members, epochs]
        out["val_by_output"].append(list(np.mean(per_i, axis=0)))
    # NEW: per-output TOTAL (incl. physics attribution)
    out["val_total_by_output"] = []
    for i in range(7):
        out["val_total_by_output"].append(list(np.mean([h["val_total_by_output"][i] for h in hists], axis=0)))

    return out

# ==================== CLI / main ====================

def main():
    import shutil  # local import so the rest of your file needn't change

    parser = argparse.ArgumentParser(description="Train quaternion+velocity NN with physics-informed losses (ensemble-ready).")
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
    parser.add_argument("--dt",       type=float, default=0.05, help="Sample period (s)")
    parser.add_argument("--norm_json", type=str, default="", help="Normalization JSON path")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed (member 0). Members use seed+i")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    # Device
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # Load train CSV once to compute fallback norms if needed
    x_tmp, y_tmp = read_csv_columns(args.train)
    norm = load_norm(args.norm_json, x_tmp, y_tmp)
    # stash a copy of the norms we actually used
    with open(os.path.join(args.out, "norm_used.json"), "w") as f:
        json.dump({
            "x_mean": norm.x_mean.tolist(),
            "x_std":  norm.x_std.tolist(),
            "y_mean": None if norm.y_mean is None else norm.y_mean.tolist(),
            "y_std":  None if norm.y_std  is None else norm.y_std.tolist()
        }, f, indent=2)

    # Build loaders
    train_loader, val_loader = build_loaders(args.train, args.val, norm, args.seq, batch_size=args.batch)

    # Loss weights (tune if needed)
    loss_w = LossWeights(mse=1.0, trans=1.0, relrot=1.0, unit=0.01)

    # Where to collect all member .pt files for easy C++ loading
    ts_collect_dir = os.path.join(args.out, "ts")
    os.makedirs(ts_collect_dir, exist_ok=True)

    # Train ensemble
    all_hists = []
    member_ckpts = []  # keep paths to each member's model.pth for ensemble export
    for m in range(args.ensemble):
        print("\n" + "="*90)
        print(f"Ensemble member {m+1}/{args.ensemble}")
        print("="*90)

        set_seed(args.seed + m)  # different init/shuffle per member

        model = QuatVelAttNet(qwidth=args.qwidth, dropout_p=args.dropout).to(device)

        cfg = TrainCfg(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            dt=args.dt,
            dropout_p=args.dropout,
            qwidth=args.qwidth,
            device=device,
            loss_w=LossWeights(mse=1.0, trans=1.0, relrot=1.0, unit=0.01),
        )

        member_dir = os.path.join(args.out, f"member_{m:02d}")
        os.makedirs(member_dir, exist_ok=True)

        # Save config used
        with open(os.path.join(member_dir, "config.json"), "w") as f:
            json.dump({
                "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                "dt": cfg.dt, "dropout_p": cfg.dropout_p, "qwidth": cfg.qwidth,
                "device": cfg.device, "seed": args.seed + m
            }, f, indent=2)

        # Train
        hist = train_one_model(model, train_loader, val_loader, cfg)
        all_hists.append(hist)

        # Save model checkpoint (wrap in dict for robust loading)
        best_pth = os.path.join(member_dir, "model.pth")
        torch.save({"state_dict": model.state_dict()}, best_pth)
        member_ckpts.append(best_pth)

        # Save per-member history
        with open(os.path.join(member_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)

        # Plots for this member
        plot_training_curves(hist, out_dir=member_dir)

        # ---------- robust TorchScript export for this member ----------
        model_cpu = model.eval().to("cpu")
        member_ts_dir = os.path.join(member_dir, "ts")
        os.makedirs(member_ts_dir, exist_ok=True)
        member_pt_path = os.path.join(member_ts_dir, f"member_{m:02d}.pt")

        example = torch.zeros(1, args.seq, 8, dtype=torch.float32)  # [B,T,8]

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

        # Also copy into the flat collection folder for the C++ MEKF loader
        shutil.copyfile(member_pt_path, os.path.join(ts_collect_dir, f"member_{m:02d}.pt"))


    # Ensemble-mean plots
    if args.ensemble > 1:
        ens_hist = average_histories(all_hists)
        with open(os.path.join(args.out, "ensemble_history_mean.json"), "w") as f:
            json.dump(ens_hist, f, indent=2)
        plot_training_curves(ens_hist, out_dir=args.out)
        print(f"\nEnsemble mean curves saved to {args.out}")

    # ---------- export a single ensemble_stateless.pt (mean of members) ----------
    class Ensemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)
        def forward(self, x):
            ys = [m(x) for m in self.members]           # each [B,T,7]
            y  = torch.stack(ys, dim=0).mean(0)         # [B,T,7]
            return y

    members = []
    for pth in member_ckpts:
        if not os.path.isfile(pth):
            print(f"[warn] missing checkpoint: {pth} (skipping)")
            continue
        m = QuatVelAttNet(qwidth=args.qwidth, dropout_p=args.dropout).eval().to("cpu")
        ckpt = torch.load(pth, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        m.load_state_dict(state, strict=True)
        members.append(m)

    if len(members) >= 1:
        ens = Ensemble(members).eval().to("cpu")
        ens_pt_path = os.path.join(args.out, "ensemble_stateless.pt")
        example = torch.zeros(1, args.seq, 8, dtype=torch.float32)

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

        # Copy to flat collection folder
        shutil.copyfile(ens_pt_path, os.path.join(ts_collect_dir, "ensemble_stateless.pt"))
    else:
        print("[export ensemble] No members available to export an ensemble.")

if __name__ == "__main__":
    main()

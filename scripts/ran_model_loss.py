# -*- coding: utf-8 -*-
"""
ran_model_loss.py

RAN-style translational physics residual for v9:
- Uses BODY-frame τ = [τ_X, τ_Y, τ_N] at CO (NO rotation of τ).
- Uses custom END convention for rotations (BODY→END).
- Uses ω_b = [p,q,r] from q-observer (w_est) to include Coriolis terms.
- Payload/trim are removed. Heave/z not forced (no τ_Z given), but we keep w dynamics with damping + Coriolis coupling.

Loss logic (pairwise t -> t+1):
  1) From model's predicted v^n_pred[t], rotate to BODY with R_bn(q_t) to get [u,v,w]_t^pred.
  2) Compute du/dt, dv/dt, dw/dt from a light RAN-like model:
       m_eff * (du/dt - v*r + w*q) = τ_X - D_u(u)
       m_eff * (dv/dt - w*p + u*r) = τ_Y - D_v(v)
       m_eff * (dw/dt - u*q + v*p) =     - D_w(w)     (no τ_Z available)
     where D_* are linear (with small non-linear option).
  3) Integrate BODY velocities to t+1 (Euler): v_b_phys[t+1] = v_b_pred[t] + dv_b/dt * dt
  4) Rotate to END with q_{t+1}: v_n_phys[t+1] = R_nb(q_{t+1}) · v_b_phys[t+1]
  5) Physics loss = MSE( v_n_pred[t+1], v_n_phys[t+1] ) averaged over pairs.

This keeps the loss simple, stable, and uses ω_b correctly.
"""

from typing import Optional
import torch


# ---------------- Custom END rotation helpers ----------------

def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=eps)

def rnb_from_quat_custom_torch(q_in: torch.Tensor) -> torch.Tensor:
    """
    q_in: [...,4] (w,x,y,z)
    returns R_nb: [...,3,3] using your custom END mapping:

      Row 1 (E): [ 2(xy+wz),  1-2(xx+zz),  2(yz-wx) ]
      Row 2 (N): [ 1-2(yy+zz), 2(xy-wz),   2(xz+wy) ]
      Row 3 (D): [ 2(xz-wy),   2(yz+wx),   1-2(xx+yy)]
    """
    q = quat_normalize(q_in)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # Build matrix
    R = torch.empty(q.shape[:-1] + (3,3), dtype=q.dtype, device=q.device)
    # Row E
    R[...,0,0] = 2.0*(xy + wz)
    R[...,0,1] = 1.0 - 2.0*(xx + zz)
    R[...,0,2] = 2.0*(yz - wx)
    # Row N
    R[...,1,0] = 1.0 - 2.0*(yy + zz)
    R[...,1,1] = 2.0*(xy - wz)
    R[...,1,2] = 2.0*(xz + wy)
    # Row D
    R[...,2,0] = 2.0*(xz - wy)
    R[...,2,1] = 2.0*(yz + wx)
    R[...,2,2] = 1.0 - 2.0*(xx + yy)
    return R

def rotate_body_to_nav(q: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
    """v^n = R_nb(q) · v^b  (custom END). Shapes [...,4], [...,3] -> [...,3]."""
    R = rnb_from_quat_custom_torch(q)           # [...,3,3]
    return torch.einsum("...ij,...j->...i", R, v_b)

def rotate_nav_to_body(q: torch.Tensor, v_n: torch.Tensor) -> torch.Tensor:
    """v^b = R_bn(q) · v^n = R_nb(q)^T · v^n  (custom END)."""
    R = rnb_from_quat_custom_torch(q)           # [...,3,3]
    Rt = R.transpose(-1, -2)
    return torch.einsum("...ij,...j->...i", Rt, v_n)


# ---------------- Physics residual (pairwise) ----------------

def ran_translational_loss_v9(
    vn_pred_bt3: torch.Tensor,     # [B,T,3] predicted [vE,vN,vD]
    x_raw_bt10: torch.Tensor,      # [B,T,10] real units (ax..az, qw..qz, tau_x,y,n)
    dt: float,
    omega_b_bt3: Optional[torch.Tensor] = None # [B,T,3] BODY rates (w_est = [p,q,r])
) -> torch.Tensor:
    """
    Lightweight 3-DOF translational step (uses τ_X, τ_Y in BODY; ω_b enters Coriolis).
    """
    B, T, _ = vn_pred_bt3.shape
    if T < 2:
        return vn_pred_bt3.new_tensor(0.0)

    device = vn_pred_bt3.device
    dtype  = vn_pred_bt3.dtype

    # Inputs split (real units)
    axayaz = x_raw_bt10[..., 0:3]              # not used directly (we use τ + dynamics)
    q_bt4  = x_raw_bt10[..., 3:7]              # quaternion BODY->END
    tau_b  = x_raw_bt10[..., 7:10]             # [tau_x, tau_y, tau_n] BODY @ CO

    # Angular rates ω_b = [p,q,r] (BODY)
    if omega_b_bt3 is None:
        omega_b_bt3 = torch.zeros((B, T, 3), dtype=dtype, device=device)
    pqr = omega_b_bt3

    # Vessel constants (RAN-style, simplified)
    m = 850.0
    # Added mass (surge only modest); fold into effective mass per axis
    Xudot = -0.1 * m
    Yvdot = -1.5 * m
    Zwdot = -1.0 * m
    m_eff_u = m - Xudot  # note Xudot is negative; m - (-) = m+|Xudot|
    m_eff_v = m - Yvdot
    m_eff_w = m - Zwdot

    # Linear damping from time constants (like RAN)
    T_surge = 1.5
    T_sway  = 2.0
    T_heave = 2.0
    Xu = - m_eff_u / T_surge
    Yv = - m_eff_v / T_sway
    Zw = - m_eff_w / T_heave

    # Pairwise t -> t+1
    vn_t   = vn_pred_bt3[:, :-1, :]    # [B,T-1,3]
    vn_tp1 = vn_pred_bt3[:, 1:,  :]    # [B,T-1,3]
    q_t    = q_bt4[:, :-1, :]          # [B,T-1,4]
    q_tp1  = q_bt4[:, 1:,  :]          # [B,T-1,4]
    tau_t  = tau_b[:,  :-1, :]         # [B,T-1,3] BODY
    pqr_t  = pqr[:,    :-1, :]         # [B,T-1,3] BODY rates

    # Rotate predicted END velocity back to BODY at time t -> v_b^pred(t)
    vb_t = rotate_nav_to_body(q_t, vn_t)   # [B,T-1,3] = [u,v,w] at time t

    u_t = vb_t[..., 0]
    v_t = vb_t[..., 1]
    w_t = vb_t[..., 2]

    p_t = pqr_t[..., 0]
    q_t_ = pqr_t[..., 1]
    r_t = pqr_t[..., 2]

    tau_x = tau_t[..., 0]
    tau_y = tau_t[..., 1]
    # tau_n = tau_t[..., 2]   # not used in pure translation here

    # Rigid-body Coriolis terms (translational):
    #   m(du/dt - v*r + w*q) = τ_X - D_u(u)
    #   m(dv/dt - w*p + u*r) = τ_Y - D_v(v)
    #   m(dw/dt - u*q + v*p) =     - D_w(w)
    # Effective masses per axis (include added mass)
    du_dt = (tau_x - Xu * u_t + m * (v_t * r_t - w_t * q_t_)) / m_eff_u
    dv_dt = (tau_y - Yv * v_t + m * (w_t * p_t - u_t * r_t)) / m_eff_v
    dw_dt = (           - Zw * w_t + m * (u_t * q_t_ - v_t * p_t)) / m_eff_w

    # Integrate BODY velocities to t+1 (Euler)
    u_tp1_phys = u_t + du_dt * dt
    v_tp1_phys = v_t + dv_dt * dt
    w_tp1_phys = w_t + dw_dt * dt
    vb_tp1_phys = torch.stack([u_tp1_phys, v_tp1_phys, w_tp1_phys], dim=-1)  # [B,T-1,3]

    # Rotate to END with q_{t+1}
    vn_tp1_phys = rotate_body_to_nav(q_tp1, vb_tp1_phys)   # [B,T-1,3]

    # Physics residual compares model's next-step END velocity to physics-propagated END velocity
    resid = vn_tp1 - vn_tp1_phys
    return (resid * resid).mean()

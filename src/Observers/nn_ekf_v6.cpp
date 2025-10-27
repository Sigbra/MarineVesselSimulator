#include "Observers/nn_ekf_v6.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <yaml-cpp/yaml.h>

// Torch only in this .cpp (kept out of main and the header)
#include <torch/script.h>
#include <torch/torch.h>   // for torch::cuda::is_available, kCPU/kCUDA, NoGradGuard

#include <Eigen/Eigenvalues>  // for Eigen::SelfAdjointEigenSolver

namespace mekf {

// ----------------- small local helpers (not touching class private) -----------------
static inline Eigen::Matrix3d skew3(const Eigen::Vector3d& a) {
  Eigen::Matrix3d S;
  S <<    0,   -a.z(),  a.y(),
        a.z(),    0,   -a.x(),
       -a.y(),  a.x(),    0;
  return S;
}

// Right Jacobian on SO(3) (free function version)
// static inline Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& phi) {
//   const double th = phi.norm();
//   const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//   if (th < 1e-8) {
//     const Eigen::Matrix3d A = skew3(phi);
//     return I - 0.5*A + (1.0/12.0)*(A*A);
//   }
//   const Eigen::Matrix3d A = skew3(phi);
//   const double th2 = th*th;
//   const double s = std::sin(th), c = std::cos(th);
//   const double a = (1.0 - c) / th2;
//   const double b = (th - s) / (th2*th);
//   return I - a*A + b*(A*A);
// }

// Inverse of Left Jacobian (free function)
// static inline Eigen::Matrix3d Jr_SO3(const Eigen::Vector3d& phi) {
//   const double th = phi.norm();
//   const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//   if (th < 1e-8) {
//     const Eigen::Matrix3d A = skew3(phi);
//     return I + 0.5*A + (1.0/12.0)*(A*A);
//   }
//   const Eigen::Matrix3d A = skew3(phi);
//   const double th2 = th*th;
//   const double s = std::sin(th), c = std::cos(th);
//   const double a = 0.5;
//   const double b = (1.0 - th*s/(2.0*(1.0 - c))) / th2;
//   return I + a*A + b*(A*A);
// }

// ---------------- Utility (class) ----------------

Eigen::Matrix3d MEKF::skew(const Eigen::Vector3d& a) {
  return skew3(a);
}

// Quaternion exponential map from rotation vector φ (axis*angle, rad)
Eigen::Quaterniond MEKF::quatExp(const Eigen::Vector3d& phi) {
  const double th = phi.norm();
  if (th < 1e-12) {
    return Eigen::Quaterniond(1.0, 0.5*phi.x(), 0.5*phi.y(), 0.5*phi.z()).normalized();
  }
  const double half = 0.5 * th;
  const double s = std::sin(half) / th;
  return Eigen::Quaterniond(std::cos(half), s*phi.x(), s*phi.y(), s*phi.z());
}

// Quaternion small-angle log-vector: returns axis*angle in R^3
Eigen::Vector3d MEKF::quatLogVec(const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = q_in.normalized();
  double w = std::clamp(q.w(), -1.0, 1.0);
  Eigen::Vector3d v(q.x(), q.y(), q.z());
  double s = v.norm();
  if (s < 1e-12) return 2.0 * v; // ~ zero
  double ang = 2.0 * std::atan2(s, w);
  return (ang / s) * v; // axis*angle
}

// SO(3) Left Jacobian: J_l(phi) = I + (1-cosθ)/θ^2 [φ] + (θ - sinθ)/θ^3 [φ]^2
Eigen::Matrix3d MEKF::leftJacobianSO3(const Eigen::Vector3d& phi) {
  const double th = phi.norm();
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < 1e-8) {
    const Eigen::Matrix3d A = skew(phi);
    return I + 0.5*A + (1.0/12.0)*(A*A);
  }
  const Eigen::Matrix3d A = skew(phi);
  const double th2 = th*th;
  const double s = std::sin(th), c = std::cos(th);
  const double a = (1.0 - c) / th2;
  const double b = (th - s) / (th2*th);
  return I + a*A + b*(A*A);
}

// ---------------- Custom END DCM builders/parsers ----------------

// ZYX (yaw=psi about +D/NED-z, pitch=theta about +E/NED-y, roll=phi about +E/NED-x)
// Rows are [E; N; D], columns are BODY axes expressed in END.
Mat3 MEKF::RnbFromEuler(double phi, double theta, double psi) {
  const double cphi = std::cos(phi), sphi = std::sin(phi);
  const double cth  = std::cos(theta), sth = std::sin(theta);
  const double cpsi = std::cos(psi),   spsi = std::sin(psi);

  Mat3 R;
  // Row 1: East
  R(0,0) =  spsi * cth;
  R(0,1) =  cpsi * cphi + sphi * sth * spsi;
  R(0,2) = -cpsi * sphi + sth * spsi * cphi;

  // Row 2: North
  R(1,0) =  cpsi * cth;
  R(1,1) = -spsi * cphi + cpsi * sth * sphi;
  R(1,2) =  spsi * sphi + cpsi * cphi * sth;

  // Row 3: Down
  R(2,0) = -sth;
  R(2,1) =  cth * sphi;
  R(2,2) =  cth * cphi;

  return R;
}


Mat3 MEKF::RnbFromQuatCustom(const Eigen::Quaterniond& q_in) {
  const Eigen::Quaterniond q = q_in.normalized();
  const double w = q.w(), x = q.x(), y = q.y(), z = q.z();
  const double xx = x*x, yy = y*y, zz = z*z;
  const double wx = w*x, wy = w*y, wz = w*z;
  const double xy = x*y, xz = x*z, yz = y*z;

  Mat3 R;
  // Row 1: East  (standard Y row)
  R(0,0) = 2.0*(xy + wz);
  R(0,1) = 1.0 - 2.0*(xx + zz);
  R(0,2) = 2.0*(yz - wx);

  // Row 2: North (standard X row)
  R(1,0) = 1.0 - 2.0*(yy + zz);
  R(1,1) = 2.0*(xy - wz);
  R(1,2) = 2.0*(xz + wy);

  // Row 3: Down  (standard Z row)
  R(2,0) = 2.0*(xz - wy);
  R(2,1) = 2.0*(yz + wx);
  R(2,2) = 1.0 - 2.0*(xx + yy);
  return R;
}

double MEKF::yawFromRnb(const Mat3& R) {
  // Heading (marine): angle from North toward East: ψ = atan2(E•xB, N•xB) = atan2(R(0,0), R(1,0))
  return std::atan2(R(0,0), R(1,0));
}

// --- Attitude accessors / dataset helpers ---

Eigen::Quaterniond MEKF::quatFromEulerDataset(double phi, double th, double psi) {
  // Build END DCM with our convention then convert to quaternion
  const Mat3 R = RnbFromEuler(phi, th, psi);
  return Eigen::Quaterniond(R).normalized();
}

double MEKF::yawFromQuatEND(const Eigen::Quaterniond& q) {
  const Mat3 R = RnbFromQuatCustom(q);
  return yawFromRnb(R);
}

Eigen::Quaterniond MEKF::attitudeQuat() const { return x_.q_nb; }
Eigen::Matrix3d    MEKF::rotationNavFromBody() const { return RnbFromQuatCustom(x_.q_nb); }

// ---------------- MEKF core ----------------

MEKF::MEKF(const Config& cfg) : cfg_(cfg) {
  P_.setIdentity();
  P_ *= 1e-2;
  nn_buf_.clear();
  nn_stride_counter_ = 0;
}

const State& MEKF::state() const { return x_; }
const Eigen::Matrix<double,15,15>& MEKF::covariance() const { return P_; }

void MEKF::setState(const State& x, const Eigen::Matrix<double,15,15>& P) {
  x_ = x;
  x_.q_nb.normalize();
  P_ = P;
  resetNNBuffer();
}

void MEKF::setNNInfer(NNInfer cb, int seq_len, int stride) {
  nn_infer_ = std::move(cb);
  if (seq_len > 0) cfg_.nn_seq_len = seq_len;
  if (stride  > 0) cfg_.nn_stride  = stride;
  resetNNBuffer();
}

void MEKF::setNN(QuatVelAttNN* nn, int seq_len, int stride) {
  if (seq_len > 0) cfg_.nn_seq_len = seq_len;
  if (stride  > 0) cfg_.nn_stride  = stride;

  if (nn) {
    setNNInfer(
      [nn](const std::vector<Eigen::Matrix<double,8,1>>& win,
           Vec3& v_mean, Eigen::Quaterniond& q_mean,
           Mat3& Rv, Mat3& Rq)->bool
      {
        return nn->infer(win, v_mean, q_mean, Rv, Rq);
      },
      cfg_.nn_seq_len, cfg_.nn_stride
    );
  } else {
    clearNNInfer();
  }
}

void MEKF::clearNNInfer() {
  nn_infer_ = nullptr;
  resetNNBuffer();
}

void MEKF::resetNNBuffer() {
  nn_buf_.clear();
  nn_stride_counter_ = 0;
}

bool MEKF::nnWindowReady() const {
  return static_cast<int>(nn_buf_.size()) >= cfg_.nn_seq_len;
}

void MEKF::pushNNSample(const Vec3& imu_omega_meas, const Vec3& imu_acc_meas) {
  // Build one 8x1 sample
  Eigen::Matrix<double,8,1> s;
  // IMU measured p,q,r, ax,ay,az  (apply sign correction for NN to match training)
  s(0) = imu_omega_meas.x() * cfg_.sign_gyro.x();
  s(1) = imu_omega_meas.y() * cfg_.sign_gyro.y();
  s(2) = imu_omega_meas.z() * cfg_.sign_gyro.z();
  s(3) = imu_acc_meas.x()   * cfg_.sign_acc.x();
  s(4) = imu_acc_meas.y()   * cfg_.sign_acc.y();
  s(5) = imu_acc_meas.z()   * cfg_.sign_acc.z();

  // cpsi,spsi from current EKF yaw (END)
  const double psi = yawFromQuatEND(x_.q_nb);
  s(6) = std::cos(psi);
  s(7) = std::sin(psi);

  // Push into deque
  nn_buf_.push_back(s);
  while (static_cast<int>(nn_buf_.size()) > cfg_.nn_seq_len) {
    nn_buf_.pop_front();
  }
}

void MEKF::buildNNWindow(std::vector<Eigen::Matrix<double,8,1>>& out) const {
  out.resize(cfg_.nn_seq_len);
  const int N = static_cast<int>(nn_buf_.size());
  const int start = N - cfg_.nn_seq_len;
  for (int i = 0; i < cfg_.nn_seq_len; ++i) {
    out[i] = nn_buf_[start + i];
  }
}

bool MEKF::maybeRunNNAndFuse() {
  if (!nn_infer_) return false;
  if (!nnWindowReady()) return false;
  if ((nn_stride_counter_++ % cfg_.nn_stride) != 0) return false;

  std::vector<Eigen::Matrix<double,8,1>> window;
  buildNNWindow(window);

  Vec3 v_mean; Eigen::Quaterniond q_mean; Mat3 Rv; Mat3 Rq;
  if (!nn_infer_(window, v_mean, q_mean, Rv, Rq)) {
    return false; // callback declined
  }

  // Fuse velocity then attitude
  updateNNVelocity(v_mean, Rv);
  updateNNAttitude(q_mean, Rq);
  return true;
}

void MEKF::propagate(const Vec3& omega_meas_body, const Vec3& acc_meas_body, double dt) {
  // Cache raw measurements for reporting p,q,r and NN features (no sign correction here)
  last_imu_omega_meas_ = omega_meas_body;
  last_imu_acc_meas_   = acc_meas_body;

  // Apply sign correction to match simulator/body frame convention
  const Vec3 omega_b = omega_meas_body.cwiseProduct(cfg_.sign_gyro) - x_.b_g;  // rad/s BODY
  const Vec3 acc_b   = acc_meas_body  .cwiseProduct(cfg_.sign_acc)  - x_.b_a;  // m/s^2 BODY (specific force)

  // Use current attitude for ω_n, then update attitude
  const Mat3 Rnb_prev = RnbFromQuatCustom(x_.q_nb);     // BODY->NAV before update
  const Vec3 omega_n  = Rnb_prev * omega_b;             // express ω in NAV
  x_.q_nb = (quatExp(omega_n * dt) * x_.q_nb).normalized();

  // Recompute R_nb AFTER attitude update and use it for translation & linearization
  const Mat3 R_nb = RnbFromQuatCustom(x_.q_nb);

  // Translational dynamics (END: g = +g in Down)
  const Vec3 acc_n = R_nb * acc_b + Vec3(0,0,cfg_.g);
  x_.v += acc_n * dt;
  x_.p += x_.v * dt + 0.5 * acc_n * dt * dt;

  // ------------- Error-state propagation (δθ in NAV frame) -------------
  Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Zero();
  Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();

  // δp_dot = δv
  F.block<3,3>(0,3).setIdentity();

  // δv_dot ≈ - R_nb [a_b]_x δθ  - R_nb δb_a  + R_nb n_a
  F.block<3,3>(3,6)  = - R_nb * skew(acc_b);
  F.block<3,3>(3,12) = - R_nb;
  G.block<3,3>(3,3)  =   R_nb;                 // accel white noise (BODY) → NAV

  // δθ_dot ≈ - [ω_n]_x δθ  - R_nb δb_g  + R_nb n_g
  F.block<3,3>(6,6)  = - skew(omega_n);
  F.block<3,3>(6,9)  = - R_nb;                 // gyro bias coupling in NAV frame
  G.block<3,3>(6,0)  =   R_nb;                 // gyro white noise (BODY) → NAV

  // Optional bias-leak terms (first-order decay)
  const double inv_tau_bg = (cfg_.tau_bg > 0.0) ? 1.0/cfg_.tau_bg : 0.0;
  const double inv_tau_ba = (cfg_.tau_ba > 0.0) ? 1.0/cfg_.tau_ba : 0.0;
  if (inv_tau_bg > 0.0) F.block<3,3>(9,9)   = -inv_tau_bg * Mat3::Identity();
  if (inv_tau_ba > 0.0) F.block<3,3>(12,12) = -inv_tau_ba * Mat3::Identity();

  // Discretize (Euler)
  const Eigen::Matrix<double,15,15> Phi = Eigen::Matrix<double,15,15>::Identity() + F * dt;

  // Continuous noise
  Eigen::Matrix<double,12,12> Qc = Eigen::Matrix<double,12,12>::Zero();
  Qc.block<3,3>(0,0)  = (cfg_.sigma_g     * cfg_.sigma_g)     * Mat3::Identity();
  Qc.block<3,3>(3,3)  = (cfg_.sigma_a     * cfg_.sigma_a)     * Mat3::Identity();
  Qc.block<3,3>(6,6)  = (cfg_.sigma_bg_rw * cfg_.sigma_bg_rw) * Mat3::Identity();
  Qc.block<3,3>(9,9)  = (cfg_.sigma_ba_rw * cfg_.sigma_ba_rw) * Mat3::Identity();

  const Eigen::Matrix<double,15,15> Qd = (G * Qc * G.transpose()) * dt;
  P_ = Phi * P_ * Phi.transpose() + Qd;

  // NN window & fusion (training-consistent)
  pushNNSample(omega_meas_body, acc_meas_body);
  (void)maybeRunNNAndFuse();

  // Bias clamping (optional)
  if (cfg_.clamp_bias) {
    x_.b_g = x_.b_g.cwiseMax(-cfg_.bias_limit_gyro).cwiseMin(cfg_.bias_limit_gyro);
    x_.b_a = x_.b_a.cwiseMax(-cfg_.bias_limit_acc ).cwiseMin(cfg_.bias_limit_acc );
  }
}

bool MEKF::updateGnssPos(const Vec3& z_nav, const Mat3& Rpos, const Vec3& r_body, double w)
{
  // Model: z ≈ p + R_nb * r_body  (END)
  if (!std::isfinite(w) || w <= 0.0) w = 1.0;

  const Mat3 Rnb = RnbFromQuatCustom(x_.q_nb);

  // Residual
  const Vec3 h = x_.p + Rnb * r_body;
  const Vec3 r = z_nav - h;

  // Jacobian wrt error-state [p(0:2), v(3:5), θ(6:8), b_g(9:11), b_a(12:14)]
  Mat<3,15> H = Mat<3,15>::Zero();
  H.block<3,3>(0, 0) = Mat3::Identity();               // d/dp
  H.block<3,3>(0, 6) = - Rnb * skew(r_body);           // d(R*r)/dδθ ≈ -R*[r]_x

  // (Optional) gating
  if (!gate(r, H, Rpos / w, cfg_.chi2_gate_pos3)) return false;

  // Kalman update via helper
  kalmanUpdate(r, H, Rpos / w);
  return true;
}

bool MEKF::updateGnssVel(const Vec3& z_v_nav,
                         const Mat3& Rvel,
                         const Vec3& r_body,     // GNSS antenna lever arm in BODY
                         const Vec3& omega_meas_body, // raw IMU (BODY)
                         double w)
{
  if (!std::isfinite(w) || w <= 0.0) w = 1.0;

  // Bias-corrected body rate (same as in propagate)
  const Vec3 omega_b = omega_meas_body.cwiseProduct(cfg_.sign_gyro) - x_.b_g;

  const Mat3 Rnb = RnbFromQuatCustom(x_.q_nb);

  // Predicted antenna velocity in END
  const Vec3 v_ant_pred = x_.v + Rnb * (omega_b.cross(r_body));

  // Residual
  const Vec3 r = z_v_nav - v_ant_pred;

  // H: only v, theta (NAV), and b_g are active
  Mat<3,15> H = Mat<3,15>::Zero();
  H.block<3,3>(0,3)  = Mat3::Identity();                 // d/dv
  H.block<3,3>(0,6)  = - Rnb * skew(omega_b.cross(r_body)); // d/dθ (NAV)
  H.block<3,3>(0,9)  =   Rnb * skew(r_body);             // d/db_g

  // Gate and update
  if (!gate(r, H, Rvel / w, cfg_.chi2_gate_vec3)) return false;

  kalmanUpdate(r, H, Rvel / w);
  return true;
}

bool MEKF::updateGnss(const Vec3& z_nav_ant,
                      const Mat3& Rpos,
                      const Vec3& r_body,
                      double w)
{
  bool any_ok = false;

  // 1) Position update (antenna position):  z ≈ p + R_nb * r_body
  any_ok |= updateGnssPos(z_nav_ant, Rpos, r_body, w);

  // 2) Pseudo-Doppler velocity from position differencing (optional)
  //    z_v ≈ (z_k - z_{k-1}) / Δt   with covariance ≈ (Rk + Rk-1)/Δt^2 + qI
  if (cfg_.use_gnss_pseudo_velocity && cfg_.gnss_dt_hint > 0.0 && has_prev_gnss_pos_) {
    const double dt = cfg_.gnss_dt_hint;

    // Build velocity measurement in END (antenna point)
    const Vec3 z_v_nav = (z_nav_ant - prev_gnss_pos_nav_) / dt;

    // Propagate covariance to velocity domain
    Mat3 Rvel = (Rpos + prev_Rpos_) / (dt * dt)
                + (cfg_.gnss_vel_deriv_q) * Mat3::Identity();

    // Use latest raw IMU body rate already cached by propagate()
    // and the same lever arm r_body for the antenna velocity model.
    any_ok |= updateGnssVel(z_v_nav, Rvel, r_body, last_imu_omega_meas_, w);
  }

  // 3) Bookkeeping for next call
  prev_gnss_pos_nav_ = z_nav_ant;
  prev_Rpos_         = Rpos;
  has_prev_gnss_pos_ = true;

  return any_ok;
}


bool MEKF::updateGnssBaseline(const Vec3& z_nav, const Mat3& Rz, const Vec3& b_body, double w)
{
  // Model: z ≈ R_nb * b_body  (END)
  if (!std::isfinite(w) || w <= 0.0) w = 1.0;

  const Mat3 Rnb = RnbFromQuatCustom(x_.q_nb);

  // Residual
  const Vec3 h = Rnb * b_body;
  const Vec3 r = z_nav - h;

  // Only attitude columns are non-zero (δθ is in NAV frame)
  Mat<3,15> H = Mat<3,15>::Zero();
  H.block<3,3>(0, 6) = - Rnb * skew(b_body);  // d(R*b)/dδθ ≈ -R*[b]_x

  // (Optional) gating
  if (!gate(r, H, Rz / w, cfg_.chi2_gate_vec3)) return false;

  // Kalman update via helper
  kalmanUpdate(r, H, Rz / w);

#ifdef MEKF_DEBUG_BASELINE
  // Quick sanity print: angle between measured and predicted baseline
  const double dot = std::max(-1.0, std::min(1.0, (z_nav.normalized()).dot(h.normalized())));
  const double ang_deg = std::acos(dot) * 180.0 / M_PI;
  if (ang_deg > 30.0) {
      std::cerr << "[MEKF] Baseline angle mismatch: " << ang_deg
                << " deg (check ant order and R_nb)\n";
  }
#endif
  return true;
}

bool MEKF::updateNNVelocity(const Vec3& z_v_n, const Mat3& Rv, double gate_chi2) {
  const Vec3 h = x_.v;
  const Vec3 r = z_v_n - h;

  Mat<3,15> H = Mat<3,15>::Zero();
  H.block<3,3>(0,3) = Mat3::Identity();

  if (!gate(r, H, Rv, (gate_chi2>0?gate_chi2:cfg_.chi2_gate_vec3))) return false;

  kalmanUpdate(r, H, Rv);
  return true;
}

bool MEKF::updateNNAttitude(const Eigen::Quaterniond& q_nn, const Mat3& Rq, double gate_chi2) {
  // Residual in small-angle domain: r = log( q_nn ⊗ q_hat^{-1} )
  const Eigen::Quaterniond dq = q_nn * x_.q_nb.conjugate();
  const Vec3 r = quatLogVec(dq); // small-angle residual

  Mat<3,15> H = Mat<3,15>::Zero();
  H.block<3,3>(0,6) = Mat3::Identity();

  if (!gate(r, H, Rq, (gate_chi2>0?gate_chi2:cfg_.chi2_gate_vec3))) return false;

  kalmanUpdate(r, H, Rq);
  return true;
}

void MEKF::kalmanUpdate(const Eigen::Vector3d& r,
                        const Eigen::Matrix<double,3,15>& H,
                        const Mat3& R)
{
  // S, K
  const Eigen::Matrix3d S = (H * P_ * H.transpose()) + R;
  const Eigen::Matrix<double,15,3> K = P_ * H.transpose() * S.inverse();

  // δx
  const Eigen::Matrix<double,15,1> dx = K * r;

  // Inject nominal state
  x_.p  += dx.segment<3>(0);
  x_.v  += dx.segment<3>(3);
  const Eigen::Vector3d dtheta = dx.segment<3>(6);
  x_.q_nb = (quatExp(dtheta) * x_.q_nb).normalized(); // multiplicative correction

  if (cfg_.estimate_bias) {
    x_.b_g += dx.segment<3>(9);
    x_.b_a += dx.segment<3>(12);
  }

  // Covariance update (Joseph)
  josephUpdate(P_, K, H, R);

  // Exact covariance reset with Γ = blkdiag(I, I, J_l(dθ), I, I)
  const Eigen::Matrix<double,15,15> Gamma = resetGamma(dtheta);
  P_ = Gamma * P_ * Gamma.transpose();

  // Optional bias clamp
  if (cfg_.clamp_bias) {
    x_.b_g = x_.b_g.cwiseMax(-cfg_.bias_limit_gyro).cwiseMin(cfg_.bias_limit_gyro);
    x_.b_a = x_.b_a.cwiseMax(-cfg_.bias_limit_acc ).cwiseMin(cfg_.bias_limit_acc );
  }
}

bool MEKF::gate(const Eigen::Vector3d& r,
                const Eigen::Matrix<double,3,15>& H,
                const Mat3& R,
                double chi2_thresh) const
{
  if (chi2_thresh <= 0.0) return true;
  const Eigen::Matrix3d S = (H * P_ * H.transpose()) + R;
  const double nis = r.transpose() * S.inverse() * r;
  return nis < chi2_thresh;
}

void MEKF::josephUpdate(Eigen::Matrix<double,15,15>& P,
                        const Eigen::Matrix<double,15,3>& K,
                        const Eigen::Matrix<double,3,15>& H,
                        const Mat3& R)
{
  const Eigen::Matrix<double,15,15> I = Eigen::Matrix<double,15,15>::Identity();
  const Eigen::Matrix<double,15,15> IKH = (I - K*H);
  P = IKH * P * IKH.transpose() + K * R * K.transpose();
}

Eigen::Matrix<double,15,15> MEKF::resetGamma(const Eigen::Vector3d& dtheta) {
  Eigen::Matrix<double,15,15> G = Eigen::Matrix<double,15,15>::Identity();
  G.block<3,3>(6,6) = leftJacobianSO3(dtheta); // exact mapping for attitude error
  return G;
}

// -------- getState12: [u v w p q r x y z phi theta psi]^T --------
Eigen::Matrix<double,12,1> MEKF::getState12() const {
  Eigen::Matrix<double,12,1> x12;

  // Body-frame linear velocity from nav velocity
  const Mat3 R_nb = RnbFromQuatCustom(x_.q_nb);  // body->nav
  const Vec3 v_b  = R_nb.transpose() * x_.v;

  // Bias-corrected angular rates (best estimate of body rates), apply sign to match simulator body axes
  const Vec3 omega_est =
      last_imu_omega_meas_.cwiseProduct(cfg_.sign_gyro) - x_.b_g;

  // ZYX Euler from our R_nb
  const double r20 = R_nb(2,0);
  const double r21 = R_nb(2,1);
  const double r22 = R_nb(2,2);
  const double r10 = R_nb(1,0);
  const double r00 = R_nb(0,0);

  const double s = std::clamp(r20, -1.0, 1.0);
  const double theta = -std::asin(s);
  const double phi   = std::atan2(r21, r22);
  const double psi   = std::atan2(r00, r10);  // heading from North toward East

  x12 << v_b.x(), v_b.y(), v_b.z(),                // u, v, w  (body)
          omega_est.x(), omega_est.y(), omega_est.z(), // p, q, r (body, bias-corrected)
          x_.p.x(), x_.p.y(), x_.p.z(),            // x, y, z  (END)
          phi, theta, psi;                         // phi, theta, psi (ZYX)

  return x12;
}

// ---------------- QuatVelAttNN (TorchScript ensemble) ----------------

struct QuatVelAttNN::Impl {
  bool use_cuda_ = false;
  torch::Device device_{torch::kCPU};
  std::vector<torch::jit::script::Module> models_;

  std::array<double,8> x_mean_{}, x_std_{};
  std::array<double,3> y_mean_{}, y_std_{};
  torch::Tensor x_mean_t_, x_std_t_, y_mean_t_, y_std_t_;

  static Eigen::Vector3d quatLogVecLocal(const Eigen::Quaterniond& q_in) {
    Eigen::Quaterniond q = q_in.normalized();
    double w = std::clamp(q.w(), -1.0, 1.0);
    Eigen::Vector3d v(q.x(), q.y(), q.z());
    double s = v.norm();
    if (s < 1e-12) return 2.0 * v;
    double ang = 2.0 * std::atan2(s, w);
    return (ang / s) * v;
  }

  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda)
  {
    #ifdef TORCH_CUDA_AVAILABLE
    use_cuda_ = use_cuda && torch::cuda::is_available();
    #else
    use_cuda_ = false;
    #endif
    device_ = use_cuda_ ? torch::kCUDA : torch::kCPU;

    // ---------- Load normalization (JSON is valid YAML) ----------
    YAML::Node ns;
    try {
        namespace fs = std::filesystem;
        fs::path norm_path(norm_json);
    #ifdef MVS_PROJECT_ROOT
        if (!norm_path.is_absolute()) {
          fs::path alt = fs::path(MVS_PROJECT_ROOT) / norm_path;
          if (fs::exists(alt)) norm_path = alt;
        }
    #endif
        ns = YAML::LoadFile(norm_path.string());
    } catch (const YAML::BadFile& e) {
        std::cerr << "[QuatVelAttNN] Failed to open norm_stats: " << norm_json << "\n";
        return false;
    }

    auto x_mean_v = ns["x_mean"]; auto x_std_v = ns["x_std"];
    auto y_mean_v = ns["y_mean"]; auto y_std_v = ns["y_std"];
    if (!x_mean_v || !x_std_v || !y_mean_v || !y_std_v) {
        std::cerr << "[QuatVelAttNN] norm_json missing keys x_mean/x_std/y_mean/y_std\n";
        return false;
    }
    for (int i=0;i<8;++i)  { x_mean_[i] = x_mean_v[i].as<double>(); x_std_[i] = x_std_v[i].as<double>(); }
    for (int i=0;i<3;++i)  { y_mean_[i] = y_mean_v[i].as<double>(); y_std_[i] = y_std_v[i].as<double>(); }

    x_mean_t_ = torch::from_blob(x_mean_.data(), {8}, torch::kDouble).clone().to(device_);
    x_std_t_  = torch::from_blob(x_std_.data(),  {8}, torch::kDouble).clone().to(device_);
    y_mean_t_ = torch::from_blob(y_mean_.data(), {3}, torch::kDouble).clone().to(device_);
    y_std_t_  = torch::from_blob(y_std_.data(),  {3}, torch::kDouble).clone().to(device_);

    // ---------- Resolve model path (file OR directory) ----------
    namespace fs = std::filesystem;
    fs::path md(model_dir);
    #ifdef MVS_PROJECT_ROOT
    if (!md.is_absolute()) {
        fs::path alt = fs::path(MVS_PROJECT_ROOT) / md;
        if (fs::exists(alt)) md = alt;
    }
    #endif
    if (!fs::exists(md)) {
        std::cerr << "[QuatVelAttNN] model path does not exist: " << md << "\n";
        return false;
    }

    auto is_model_ext = [](const fs::path& p){
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return (ext == ".pt" || ext == ".ts" || ext == ".pth"); // (pth only if it’s a scripted archive; state_dict won't load)
    };

    // Collect **member_XX.pt** files first (preferred for covariance).
    std::vector<std::string> files;

    if (fs::is_regular_file(md)) {
        if (is_model_ext(md)) files.push_back(md.string());
    } else if (fs::is_directory(md)) {
        // First pass: strictly member_*.pt
        for (const auto& e : fs::directory_iterator(md)) {
          if (!e.is_regular_file()) continue;
          const auto& p = e.path();
          if (!is_model_ext(p)) continue;
          const auto name = p.filename().string();
          if (name.rfind("member_", 0) == 0) { // starts with "member_"
              files.push_back(p.string());
          }
        }
        // Fallback: if no member_*.pt found, take any *.pt but **exclude** ensemble_* to avoid double-counting mean
        if (files.empty()) {
          for (const auto& e : fs::directory_iterator(md)) {
            if (!e.is_regular_file()) continue;
            const auto& p = e.path();
            if (!is_model_ext(p)) continue;
            const auto name = p.filename().string();
            if (name.find("ensemble") != std::string::npos) continue; // skip ensemble*
            files.push_back(p.string());
          }
        }
    }

    if (files.empty()) {
        std::cerr << "[QuatVelAttNN] No TorchScript members found at: " << md << "\n";
        if (fs::is_directory(md)) {
          std::cerr << "[QuatVelAttNN] Directory listing:\n";
          for (const auto& e : fs::directory_iterator(md))
              std::cerr << "  - " << e.path().filename().string() << "\n";
        }
        return false;
    }

    std::sort(files.begin(), files.end());

    // ---------- Load models ----------
    try {
        models_.clear();
        models_.reserve(files.size());
        for (const auto& f : files) {
          auto m = torch::jit::load(f, device_);
          m.eval();
          models_.push_back(std::move(m));
          std::cerr << "[QuatVelAttNN] Loaded " << f << (use_cuda_ ? " (CUDA)\n" : " (CPU)\n");
        }
    } catch (const c10::Error& e) {
        std::cerr << "[QuatVelAttNN] Failed to load model: " << e.what() << "\n";
        return false;
    }

    if (models_.size() == 1) {
        std::cerr << "[QuatVelAttNN] WARNING: only 1 model loaded; covariance will be near floor.\n";
    } else {
        std::cerr << "[QuatVelAttNN] Ensemble size: " << models_.size() << " (covariance from member spread)\n";
    }
    return true;
  }

  bool infer(const std::vector<Eigen::Matrix<double,8,1>>& window,
             Vec3& v_mean, Eigen::Quaterniond& q_mean,
             Mat3& Rv, Mat3& Rq)
  {
    if (models_.empty()) return false;
    const int T = static_cast<int>(window.size());
    if (T <= 0) return false;

    torch::NoGradGuard ng;

    // Build input tensor [1, T, 8] (double for precision; cast to float for model if needed)
    torch::Tensor x = torch::empty({1, T, 8}, torch::TensorOptions().dtype(torch::kDouble).device(device_));
    {
      auto xa = x.accessor<double,3>();
      for (int t=0; t<T; ++t)
        for (int j=0; j<8; ++j) xa[0][t][j] = window[t](j);
    }
    // Normalize
    x = (x - x_mean_t_.view({1,1,8})) / x_std_t_.view({1,1,8});

    const int M = static_cast<int>(models_.size());
    std::vector<Eigen::Vector3d> vel_list; vel_list.reserve(M);
    std::vector<Eigen::Quaterniond> quat_list; quat_list.reserve(M);

    for (int m=0; m<M; ++m) {
      // Forward: expect [1, T, 7] = [vE,vN,vD,qw,qx,qy,qz]
      auto out_any = models_[m].forward({ x.to(torch::kFloat) });
      torch::Tensor y = out_any.toTensor();
      if (y.dim()!=3 || y.size(0)!=1 || y.size(2)!=7) {
        std::cerr << "[QuatVelAttNN] Bad output shape from model m="<<m<<": " << y.sizes() << "\n";
        return false;
      }
      torch::Tensor last = y.index({0, T-1});            // [7]
      auto ycpu = last.to(torch::kDouble).to(torch::kCPU);
      std::array<double,7> tmp{};
      std::memcpy(tmp.data(), ycpu.data_ptr<double>(), 7*sizeof(double));

      // De-normalize velocities
      Eigen::Vector3d v_nav;
      for (int k=0;k<3;++k) v_nav(k) = tmp[k] * y_std_[k] + y_mean_[k];
      vel_list.push_back(v_nav);

      // Quaternion normalized
      Eigen::Quaterniond q(tmp[3], tmp[4], tmp[5], tmp[6]);
      q.normalize();
      quat_list.push_back(q);
    }

    // Quaternion mean (Markley) with sign alignment to first member
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    Eigen::Vector4d qref(quat_list[0].w(), quat_list[0].x(), quat_list[0].y(), quat_list[0].z());
    for (int i=0;i<M;++i) {
      Eigen::Vector4d qi(quat_list[i].w(), quat_list[i].x(), quat_list[i].y(), quat_list[i].z());
      if (qi.dot(qref) < 0.0) qi = -qi;
      A += qi * qi.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(A);
    Eigen::Vector4d qv = es.eigenvectors().col(3);
    q_mean = Eigen::Quaterniond(qv(0), qv(1), qv(2), qv(3)).normalized();

    // Velocity mean
    v_mean.setZero();
    for (const auto& v : vel_list) v_mean += v;
    v_mean /= static_cast<double>(M);

    // Sample covariance over velocities (positive-definite with a small floor)
    Eigen::Matrix<double,3,Eigen::Dynamic> Dv(3, M);
    for (int i=0;i<M;++i) Dv.col(i) = vel_list[i] - v_mean;
    Rv = (Dv * Dv.transpose()) / std::max(1, M-1);
    Rv += 1e-5 * Eigen::Matrix3d::Identity();

    // Small-angle covariance around q_mean from ensemble spread
    Eigen::Matrix<double,3,Eigen::Dynamic> Dq(3, M);
    for (int i=0;i<M;++i) {
      Eigen::Quaterniond dq = quat_list[i] * q_mean.conjugate();
      Eigen::Vector3d r = quatLogVecLocal(dq); // axis*angle
      Dq.col(i) = r;
    }
    Rq = (Dq * Dq.transpose()) / std::max(1, M-1);
    Rq += 1e-6 * Eigen::Matrix3d::Identity();

    return true;
  }
};

// PIMPL forwarding
QuatVelAttNN::QuatVelAttNN() : impl_(std::make_unique<Impl>()) {}
QuatVelAttNN::~QuatVelAttNN() = default;
QuatVelAttNN::QuatVelAttNN(QuatVelAttNN&&) noexcept = default;
QuatVelAttNN& QuatVelAttNN::operator=(QuatVelAttNN&&) noexcept = default;

bool QuatVelAttNN::init(const std::string& model_dir,
                        const std::string& norm_json,
                        bool use_cuda) {
  return impl_->init(model_dir, norm_json, use_cuda);
}

bool QuatVelAttNN::infer(const std::vector<Eigen::Matrix<double,8,1>>& window,
                         Vec3& v_mean,
                         Eigen::Quaterniond& q_mean,
                         Mat3& Rv,
                         Mat3& Rq) {
  return impl_->infer(window, v_mean, q_mean, Rv, Rq);
}

} // namespace mekf



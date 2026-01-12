#pragma once

#include "Utilities/calculations.hpp"   // ::Quat, ::Vec3, ::Mat3, RnbFromQuatCustom

namespace qobs {

using ::Quat;
using ::Vec3;
using ::Mat3;

//-------------------------------------------------------------------------
// Configuration
//-------------------------------------------------------------------------
struct Config {
  // MATLAB: Ki (often diagonal, but keep general 3x3)
  Mat3   Ki = Mat3::Identity() * 1e-3;

  // MATLAB: gains
  double k1 = 0.9;   // specific-force injection gain
  double k2 = 1.2;   // heading / magnetic injection gain

  // MATLAB 9-DOF only: magnetic reference vector expressed in END
  Vec3   m_ref_end = Vec3(1.0, 0.0, 0.0);

  // Numerical guards (MATLAB normalizes without guards; we guard against /0)
  double accel_min_norm = 1e-12;
  double mag_min_norm   = 1e-12;
};

//-------------------------------------------------------------------------
// QuatObserver
//
// END convention:
//   NAV/END: x East, y North, z Down
//   BODY   : x forward, y starboard, z Down
//
// MATLAB-equivalent observer (Grip et al.) in discrete time:
//   sigma = k1 * v1 x (R_bn * v01) + k2 * v2 x (R_bn * v02)
//   w_est = w_imu - b + sigma
//   q     = expm( Tquat(w_est) * dt ) * q
//   b     = b - dt * Ki * sigma
//
// Mode behavior matching your MATLAB file:
//   step6DOF: sigma = 0 (predictor only)
//   step7DOF: accel + heading-vector injection (psi)
//   step9DOF: accel + magnetometer-vector injection (m_ref)
//-------------------------------------------------------------------------
class QuatObserver {
public:
  explicit QuatObserver(const Config& cfg = Config{});

  // State access
  void setQuat(const Quat& q);
  void setBiasGyro(const Vec3& b);

  Quat quat() const { return q_nb_; }                 // BODY→END quaternion
  Mat3 R_endb() const { return RnbFromQuatCustom(q_nb_); }
  const Vec3& bias_gyro() const { return b_g_; }
  const Vec3& w_est() const { return w_est_; }

  // Steps
  void step6DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b);
  void step7DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, double psi_meas_end);
  void step9DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, const Vec3& mag_b);

  // ---------------------------------------------------------------------
  // MATLAB Tquat equivalents (no Eigen; raw arrays)
  // ---------------------------------------------------------------------
  // Tq(q): 4x3 such that q_dot = Tq(q) * w
  static void Tquat_q(const Quat& q, double Tq_out[4][3]);
  // Tw(w): 4x4 such that q_dot = Tw(w) * q
  static void Tquat_w(const Vec3& w_b, double Tw_out[4][4]);

private:
  // Quaternion utilities (custom Quat)
  static Quat quatMul_(const Quat& a, const Quat& b);
  static Quat quatNormalize_(const Quat& q);
  static Quat normalizeUnit_(const Quat& q);

  // Exact discretization equivalent to expm(Tquat(w)*dt)*q (MATLAB)
  static Quat expmTquatW_times_q_(const Vec3& w_b, double dt, const Quat& q_k);

  // Injection terms (sigma in BODY)
  Vec3 sigma7_(const Vec3& f_imu_b, double psi_meas_end) const;
  Vec3 sigma9_(const Vec3& f_imu_b, const Vec3& m_imu_b) const;

private:
  Config cfg_;
  Quat   q_nb_;   // BODY -> END attitude
  Vec3   b_g_;    // gyro bias estimate (BODY)
  Vec3   w_est_;  // estimated body rates used to propagate q
};

} // namespace qobs
#ifndef QUAT_OBSERVER_HPP
#define QUAT_OBSERVER_HPP


#pragma once
// QuatObserver.hpp — Quaternion observer (Grip et al., 2013) with gyro-bias estimation
// Frames: BODY (x fwd, y starboard, z down), NAV/END (E, N, D; Down +).
// Quaternion maps BODY→NAV and uses Eigen ordering (w, x, y, z).

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Utilities/calculations.hpp"

namespace qobs {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

struct Config {
  // Gains
  Mat3  Ki  = Mat3::Identity() * 1e-3; // integral gain for gyro-bias (diag)
  double k1 = 1.0;                     // accel vector injection gain
  double k2 = 0.5;                     // mag/heading injection gain

  // Robustness guards
  double accel_min_norm = 1e-3;        // [m/s^2] minimum |f| for normalization
  double mag_min_norm   = 1e-6;        // [arb]    minimum |m| for normalization

  // Reference magnetic field in NAV (set from site/calibration)
  Vec3 m_ref_nav = Vec3(1, 0, 0);      // default points East; replace as needed
};

class QuatObserver {
public:
  explicit QuatObserver(const Config& cfg = Config());

  // State access
  const Vec3& w_est()              const { return w_est_; } // BODY
  const Eigen::Quaterniond& quat() const { return q_nb_; }  // BODY→NAV
  const Vec3& bias_gyro()          const { return b_ars_; } // BODY
  Mat3 Rnb() const { return RnbFromQuatCustom(q_nb_); }     // custom END DCM

  void setQuat(const Eigen::Quaterniond& q_nb) { q_nb_ = q_nb.normalized(); }
  void setBiasGyro(const Vec3& b)             { b_ars_ = b; }
  void setMagRefNav(const Vec3& m_ref)        { cfg_.m_ref_nav = m_ref; }

  // 9-DOF step: accel, gyro, mag in BODY
  void step9DOF(double h, const Vec3& f_imu_b, const Vec3& w_imu_b, const Vec3& m_imu_b);

  // 7-DOF step (compass heading): pass psi [rad] or (cospsi, sinpsi)
  void step7DOF(double h, const Vec3& f_imu_b, const Vec3& w_imu_b, double psi);

  // 6-DOF step: accel and gyro only (prediction, sigma = 0)
  void step6DOF(double h, const Vec3& /*f_imu_b*/, const Vec3& w_imu_b);

  // Config access
  const Config& cfg() const { return cfg_; }
  void setConfig(const Config& c) { cfg_ = c; }

private:
  static Eigen::Quaterniond quatExp(const Vec3& phi);  // exp on SO(3) via quaternion

  // Core update given an injection sigma (in BODY)
  void integrate(double h, const Vec3& w_imu_b, const Vec3& sigma);

private:
  Config cfg_{};
  Vec3   w_est_{0.0, 0.0, 0.0};
  Eigen::Quaterniond q_nb_ = Eigen::Quaterniond::Identity(); // BODY→NAV
  Vec3 b_ars_ = Vec3::Zero();                                 // gyro bias (BODY)
};

} // namespace qobs

#endif
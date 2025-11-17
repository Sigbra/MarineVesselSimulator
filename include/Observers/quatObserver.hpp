#pragma once

#include "Utilities/calculations.hpp"   // brings ::Quat, ::Vec3, ::Mat3, RnbFromQuatCustom, yawFromQuatEND, ssa

namespace qobs {

// Reuse the shared math aliases and quaternion type
using ::Vec3;   // Eigen::Vector3d
using ::Mat3;   // Eigen::Matrix3d
using ::Quat;   // non-Eigen quaternion {w,x,y,z}

//--------------------------------------------------------------------
// Configuration
//--------------------------------------------------------------------
struct Config {
  Mat3   Ki             = Mat3::Identity() * 1e-3; // gyro-bias integral gain
  double k1             = 0.9;                     // accel vector gain
  double k2             = 1.2;                     // heading (compass/GNSS) gain
  double accel_min_norm = 0.5;                     // guard against near-zero |f|
  double mag_min_norm   = 1e-6;                    // guard for magnetometer use
};

//--------------------------------------------------------------------
// Quaternion Observer (BODY -> END convention)
//--------------------------------------------------------------------
class QuatObserver {
public:
  explicit QuatObserver(const Config& cfg);

  // Setters / getters
  void setQuat(const Quat& q);
  void setBiasGyro(const Vec3& b);
  const Vec3& bias_gyro() const { return b_g_; }
  Vec3 w_est() const { return w_est_; }

  Quat quat() const { return q_nb_; }                              // BODY→END quaternion
  Mat3 Rnb()  const { return RnbFromQuatCustom(q_nb_); }           // BODY→END DCM

  // Update steps
  void step6DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b);
  void step7DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, double psi_meas_end);
  void step9DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, const Vec3& mag_b);

private:
  // Helpers
  static void normalize_canonical(Quat& q);                        // normalize + deterministic hemisphere
  static Quat integrate_body_rates(const Quat& q, const Vec3& omega_b, double dt);

private:
  Config cfg_;
  Quat   q_nb_;     // BODY -> END attitude
  Vec3   b_g_;      // gyro bias estimate (BODY)
  Vec3   w_est_;    // estimated body rates used to integrate q
};

} // namespace qobs

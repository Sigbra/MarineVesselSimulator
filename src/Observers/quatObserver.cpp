#include "Observers/quatObserver.hpp"
#include <cmath>

namespace qobs {

// --------- small helpers ---------
static inline Quat quat_add(const Quat& a, const Quat& b){
  Quat c; c.w=a.w+b.w; c.x=a.x+b.x; c.y=a.y+b.y; c.z=a.z+b.z; return c;
}
static inline Quat quat_scale(const Quat& a, double s){
  Quat c; c.w=a.w*s; c.x=a.x*s; c.y=a.y*s; c.z=a.z*s; return c;
}

void QuatObserver::normalize_canonical(Quat& q){
  const double n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
  if (n2 > 0.0) {
    const double invn = 1.0 / std::sqrt(n2);
    q.w *= invn; q.x *= invn; q.y *= invn; q.z *= invn;
  } else {
    q.w = 1.0; q.x = q.y = q.z = 0.0; // identity fallback
  }
  // deterministic hemisphere (matches your earlier rule)
  if (q.w < 0.0 || (std::abs(q.w) <= 1e-12 && q.z < 0.0)) {
    q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z;
  }
}

// Integrate dq/dt = 0.5 * q ⊗ [0, omega_b]  (BODY rates)
Quat QuatObserver::integrate_body_rates(const Quat& q, const Vec3& omega_b, double dt){
  // q = (w,x,y,z), omega=(wx,wy,wz)
  const double w = q.w, x = q.x, y = q.y, z = q.z;
  const double wx = omega_b.x(), wy = omega_b.y(), wz = omega_b.z();

  Quat dq;
  dq.w = -0.5 * ( x*wx + y*wy + z*wz );
  dq.x =  0.5 * ( w*wx + y*wz - z*wy );
  dq.y =  0.5 * ( w*wy - x*wz + z*wx );
  dq.z =  0.5 * ( w*wz + x*wy - y*wx );

  Quat q_new = quat_add(q, quat_scale(dq, dt));
  normalize_canonical(q_new);
  return q_new;
}

// --------- class ---------
QuatObserver::QuatObserver(const Config& cfg) : cfg_(cfg) {
  q_nb_.w = 1.0; q_nb_.x = q_nb_.y = q_nb_.z = 0.0;
  b_g_.setZero();
  w_est_.setZero();
}

void QuatObserver::setQuat(const Quat& q){
  q_nb_ = q;
  normalize_canonical(q_nb_);
}

void QuatObserver::setBiasGyro(const Vec3& b){ b_g_ = b; }

// 6DOF: gyro + accel (gravity vector correction)
void QuatObserver::step6DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b){
  // Reject tiny accel norms to avoid numerical noise
  const double an = accel_b.norm();
  Vec3 a_dir = accel_b;
  if (an > cfg_.accel_min_norm) a_dir /= an; else a_dir = Vec3(0,0,1); // Down (+z) in BODY

  // Predicted gravity direction in BODY from current attitude:
  const Mat3 R_nb = RnbFromQuatCustom(q_nb_);     // BODY→END
  const Mat3 R_bn = R_nb.transpose();
  const Vec3 g_b_hat = R_bn.col(2);               // END z (Down) expressed in BODY

  // Error as cross product (drive g_b_hat → a_dir)
  const Vec3 e_acc = g_b_hat.cross(a_dir);

  // Estimated body rates used for integration
  Vec3 w_tilde = gyro_b - b_g_ + cfg_.k1 * e_acc;

  // Bias integral (classic Mahony-like)
  b_g_ += cfg_.Ki * e_acc * dt;

  // Integrate quaternion
  w_est_ = w_tilde;
  q_nb_ = integrate_body_rates(q_nb_, w_tilde, dt);
}

// 7DOF: 6DOF + heading aid (psi in END, North→East)
void QuatObserver::step7DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, double psi_meas_end){
  // 6DOF part first
  // Use a temporary (no bias update yet), then apply heading correction before final bias update & integration
  const double an = accel_b.norm();
  Vec3 a_dir = accel_b;
  if (an > cfg_.accel_min_norm) a_dir /= an; else a_dir = Vec3(0,0,1);

  const Mat3 R_nb0 = RnbFromQuatCustom(q_nb_);
  const Mat3 R_bn0 = R_nb0.transpose();
  const Vec3 g_b_hat = R_bn0.col(2);
  const Vec3 e_acc = g_b_hat.cross(a_dir);

  // Heading error (END convention)
  const double psi_hat = yawFromQuatEND(q_nb_);             // from current q
  const double e_psi   = ssa(psi_meas_end - psi_hat);       // wrap to (-pi,pi]

  // apply yaw correction on BODY z-axis (reasonable small-angle approx)
  Vec3 w_tilde = gyro_b - b_g_ + cfg_.k1 * e_acc + Vec3(0,0, cfg_.k2 * e_psi);

  // Bias integral only from accel vector (keeps it stable)
  b_g_ += cfg_.Ki * e_acc * dt;

  // Integrate quaternion with corrected rates
  w_est_ = w_tilde;
  q_nb_  = integrate_body_rates(q_nb_, w_tilde, dt);
}

// 9DOF: 6DOF + magnetometer → heading
void QuatObserver::step9DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, const Vec3& mag_b){
  // crude yaw from magnetometer in BODY frame (no declination compensation here)
  double psi_mag = 0.0;
  if (mag_b.head<2>().norm() > cfg_.mag_min_norm) {
    psi_mag = std::atan2(mag_b.y(), mag_b.x()); // yaw ~ atan2(My, Mx)
  } else {
    // if mag unusable, just run 6DOF
    step6DOF(dt, accel_b, gyro_b);
    return;
  }
  step7DOF(dt, accel_b, gyro_b, psi_mag);
}

} // namespace qobs

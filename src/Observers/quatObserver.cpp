// QuatObserver.cpp — Quaternion observer (Grip et al., 2013) with gyro-bias estimation
// Converted to c++ from the MSS Toolbox MATLAB implementation.
#include "Observers/quatObserver.hpp"
#include "Utilities/calculations.hpp"
#include <cmath>

namespace qobs {

// -------- small utils --------
static inline double clamp_norm(double n, double eps){ return (std::abs(n) < eps) ? eps : n; }

QuatObserver::QuatObserver(const Config& cfg) : cfg_(cfg) {}

Eigen::Quaterniond QuatObserver::quatExp(const Vec3& phi){
  const double th = phi.norm();
  if (th < 1e-12) return Eigen::Quaterniond(1.0, 0.5*phi.x(), 0.5*phi.y(), 0.5*phi.z()).normalized();
  const double half = 0.5 * th;
  const double s = std::sin(half) / th;
  return Eigen::Quaterniond(std::cos(half), s*phi.x(), s*phi.y(), s*phi.z());
}

// -------- Core integration --------
void QuatObserver::integrate(double h, const Vec3& w_imu_b, const Vec3& sigma){
  // w_estimated = w_imu - b_ars + sigma (BODY)
  w_est_ = w_imu_b - b_ars_ + sigma;
  // Discrete integration via quaternion exponential: q_{k+1} = exp( w_est_ * h ) ⊗ q_k
  q_nb_ = (quatExp(w_est_ * h) * q_nb_).normalized();
  // Bias integral term: b_{k+1} = b_k - h * Ki * sigma
  b_ars_ = b_ars_ - h * (cfg_.Ki * sigma);
}

// -------- Update modes --------
void QuatObserver::step9DOF(double h, const Vec3& f_imu_b, const Vec3& w_imu_b, const Vec3& m_imu_b){
  // References in NAV
  const Vec3 v01(0,0,-1);                         // gravity (measuring −g at rest)
  const Vec3 v02 = cfg_.m_ref_nav.normalized();   // magnetic reference (NAV)

  // Transform NAV references to BODY: R_bn = R_nb^T
  const Mat3 R_nb = RnbFromQuatCustom(q_nb_);
  const Mat3 R_bn = R_nb.transpose();

  // Normalize measurements (guard tiny norms)
  const double fn = f_imu_b.norm();
  const double mn = m_imu_b.norm();
  const Vec3 v1 = f_imu_b / clamp_norm(fn, cfg_.accel_min_norm);
  const Vec3 v2 = m_imu_b / clamp_norm(mn, cfg_.mag_min_norm);

  // sigma = k1 * v1 × (R_bn v01) + k2 * v2 × (R_bn v02)
  const Vec3 sigma1 = cfg_.k1 * (v1.cross(R_bn * v01));
  const Vec3 sigma2 = cfg_.k2 * (v2.cross(R_bn * v02));
  const Vec3 sigma  = sigma1 + sigma2;

  integrate(h, w_imu_b, sigma);
}

void QuatObserver::step7DOF(double h, const Vec3& f_imu_b, const Vec3& w_imu_b, double psi){
  // References in NAV
  const Vec3 v01(0,0,-1);
  const Vec3 v02(std::sin(psi), std::cos(psi), 0.0); // unit vector in horizontal plane

  const Mat3 R_nb = RnbFromQuatCustom(q_nb_);
  const Mat3 R_bn = R_nb.transpose();

  const double fn = f_imu_b.norm();
  const Vec3 v1 = f_imu_b / clamp_norm(fn, cfg_.accel_min_norm);
  const Vec3 v2_body(1,0,0); // heading direction in BODY

  const Vec3 sigma1 = cfg_.k1 * (v1.cross(R_bn * v01));
  const Vec3 sigma2 = cfg_.k2 * (v2_body.cross(R_bn * v02));
  const Vec3 sigma  = sigma1 + sigma2;

  integrate(h, w_imu_b, sigma);
}

void QuatObserver::step6DOF(double h, const Vec3& /*f_imu_b*/, const Vec3& w_imu_b){
  const Vec3 sigma = Vec3::Zero(); // no aiding
  integrate(h, w_imu_b, sigma);
}

} // namespace qobs

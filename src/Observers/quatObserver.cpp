#include "Observers/quatObserver.hpp"

#include <cmath>

namespace qobs {

//-------------------------------------------------------------------------
// Quaternion utilities (Hamilton product, [w,x,y,z])
//-------------------------------------------------------------------------
Quat QuatObserver::quatMul_(const Quat& a, const Quat& b)
{ 
  Quat c;
  c.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
  c.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
  c.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
  c.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
  return c;
}

Quat QuatObserver::quatNormalize_(const Quat& q_in)
{
  Quat q = q_in;
  const double n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
  if (n2 > 0.0) {
    const double invn = 1.0 / std::sqrt(n2);
    q.w *= invn; q.x *= invn; q.y *= invn; q.z *= invn;
  } else {
    q.w = 1.0; q.x = 0.0; q.y = 0.0; q.z = 0.0;
  }
  return q;
}

// MATLAB-equivalent unit normalization only (no deterministic hemisphere).
Quat QuatObserver::normalizeUnit_(const Quat& q_in)
{
  return quatNormalize_(q_in);
}

//-------------------------------------------------------------------------
// MATLAB Tquat equivalents
//-------------------------------------------------------------------------
void QuatObserver::Tquat_q(const Quat& q_in, double Tq_out[4][3])
{
  const Quat q = quatNormalize_(q_in);
  const double eta  = q.w;
  const double eps1 = q.x;
  const double eps2 = q.y;
  const double eps3 = q.z;

  // MATLAB:
  // T = 0.5 * [
  //   -eps1 -eps2 -eps3
  //    eta  -eps3  eps2
  //    eps3  eta  -eps1
  //   -eps2  eps1  eta ]
  Tq_out[0][0] = -0.5*eps1;  Tq_out[0][1] = -0.5*eps2;  Tq_out[0][2] = -0.5*eps3;
  Tq_out[1][0] =  0.5*eta;   Tq_out[1][1] = -0.5*eps3;  Tq_out[1][2] =  0.5*eps2;
  Tq_out[2][0] =  0.5*eps3;  Tq_out[2][1] =  0.5*eta;   Tq_out[2][2] = -0.5*eps1;
  Tq_out[3][0] = -0.5*eps2;  Tq_out[3][1] =  0.5*eps1;  Tq_out[3][2] =  0.5*eta;
}

void QuatObserver::Tquat_w(const Vec3& w_b, double Tw_out[4][4])
{
  const double wx = w_b.x();
  const double wy = w_b.y();
  const double wz = w_b.z();

  // MATLAB:
  // Tw = 0.5 * [
  //   0  -w'
  //   w  -Smtrx(w) ]
  // with Smtrx(w) = [ 0 -wz wy; wz 0 -wx; -wy wx 0 ]

  // row 0
  Tw_out[0][0] = 0.0;
  Tw_out[0][1] = -0.5*wx;
  Tw_out[0][2] = -0.5*wy;
  Tw_out[0][3] = -0.5*wz;

  // row 1
  Tw_out[1][0] =  0.5*wx;
  Tw_out[1][1] =  0.0;
  Tw_out[1][2] =  0.5*wz;
  Tw_out[1][3] = -0.5*wy;

  // row 2
  Tw_out[2][0] =  0.5*wy;
  Tw_out[2][1] = -0.5*wz;
  Tw_out[2][2] =  0.0;
  Tw_out[2][3] =  0.5*wx;

  // row 3
  Tw_out[3][0] =  0.5*wz;
  Tw_out[3][1] =  0.5*wy;
  Tw_out[3][2] = -0.5*wx;
  Tw_out[3][3] =  0.0;
}

//-------------------------------------------------------------------------
// Exact discrete update equivalent to:
//   q_{k+1} = expm( Tquat(w) * dt ) * q_k
// where Tquat(w) matches the MATLAB Tw above.
//-------------------------------------------------------------------------
Quat QuatObserver::expmTquatW_times_q_(const Vec3& w_b, double dt, const Quat& q_k)
{
  const double wx = w_b.x();
  const double wy = w_b.y();
  const double wz = w_b.z();

  const double wnorm = std::sqrt(wx*wx + wy*wy + wz*wz);

  // dq = [cos(|w|dt/2), (w/|w|)*sin(|w|dt/2)]
  Quat dq;
  if (wnorm <= 1e-12) {
    // Small-angle limit: sin(half)/|w| ≈ dt/2
    dq.w = 1.0;
    dq.x = 0.5 * wx * dt;
    dq.y = 0.5 * wy * dt;
    dq.z = 0.5 * wz * dt;
  } else {
    const double half = 0.5 * wnorm * dt;
    const double s_over_w = std::sin(half) / wnorm;
    dq.w = std::cos(half);
    dq.x = wx * s_over_w;
    dq.y = wy * s_over_w;
    dq.z = wz * s_over_w;
  }

  // MATLAB Tw corresponds to: q_dot = 0.5 * (q ⊗ [0,w])
  // therefore: q_{k+1} = q_k ⊗ dq
  return quatNormalize_(quatMul_(quatNormalize_(q_k), dq));
}

//-------------------------------------------------------------------------
// Construction / setters
//-------------------------------------------------------------------------
QuatObserver::QuatObserver(const Config& cfg) : cfg_(cfg)
{
  q_nb_  = normalizeUnit_(Quat{});
  b_g_   = Vec3::Zero();
  w_est_ = Vec3::Zero();
}

void QuatObserver::setQuat(const Quat& q)
{
  q_nb_ = normalizeUnit_(q);
}

void QuatObserver::setBiasGyro(const Vec3& b)
{
  b_g_ = b;
}

//-------------------------------------------------------------------------
// Injection terms (MATLAB-equivalent, adapted for END)
//-------------------------------------------------------------------------
Vec3 QuatObserver::sigma7_(const Vec3& f_imu_b, double psi_meas_end) const
{
  // MATLAB: R_transposed = Rquat(q)' => R_bn
  const Mat3 R_nb = RnbFromQuatCustom(q_nb_); // BODY -> END
  const Mat3 R_bn = R_nb.transpose();         // END  -> BODY

  Vec3 sigma = Vec3::Zero();

  // sigma1 (accel): v01 = [0;0;-1] in NAV (Up, since measuring -g at rest)
  const double fn = f_imu_b.norm();
  if (fn > cfg_.accel_min_norm) {
    const Vec3 v1 = f_imu_b / fn;
    const Vec3 v01_end(0.0, 0.0, -1.0);
    const Vec3 v01_b = R_bn * v01_end;
    sigma += cfg_.k1 * v1.cross(v01_b);
  }

  // sigma2 (heading):
  // MATLAB NED: v02 = [cos(psi); sin(psi); 0]
  // END (E,N,D): v02 = [sin(psi); cos(psi); 0]
  const Vec3 v2_b(1.0, 0.0, 0.0);
  const Vec3 v02_end(std::sin(psi_meas_end), std::cos(psi_meas_end), 0.0);
  const Vec3 v02_b = R_bn * v02_end;
  sigma += cfg_.k2 * v2_b.cross(v02_b);

  return sigma;
}

Vec3 QuatObserver::sigma9_(const Vec3& f_imu_b, const Vec3& m_imu_b) const
{
  const Mat3 R_nb = RnbFromQuatCustom(q_nb_); // BODY -> END
  const Mat3 R_bn = R_nb.transpose();         // END  -> BODY

  Vec3 sigma = Vec3::Zero();

  // sigma1 (accel)
  const double fn = f_imu_b.norm();
  if (fn > cfg_.accel_min_norm) {
    const Vec3 v1 = f_imu_b / fn;
    const Vec3 v01_end(0.0, 0.0, -1.0);
    const Vec3 v01_b = R_bn * v01_end;
    sigma += cfg_.k1 * v1.cross(v01_b);
  }

  // sigma2 (mag): v02 = m_ref/||m_ref|| in NAV, v2 = m_imu/||m_imu|| in BODY
  const double mn = m_imu_b.norm();
  const double rn = cfg_.m_ref_end.norm();
  if (mn > cfg_.mag_min_norm && rn > cfg_.mag_min_norm) {
    const Vec3 v2_b = m_imu_b / mn;
    const Vec3 v02_end = cfg_.m_ref_end / rn;
    const Vec3 v02_b = R_bn * v02_end;
    sigma += cfg_.k2 * v2_b.cross(v02_b);
  }

  return sigma;
}

//-------------------------------------------------------------------------
// Step functions (match MATLAB mode behavior)
//-------------------------------------------------------------------------

// 6DOF: MATLAB predictor step only (sigma = 0)
void QuatObserver::step6DOF(double dt, const Vec3& /*accel_b*/, const Vec3& gyro_b)
{
  const Vec3 sigma = Vec3::Zero();

  w_est_ = gyro_b - b_g_ + sigma;
  q_nb_  = normalizeUnit_(expmTquatW_times_q_(w_est_, dt, q_nb_));

  // MATLAB: b = b - dt * Ki * sigma (no change since sigma=0)
  b_g_   += - (cfg_.Ki * sigma) * dt;
}

// 7DOF: MATLAB corrector (accel + heading)
void QuatObserver::step7DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, double psi_meas_end)
{
  const Vec3 sigma = sigma7_(accel_b, psi_meas_end);

  w_est_ = gyro_b - b_g_ + sigma;
  q_nb_  = normalizeUnit_(expmTquatW_times_q_(w_est_, dt, q_nb_));

  // MATLAB: b = b - dt * Ki * sigma
  b_g_   += - (cfg_.Ki * sigma) * dt;
}

// 9DOF: MATLAB corrector (accel + mag)
void QuatObserver::step9DOF(double dt, const Vec3& accel_b, const Vec3& gyro_b, const Vec3& mag_b)
{
  const Vec3 sigma = sigma9_(accel_b, mag_b);

  w_est_ = gyro_b - b_g_ + sigma;
  q_nb_  = normalizeUnit_(expmTquatW_times_q_(w_est_, dt, q_nb_));

  b_g_   += - (cfg_.Ki * sigma) * dt;
}

} // namespace qobs
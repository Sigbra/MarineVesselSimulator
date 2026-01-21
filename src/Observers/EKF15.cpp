#include "Observers/EKF15.hpp"
#include <cmath>
#include <Eigen/Dense>
#include "Models/model_utilities.hpp"  
#include "Utilities/calculations.hpp"

EKF15::EKF15()
: g_(9.80665),
  x_(VecN::Zero()),
  P_(MatN::Identity() * 1e-3),
  Qw_(Eigen::Matrix<double,12,12>::Zero()),
  R_pos_(Eigen::Matrix3d::Identity() * (0.5*0.5)),
  R_head_(std::pow(0.5*M_PI/180.0, 2))
{
  // default process noise: tune as needed
  Eigen::Matrix<double,12,1> q_diag;
  q_diag.segment<3>(0).setConstant(4e-4);  // accel noise
  q_diag.segment<3>(3).setConstant(4e-6);  // gyro noise
  q_diag.segment<3>(6).setConstant(2e-6);  // accel bias RW
  q_diag.segment<3>(9).setConstant(2e-7);  // gyro bias RW
  setProcessNoise(q_diag);
}

void EKF15::initObserverEKF15(double psi0, const Eigen::Vector3d& r0_end)
{
    VecN x0 = VecN::Zero();

    // v^b = 0
    x0.segment<3>(0).setZero();

    // r^END = initial position [xE, yN, zD]
    x0.segment<3>(3) = r0_end;

    // phi, theta, psi (level, with given yaw)
    x0(6) = 0.0;      // roll
    x0(7) = 0.0;      // pitch
    x0(8) = psi0;     // yaw: 0 at North, CW+

    setState(x0);     // uses existing setState (wraps yaw via ssa)
}

void EKF15::setProcessNoise(const Eigen::Matrix<double,12,1>& q_diag)
{
  Qw_.setZero();
  for(int i=0;i<12;++i)
    Qw_(i,i) = q_diag(i);
}

// ------------------------- predict -----------------------------

void EKF15::predict(const Eigen::Vector3d& a_m,
                    const Eigen::Vector3d& w_m,
                    double dt)
{
  // Interpret (x_,P_) as posterior at time k: x[k], P[k]
  const VecN xk = x_;
  const MatN Pk = P_;

  // Continuous dynamics f0(x[k], u[k])
  const VecN fk = f(xk, a_m, w_m);

  // A = ∂f0/∂x |_{xk}
  const MatN A  = A_numeric(xk, a_m, w_m);
  const MatN Ad = MatN::Identity() + A * dt;

  // G(xk) for process noise mapping
  const MatG G  = G_matrix(xk);
  const MatG Ed = G * dt;  // E_d = h G

  // State predict: x^-[k+1] = x[k] + h f_k(x[k],u[k],0)
  x_ = xk + fk * dt;
  x_(8) = ssa(x_(8));   // wrap yaw ψ

  // Covariance predict: P^- = Ad P Ad^T + Ed Qw Ed^T
  P_ = Ad * Pk * Ad.transpose() + Ed * Qw_ * Ed.transpose();

  // Symmetrize numerically
  P_ = 0.5 * (P_ + P_.transpose());
}

// ------------------------ updatePos ----------------------------

// void EKF15::updatePos(const Eigen::Vector3d& y_pos, const Eigen::Vector3d& r_body)
// {
//   const double phi   = x_(6);
//   const double theta = x_(7);
//   const double psi   = x_(8);

//   const Eigen::Matrix3d Rnb = Rzyx(phi, theta, psi);
//   const Eigen::Vector3d y_pos_c = y_pos - Rnb * r_body;  // implied measurement of p^END

//   // Measurement model: y = r^END + n, r^END at indices 3..5
//   Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();
//   H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

//   Eigen::Vector3d yhat  = x_.segment<3>(3);
//   Eigen::Vector3d innov = y_pos_c - yhat;

//   Eigen::Matrix3d S = H * P_ * H.transpose() + R_pos_;

//   Eigen::Matrix<double,15,3> K =
//       P_ * H.transpose() * S.ldlt().solve(Eigen::Matrix3d::Identity());

//   // State update
//   x_ += K * innov;

//   // Covariance update (Joseph form)
//   MatN I   = MatN::Identity();
//   MatN IKH = I - K * H;
//   P_ = IKH * P_ * IKH.transpose() + K * R_pos_ * K.transpose();

//   P_ = 0.5 * (P_ + P_.transpose());

//   // wrap yaw (position update can couple into ψ via cross-covariance)
//   x_(8) = ssa(x_(8));
// }

void EKF15::updatePos(const Eigen::Vector3d& y_pos,
                      const Eigen::Vector3d& r_body)
{
  // Measurement model (antenna): y = p^END + R_nb(φ,θ,ψ)*r_body + n
  Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();
  H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

  const double phi   = x_(6);
  const double theta = x_(7);
  const double psi   = x_(8);

  const Eigen::Matrix3d R0 = Rzyx(phi, theta, psi);
  const Eigen::Vector3d yhat  = x_.segment<3>(3) + R0 * r_body;
  const Eigen::Vector3d innov = y_pos - yhat;

  // Attitude sensitivity: ∂(R(φ,θ,ψ) r)/∂[φ θ ψ]
  const double eps = 1e-7;
  H.col(6) = (Rzyx(phi + eps, theta,     psi    ) * r_body - R0 * r_body) / eps; // d/dphi
  H.col(7) = (Rzyx(phi,       theta + eps, psi   ) * r_body - R0 * r_body) / eps; // d/dtheta
  H.col(8) = (Rzyx(phi,       theta,     psi + eps) * r_body - R0 * r_body) / eps; // d/dpsi

  Eigen::Matrix3d S = H * P_ * H.transpose() + R_pos_;

  Eigen::Matrix<double,15,3> K =
      P_ * H.transpose() * S.ldlt().solve(Eigen::Matrix3d::Identity());

  x_ += K * innov;

  MatN I   = MatN::Identity();
  MatN IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R_pos_ * K.transpose();

  P_ = 0.5 * (P_ + P_.transpose());
  x_(8) = ssa(x_(8));
}


// ----------------------- updateHeading -------------------------

void EKF15::updateHeading(double y_head)
{
  // Measurement model: y = psi + n, psi at index 8
  Eigen::Matrix<double,1,15> H = Eigen::Matrix<double,1,15>::Zero();
  H(0,8) = 1.0;

  double innov = ssa(y_head - x_(8));

  double S = (H * P_ * H.transpose())(0,0) + R_head_;

  Eigen::Matrix<double,15,1> K = P_ * H.transpose() / S;

  x_ += K * innov;

  MatN I   = MatN::Identity();
  MatN IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * (R_head_) * K.transpose();

  P_ = 0.5 * (P_ + P_.transpose());
  x_(8) = ssa(x_(8));
}

// ----------------------- dynamics f0 ---------------------------

EKF15::VecN EKF15::f(const VecN& xs,
                     const Eigen::Vector3d& a_m,
                     const Eigen::Vector3d& w_m) const
{
  VecN xd = VecN::Zero();

  const Eigen::Vector3d v   = xs.segment<3>(0);  // v^b (body: x fwd, y stb, z down)
  const Eigen::Vector3d r   = xs.segment<3>(3);  // r^END (not used directly here)
  const double phi   = xs(6);
  const double theta = xs(7);
  const double psi   = xs(8);
  const Eigen::Vector3d b_a = xs.segment<3>(9);
  const Eigen::Vector3d b_g = xs.segment<3>(12);

  // Bias-corrected gyro
  const Eigen::Vector3d omega = w_m - b_g;

  // Rnb: Body -> END (x east, y north, z down) with ψ=0 at North, +CW
  const Eigen::Matrix3d Rnb = Rzyx(phi, theta, psi);
  const Eigen::Matrix3d Rbn = Rnb.transpose();
  const Eigen::Vector3d g_n(0.0, 0.0, g_);   // gravity in END (positive down)

  const Eigen::Vector3d a_corr = a_m - b_a;

  // vdot^b = a_corr + Rbn*g^n - ω x v
  xd.segment<3>(0) = a_corr + Rbn * g_n - Smtrx(omega) * v;

  // rdot^END = Rnb * v^b
  xd.segment<3>(3) = Rnb * v;

  // Euler rates = Tzyx(phi,theta) * ω
  xd.segment<3>(6) = Tzyx(phi, theta) * omega;

  // bias derivatives = 0 (random walk via process noise only)
  xd.segment<3>(9).setZero();   // b_a
  xd.segment<3>(12).setZero();  // b_g

  return xd;
}

// --------------------- numeric Jacobian A ----------------------

EKF15::MatN EKF15::A_numeric(const VecN& xs,
                             const Eigen::Vector3d& a_m,
                             const Eigen::Vector3d& w_m,
                             double eps) const
{
  MatN A = MatN::Zero();
  const VecN f0 = f(xs, a_m, w_m);

  for(int i=0;i<N;++i){
    VecN xh = xs;
    xh(i) += eps;
    VecN fh = f(xh, a_m, w_m);
    A.col(i) = (fh - f0) / eps;
  }
  return A;
}

// -------------------- process noise mapping G ------------------

EKF15::MatG EKF15::G_matrix(const VecN& xs) const
{
  MatG G = MatG::Zero();

  const double phi   = xs(6);
  const double theta = xs(7);
  const double psi   = xs(8);

  const Eigen::Matrix3d T = Tzyx(phi, theta);

  // n_a -> vdot^b (states 0..2, cols 0..2)
  G.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

  // n_omega -> Euler rates (states 6..8, cols 3..5)
  G.block<3,3>(6,3) = T;

  // n_ba   -> b_a dot (states 9..11, cols 6..8)
  G.block<3,3>(9,6) = Eigen::Matrix3d::Identity();

  // n_bg   -> b_g dot (states 12..14, cols 9..11)
  G.block<3,3>(12,9) = Eigen::Matrix3d::Identity();

  return G;
}
#include "Observers/EKF18.hpp"
#include "Models/model_utilities.hpp"
#include <cmath>

// --------------------------- ctor / config ---------------------------
EKF18::EKF18()
: g_(9.80665),
  x_(VecN::Zero()),
  P_(MatN::Identity() * 1e-2),
  q_(VecN::Zero()),
  R_gyro_(Eigen::Matrix3d::Identity() * (0.005*0.005)),
  R_pos_ (Eigen::Matrix3d::Identity() * (0.05*0.05)),
  R_head_(std::pow(0.5*M_PI/180.0, 2))
{
  // Defaults chosen for stability; tune as needed
  q_.segment<3>(0).setConstant(1e-3);  // u,v,w
  q_.segment<3>(3).setConstant(1e-4);  // p,q,r
  q_.segment<3>(6).setConstant(1e-6);  // x,y,z
  q_.segment<3>(9).setConstant(1e-5);  // phi,theta,psi
  q_.segment<3>(12).setConstant(1e-6); // b_ax,b_ay,b_az
  q_.segment<3>(15).setConstant(1e-8); // b_gx,b_gy,b_gz

  // Bias priors more uncertain
  P_.block<3,3>(12,12).setIdentity(); P_.block<3,3>(12,12) *= 1e-2;
  P_.block<3,3>(15,15).setIdentity(); P_.block<3,3>(15,15) *= 1e-2;
}

void EKF18::setGravity(double g){ g_ = g; }
void EKF18::setState(const VecN& x){ x_ = x; x_(11)=wrapPi(x_(11)); }
void EKF18::setCovariance(const MatN& P){ P_ = P; }
void EKF18::setProcessNoise(const VecN& q){ q_ = q; }
void EKF18::setRgyro(const Eigen::Matrix3d& R){ R_gyro_ = R; }
void EKF18::setRpos (const Eigen::Matrix3d& R){ R_pos_  = R; }
void EKF18::setRhead(double R){ R_head_ = R; }

Eigen::VectorXd EKF18::getState() const {
  return Eigen::VectorXd::Map(x_.data(), x_.size());
}

Eigen::VectorXd EKF18::getState12() const {
  return Eigen::VectorXd::Map(x_.data(), 12);
}

Eigen::MatrixXd EKF18::getCovariance() const {
  return Eigen::MatrixXd(P_);
}

// ----------------------------- predict ------------------------------
void EKF18::predict(const Eigen::Vector3d& a_m, double dt){
  // guard near cos(theta) ~ 0
  if (std::abs(std::cos(x_(10))) < 1e-6)
    x_(10) = (x_(10)>0.0 ? (M_PI/2 - 1e-6) : (-M_PI/2 + 1e-6));

  // state propagation
  VecN xdot = f(x_, a_m);
  x_ += xdot * dt;
  x_(11) = wrapPi(x_(11));

  // covariance propagation
  MatN A  = A_numeric(x_, a_m);
  MatN Fd = MatN::Identity() + A * dt;

  MatN Qd = MatN::Zero();
  for(int i=0;i<18;++i) Qd(i,i) = q_(i) * dt;

  P_ = Fd * P_ * Fd.transpose() + Qd;
}

// ------------------------------ updates -----------------------------
void EKF18::updateGyro(const Eigen::Vector3d& y_gyro){
  // y = w + b_g + n
  Eigen::Matrix<double,3,18> H = Eigen::Matrix<double,3,18>::Zero();
  H.block<3,3>(0,3)  = Eigen::Matrix3d::Identity(); // w
  H.block<3,3>(0,15) = Eigen::Matrix3d::Identity(); // b_g

  Eigen::Vector3d yhat = x_.segment<3>(3) + x_.segment<3>(15);
  Eigen::Vector3d innov = y_gyro - yhat;

  Eigen::Matrix3d S = H*P_*H.transpose() + R_gyro_;
  Eigen::Matrix<double,18,3> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

void EKF18::updatePos(const Eigen::Vector3d& y_pos){
  // y = [x y z]^T + n
  Eigen::Matrix<double,3,18> H = Eigen::Matrix<double,3,18>::Zero();
  H.block<3,3>(0,6) = Eigen::Matrix3d::Identity();

  Eigen::Vector3d yhat = x_.segment<3>(6);
  Eigen::Vector3d innov = y_pos - yhat;

  Eigen::Matrix3d S = H*P_*H.transpose() + R_pos_;
  Eigen::Matrix<double,18,3> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

void EKF18::updateHeading(double y_head){
  // y = psi + n
  Eigen::Matrix<double,1,18> H = Eigen::Matrix<double,1,18>::Zero();
  H(0,11) = 1.0;

  double innov = wrapPi(y_head - x_(11));
  double S     = (H*P_*H.transpose())(0,0) + R_head_;
  Eigen::Matrix<double,18,1> K = P_*H.transpose() / S;

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
  x_(11) = wrapPi(x_(11));
}

// ---------------------------- dynamics ------------------------------
EKF18::VecN EKF18::f(const VecN& xs, const Eigen::Vector3d& a_m) const {
  VecN xd = VecN::Zero();

  const Eigen::Vector3d v    = xs.segment<3>(0);    // [u v w]^b
  const Eigen::Vector3d w    = xs.segment<3>(3);    // [p q r]^b
  const Eigen::Vector3d b_a  = xs.segment<3>(12);
  // const Eigen::Vector3d b_g  = xs.segment<3>(15); // used in measurement

  const double phi=xs(9), theta=xs(10), psi=xs(11);

  const Eigen::Matrix3d Rnb = Rzyx(phi,theta,psi);  // body->nav
  const Eigen::Matrix3d Rbn = Rnb.transpose();
  const Eigen::Vector3d g_n(0,0,g_);

  // specific force corrected by accel bias
  const Eigen::Vector3d a_corr = a_m - b_a;

  // vdot^b = a_corr + Rbn*g^n - w x v
  xd.segment<3>(0) = a_corr + Rbn*g_n - Smtrx(w)*v;

  // wdot: random walk via Q only
  xd.segment<3>(3).setZero();

  // rdot^n = Rnb * v^b
  xd.segment<3>(6) = Rnb * v;

  // Euler rates = Tzyx(phi,theta) * w
  xd.segment<3>(9) = Tzyx(phi,theta) * w;

  // bias dynamics: random walk (derivative = 0; Q handles diffusion)
  xd.segment<3>(12).setZero(); // b_a
  xd.segment<3>(15).setZero(); // b_g

  return xd;
}

EKF18::MatN EKF18::A_numeric(const VecN& xs, const Eigen::Vector3d& a_m, double eps) const {
  MatN A = MatN::Zero();
  VecN f0 = f(xs, a_m);
  for(int i=0;i<18;++i){
    VecN xh = xs; xh(i) += eps;
    A.col(i) = (f(xh, a_m) - f0) / eps;
  }
  return A;
}

// ---------------------------- helpers -------------------------------
double EKF18::wrapPi(double a){
  while(a<=-M_PI) a += 2*M_PI;
  while(a>  M_PI) a -= 2*M_PI;
  return a;
}

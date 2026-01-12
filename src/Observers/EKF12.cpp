#include "Observers/EKF12.hpp"
#include "Models/model_utilities.hpp"
#include <cmath>

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
EKF12::EKF12()
: g_(9.80665),
  x_(Vec12::Zero()),
  P_(Mat12::Identity() * 1e-2),
  q_(Vec12::Zero()),
  R_gyro_(Eigen::Matrix3d::Identity() * (0.005*0.005)),
  R_pos_ (Eigen::Matrix3d::Identity() * (0.05*0.05)),
  R_head_(std::pow(0.5*M_PI/180.0, 2))
{
  q_.segment<3>(0).setConstant(0.05*0.05);   // u,v,w
  q_.segment<3>(3).setConstant(0.01*0.01);   // p,q,r
  q_.segment<3>(6).setConstant(1e-4);        // x,y,z
  q_.segment<3>(9).setConstant(1e-5);        // phi,theta,psi
}

// -----------------------------------------------------------------------------
// Config setters
// -----------------------------------------------------------------------------
void EKF12::setGravity(double g){ g_ = g; }
void EKF12::setState(const Vec12& x){ x_ = x; x_(11)=wrapPi(x_(11)); }
void EKF12::setCovariance(const Mat12& P){ P_ = P; }
void EKF12::setProcessNoise(const Vec12& q){ q_ = q; }
void EKF12::setRgyro(const Eigen::Matrix3d& R){ R_gyro_ = R; }
void EKF12::setRpos (const Eigen::Matrix3d& R){ R_pos_  = R; }
void EKF12::setRhead(double R){ R_head_ = R; }

// -----------------------------------------------------------------------------
// Getters
// -----------------------------------------------------------------------------
Eigen::VectorXd EKF12::getState() const {
  return Eigen::VectorXd::Map(x_.data(), x_.size());
}

Eigen::MatrixXd EKF12::getCovariance() const {
  return Eigen::MatrixXd(P_);
}

// -----------------------------------------------------------------------------
// Prediction
// -----------------------------------------------------------------------------
void EKF12::predict(const Eigen::Vector3d& a_m, double dt){
  if (std::abs(std::cos(x_(10))) < 1e-6)
    x_(10) = (x_(10)>0.0 ? (M_PI/2 - 1e-6) : (-M_PI/2 + 1e-6));

  Vec12 xdot = f(x_, a_m);
  x_ += xdot * dt;
  x_(11) = wrapPi(x_(11));

  Mat12 A  = A_numeric(x_, a_m);
  Mat12 Fd = Mat12::Identity() + A * dt;

  Mat12 Qd = Mat12::Zero();
  for(int i=0;i<12;++i) Qd(i,i) = q_(i) * dt;

  P_ = Fd * P_ * Fd.transpose() + Qd;
}

// -----------------------------------------------------------------------------
// Updates
// -----------------------------------------------------------------------------
void EKF12::updateGyro(const Eigen::Vector3d& y_gyro){
  Eigen::Matrix<double,3,12> H = Eigen::Matrix<double,3,12>::Zero();
  H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

  Eigen::Vector3d innov = y_gyro - x_.segment<3>(3);
  Eigen::Matrix3d  S    = H*P_*H.transpose() + R_gyro_;
  Eigen::Matrix<double,12,3> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (Mat12::Identity() - K*H) * P_;
}

void EKF12::updatePos(const Eigen::Vector3d& y_pos){
  Eigen::Matrix<double,3,12> H = Eigen::Matrix<double,3,12>::Zero();
  H.block<3,3>(0,6) = Eigen::Matrix3d::Identity();

  Eigen::Vector3d innov = y_pos - x_.segment<3>(6);
  Eigen::Matrix3d  S    = H*P_*H.transpose() + R_pos_;
  Eigen::Matrix<double,12,3> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (Mat12::Identity() - K*H) * P_;
}

void EKF12::updateHeading(double y_head){
  Eigen::Matrix<double,1,12> H = Eigen::Matrix<double,1,12>::Zero();
  H(0,11) = 1.0;

  double innov = wrapPi(y_head - x_(11));
  double S     = (H*P_*H.transpose())(0,0) + R_head_;
  Eigen::Matrix<double,12,1> K = P_*H.transpose() / S;

  x_ += K * innov;
  P_  = (Mat12::Identity() - K*H) * P_;
  x_(11) = wrapPi(x_(11));
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------
EKF12::Vec12 EKF12::f(const Vec12& xs, const Eigen::Vector3d& a_m) const {
  Vec12 xd = Vec12::Zero();

  const Eigen::Vector3d v = xs.segment<3>(0);
  const Eigen::Vector3d w = xs.segment<3>(3);
  const double phi=xs(9), theta=xs(10), psi=xs(11);

  const Eigen::Matrix3d Rnb = Rzyx(phi,theta,psi);
  const Eigen::Matrix3d Rbn = Rnb.transpose();
  const Eigen::Vector3d g_n(0,0,g_);

  xd.segment<3>(0) = a_m + Rbn*g_n - Smtrx(w)*v;
  xd.segment<3>(3).setZero();
  xd.segment<3>(6) = Rnb * v;
  xd.segment<3>(9) = Tzyx(phi,theta) * w;

  return xd;
}

EKF12::Mat12 EKF12::A_numeric(const Vec12& xs, const Eigen::Vector3d& a_m, double eps) const {
  Mat12 A = Mat12::Zero();
  Vec12 f0 = f(xs,a_m);
  for(int i=0;i<12;++i){
    Vec12 xh = xs; xh(i)+=eps;
    A.col(i) = (f(xh,a_m)-f0)/eps;
  }
  return A;
}

double EKF12::wrapPi(double a){
  while(a<=-M_PI) a+=2*M_PI;
  while(a> M_PI)  a-=2*M_PI;
  return a;
}
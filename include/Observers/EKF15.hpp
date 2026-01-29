#ifndef EKF15_HPP
#define EKF15_HPP


#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <functional>
#include "Models/model_utilities.hpp"  // Rzyx(phi,theta,psi), Tzyx(phi,theta), ssa()
#include "Utilities/calculations.hpp"

class EKF15 {
public:
  static constexpr int N = 15;
  using VecN = Eigen::Matrix<double,N,1>;
  using MatN = Eigen::Matrix<double,N,N>;
  using MatG = Eigen::Matrix<double,N,12>; // process noise mapping G(x)
  
  EKF15();

  // configuration
  void setGravity(double g){ g_ = g; }

  void setState(const VecN& x){
    x_ = x;
    x_(8) = ssa(x_(8));   // wrap yaw ψ
  }

  void initObserverEKF15(double psi0, const Eigen::Vector3d& r0_end);

  void setCovariance(const MatN& P){ P_ = P; }

  // process noise diag for w = [n_a, n_omega, n_ba, n_bg] (12x1)
  void setProcessNoise(const Eigen::Matrix<double,12,1>& q_diag);

  void setRpos(const Eigen::Matrix3d& R){ R_pos_ = R; }
  void setRhead(double R){ R_head_ = R; }

  // accessors
  VecN getState() const { return x_; }
  MatN getCovariance() const { return P_; }

  // EKF steps (discrete-time)
  void predict(const Eigen::Vector3d& a_m,
               const Eigen::Vector3d& w_m,
               double dt);

  void updatePos(const Eigen::Vector3d& y_pos, const Eigen::Vector3d& r_body);
  void updateHeading(double y_head);

private:
  // continuous-time dynamics: xdot = f0(x,u), with u = (a_m, w_m)
  VecN f(const VecN& x,
         const Eigen::Vector3d& a_m,
         const Eigen::Vector3d& w_m) const;

  // Jacobian A = ∂f0/∂x (numeric)
  MatN A_numeric(const VecN& x,
                 const Eigen::Vector3d& a_m,
                 const Eigen::Vector3d& w_m,
                 double eps = 1e-6) const;

  // Process noise mapping G(x) such that xdot = f0 + G w
  MatG G_matrix(const VecN& x) const;

private:
  double g_;     // gravity [m/s^2], positive down in END frame

  VecN x_;       // state
  MatN P_;       // covariance

  // Process noise covariance for w[k] (discrete)
  // w = [n_a, n_omega, n_ba, n_bg], each 3x1
  Eigen::Matrix<double,12,12> Qw_;

  // Measurement noise
  Eigen::Matrix3d R_pos_; // position noise covariance
  double R_head_;         // heading noise variance
};

#endif
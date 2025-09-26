#ifndef EKF18_HPP
#define EKF18_HPP

#pragma once
#include <Eigen/Dense>
#include "Models/model_utilities.hpp"

// 18-state EKF (adds accel & gyro biases)
// x = [u v w p q r x y z phi theta psi b_ax b_ay b_az b_gx b_gy b_gz]^T
class EKF18 {
public:
  using VecN = Eigen::Matrix<double,18,1>;
  using MatN = Eigen::Matrix<double,18,18>;

  EKF18();

  // Config
  void setGravity(double g);
  void setState(const VecN& x);
  void setCovariance(const MatN& P);
  void setProcessNoise(const VecN& q);            // noise power per state (scaled by dt)
  void setRgyro(const Eigen::Matrix3d& R);        // gyro meas cov (rad/s)^2
  void setRpos (const Eigen::Matrix3d& R);        // GNSS pos cov m^2
  void setRhead(double R);                        // heading cov rad^2

  // Getters (dynamic for convenience)
  Eigen::VectorXd getState() const;               // 18x1
  Eigen::VectorXd getState12() const;             // 12x1
  Eigen::MatrixXd getCovariance() const;          // 18x18

  // Predict with variable dt. a_m is measured specific force (m/s^2) in body.
  void predict(const Eigen::Vector3d& a_m, double dt);

  // Updates
  void updateGyro(const Eigen::Vector3d& y_gyro); // y = w + b_g + n
  void updatePos (const Eigen::Vector3d& y_pos);  // y = [x y z]^T + n
  void updateHeading(double y_head);              // y = psi + n (rad)

private:
  VecN f(const VecN& xs, const Eigen::Vector3d& a_m) const;     // xdot
  MatN A_numeric(const VecN& xs, const Eigen::Vector3d& a_m,
                 double eps=1e-7) const;
  static double wrapPi(double a);

  double g_;
  VecN   x_;
  MatN   P_;
  VecN   q_;                       // process noise power per state
  Eigen::Matrix3d R_gyro_;         // gyro measurement covariance
  Eigen::Matrix3d R_pos_;          // position measurement covariance
  double R_head_;                  // heading measurement variance
};

#endif 
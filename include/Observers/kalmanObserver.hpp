#ifndef EKF12_HPP
#define EKF12_HPP

#include <Eigen/Dense>
#include "Models/model_utilities.hpp"

// -----------------------------------------------------------------------------
// 12-state EKF for vessel INS with IMU (acc, gyro) + GNSS (pos, heading)
// State x = [u v w p q r x_n y_n z_n phi theta psi]^T
// -----------------------------------------------------------------------------
class EKF12 {
public:
  using Vec12 = Eigen::Matrix<double,12,1>;
  using Mat12 = Eigen::Matrix<double,12,12>;

  EKF12();

  // --- Configuration ---
  void setGravity(double g);
  void setState(const Vec12& x);
  void setCovariance(const Mat12& P);
  void setProcessNoise(const Vec12& q);
  void setRgyro(const Eigen::Matrix3d& R);
  void setRpos (const Eigen::Matrix3d& R);
  void setRhead(double R);

  // --- Getters (dynamic types for flexibility) ---
  Eigen::VectorXd getState() const;        // 12x1 state vector
  Eigen::MatrixXd getCovariance() const;   // 12x12 covariance matrix

  // --- Prediction ---
  void predict(const Eigen::Vector3d& a_m, double dt);

  // --- Updates ---
  void updateGyro(const Eigen::Vector3d& y_gyro);
  void updatePos(const Eigen::Vector3d& y_pos);
  void updateHeading(double y_head);

private:
  Vec12 f(const Vec12& xs, const Eigen::Vector3d& a_m) const;
  Mat12 A_numeric(const Vec12& xs, const Eigen::Vector3d& a_m, double eps=1e-7) const;
  static double wrapPi(double a);

  // --- Members ---
  double g_;
  Vec12   x_;
  Mat12   P_;
  Vec12   q_;
  Eigen::Matrix3d R_gyro_;
  Eigen::Matrix3d R_pos_;
  double R_head_;
};

#endif // KALMANFILTER15_HPP

#ifndef NN_EKF_V5_HPP
#define NN_EKF_V5_HPP

// ============================================================================
// nn_ekf_v5.hpp
// ============================================================================

#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <utility>

// External kinematics helpers (provided elsewhere in your project)
Eigen::Matrix3d Rzyx(double phi, double theta, double psi);
double ssa(double a);  // wrap to [-pi, pi]

// ================================ NN Interface ================================
// Torch headers are intentionally kept out of the header (pImpl pattern).
class NN_INTERFACE_V5 {
public:
  NN_INTERFACE_V5();
  ~NN_INTERFACE_V5();

  // Load TorchScript model (stateless export expected).
  // Will try both CPU and (optionally) CUDA depending on availability.
  void load(const std::string& model_path, bool use_cuda_request);

  // Reset internal hidden state of the stateless wrapper.
  void reset();

  // Forward the NN once.
  // Input (B=1): [ax, ay, az, wx, wy, wz, cpsi, spsi]
  // Returns:
  //   - mu9: [Vn, Ve, p, q, r, phi, theta, cpsi, spsi] (PHYSICAL units)
  //   - Cov9: predictive covariance (9x9) in PHYSICAL units if ensemble export,
  //           otherwise diagonal covariance is built from per-dim log-variance.
  std::pair<std::array<double,9>, Eigen::Matrix<double,9,9>>
  predict(const std::array<double,8>& x_in) const;

private:
  struct Impl;
  Impl* p_;
};

// ================================== EKF v5 ===================================
class NN_EKF_V5 {
public:
  static constexpr int NX = 8; // [Vn, Ve, r, pE, pN, psi, bVn, bVe]
  using MatN = Eigen::Matrix<double, NX, NX>;
  using VecN = Eigen::Matrix<double, NX, 1>;

  NN_EKF_V5();

  // Basic configuration
  void setDt(double h);
  void setState(const VecN& x0);
  void setCovariance(const MatN& P0);
  void setProcessNoise(const MatN& Qd);

  // NN backend
  void initNN(const std::string& model_path, bool use_cuda_request);
  void resetNN();

  // Access EKF state (after predict/update)
  const VecN& state() const;

  // EKF steps
  void predict();
  void updateFromIMU(const std::array<double,8>& imu8); // IMU -> NN -> EKF update
  void updateNN(const std::array<double,9>& mu9,
                const Eigen::Matrix<double,9,9>& Cov9);
  void updatePosEN(const Eigen::Vector2d& pEN,
                   const Eigen::Matrix2d& R_EN);
  void updateHeading(double psi_meas, double R_psi);

  // Legacy 12-state export: [u v w p q r pE pN pD phi theta psi]
  Eigen::VectorXd getState12() const;

private:
  // Continuous-time dynamics: xdot = f(x)
  VecN f(const VecN& xs) const;

  // Numeric Jacobian of f
  MatN A_numeric(const VecN& xs, double eps = 1e-6) const;

private:
  // Time step
  double h_;

  // EKF core
  VecN x_;   // state
  MatN P_;   // covariance
  MatN Qd_;  // process noise (discrete)

  // NN interface
  std::unique_ptr<NN_INTERFACE_V5> nn_;

  // Cached orientation/gyro from NN for getState12()
  double nn_p_     = 0.0;
  double nn_q_     = 0.0;
  double nn_phi_   = 0.0;
  double nn_theta_ = 0.0;
};

#endif
#ifndef NN_EKF_V1_HPP
#define NN_EKF_V1_HPP

#pragma once
#include <Eigen/Dense>
#include <optional>

/**
 * NN_EKF_V1
 * 18-state EKF that fuses:
 *  - High-rate NN outputs: BODY velocities/rates [+ diagonal variances]
 *  - Slow GNSS: END position (E,N,[D optional]) and heading
 *
 * State x = [ u v w p q r  pE pN pD  phi theta psi  bu bv bw  bp bq br ]^T
 *
 * Frames:
 *  - BODY: x forward, y starboard, z down
 *  - Global: END (East, North, Down)
 *
 * Uses external helpers you already defined:
 *   Eigen::Matrix3d Rzyx(double phi, double theta, double psi); // body->END
 *   Eigen::Matrix3d Tzyx(double phi, double theta);
 *   double ssa(double a); // smallest-signed-angle
 */
class NN_EKF_V1 {
public:
  static constexpr int NX = 18;
  static constexpr int NY_NN = 6;   // [u v w p q r]
  static constexpr int NY_POS = 2;  // [pE pN]
  static constexpr int NY_POS3 = 3; // optional [pE pN pD]
  static constexpr int NY_HEAD = 1; // [psi]

  using VecN = Eigen::Matrix<double,NX,1>;
  using MatN = Eigen::Matrix<double,NX,NX>;

  NN_EKF_V1();

  // Config
  void setDt(double h);
  void setGravity(double g);
  void setState(const VecN& x0);           // set initial state
  void setCovariance(const MatN& P0);      // set initial covariance
  void setProcessNoise(const MatN& Qd);    // discrete process noise (per step)
  const VecN&  state() const;
  const MatN&  covariance() const;

  // Time update (predict) — no direct IMU used; all inputs come via measurement updates
  void predict();

  // High-rate NN measurement: BODY velocities & rates + diagonal variances
  // mu_uvwpr: [u v w p q r]^T (means), var_uvwpr: [σ_u^2 ... σ_r^2]^T
  void updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                const Eigen::Matrix<double,6,1>& var_uvwpr);

  // Slow GNSS (position EN only). D is ignored with large variance.
  void updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN);

  // Optional 3D position update
  void updatePosEND(const Eigen::Vector3d& pEND, const Eigen::Matrix3d& R_END);

  // Heading update (use ssa for innovation)
  void updateHeading(double psi, double R_psi);

  // Export EKF18-compatible 12-state vector:
  // [u v w p q r  x(E) y(N) z(D)  phi theta psi]^T
  Eigen::VectorXd getState12() const;

private:
  // Continuous dynamics xdot = f(x). Discretized with forward Euler (as in your EKF13).
  VecN f(const VecN& xs) const;

  // Numerical Jacobian for A = ∂f/∂x
  MatN A_numeric(const VecN& xs, double eps=1e-7) const;

private:
  // Sampling and constants
  double h_;     // step (s)
  double g_;     // gravity (m/s^2)

  // Filter state
  VecN  x_;
  MatN  P_;

  // Noise
  MatN  Qd_;     // discrete process noise per step
};

class NNObserver {
public:
  NNObserver();
  ~NNObserver();

  // Auto-select device, then load (tries given path, env override, and fallbacks).
  void load(const std::string& path);
  void load(const std::string& path, bool use_cuda_request);

  // Clear recurrent state (only used for stateless model; safe to call always).
  void reset();

  // One-step prediction: returns (mu[6], var[6])
  std::pair<std::array<double,6>, std::array<double,6>>
  predict_uvwpqr(const std::array<double,6>& imu) const;

private:
  struct Impl;
  Impl* p_;
};


#endif

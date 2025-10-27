#ifndef NN_EKF_V4_HPP
#define NN_EKF_V4_HPP

// ============================================================================
// nn_ekf_v4.hpp
// ----------------------------------------------------------------------------
// EKF v4 using NN observer v4 outputs.
// States (NX=8):
//   x = [ Vn, Ve, r, pE, pN, psi, bVn, bVe ]^T
// where pE (East), pN (North), psi (yaw). bVn/bVe are velocity bias states
// that are ONLY corrected indirectly when GNSS updates position/heading.
//
// NN provides at each IMU step (7D):
//   [Vn, Ve, p, q, r, phi, theta] and predictive covariance.
// EKF update uses ONLY [Vn, Ve, r]; the rest (p, q, phi, theta) are cached
// and surfaced via getState12().
//
// Frames:
//   Body (x fwd, y starboard, z down)
//   END  (x east, y north, z down)
// Rotation: v_nav = Rzyx(phi, theta, psi) * v_body
// ----------------------------------------------------------------------------
// Keep heavy deps out of the header.
// ============================================================================
#pragma once
#include <array>
#include <memory>
#include <string>
#include <utility>     // std::pair
#include <cmath>       // std::cos/sin
#include <Eigen/Core>
#include <Eigen/Dense>

// Forward decls (you provide these elsewhere)
Eigen::Matrix3d Rzyx(double phi, double theta, double psi);
Eigen::Matrix3d Tzyx(double phi, double theta);
double ssa(double a);

// ===================================================================================
//                                   NN Interface V4
// ===================================================================================

struct NN_INTERFACE_V4 {
  struct Impl;  // PIMPL

  NN_INTERFACE_V4();
  ~NN_INTERFACE_V4();

  void load(const std::string& model_path, bool use_cuda_request);
  void reset();

  // Helper: from heading to cos/sin
  static inline std::pair<double,double> QuatPsi(double psi) {
    return {std::cos(psi), std::sin(psi)};
  }

  // Predict: inputs [ax,ay,az, wx,wy,wz, cpsi,spsi] -> (μ[7], Σ[7x7])
  std::pair<std::array<double,7>, Eigen::Matrix<double,7,7>>
  predict(const std::array<double,8>& x_in) const;

private:
  Impl* p_;
};

// ===================================================================================
//                                       EKF v4
// ===================================================================================
class NN_EKF_V4 {
public:
  static constexpr int NX = 8;
  using VecN = Eigen::Matrix<double, NX, 1>;
  using MatN = Eigen::Matrix<double, NX, NX>;

  NN_EKF_V4();

  // Configuration
  void setDt(double h);
  void setState(const VecN& x0);
  void setCovariance(const MatN& P0);
  void setProcessNoise(const MatN& Qd);

  const VecN& state() const { return x_; }
  const MatN& covariance() const { return P_; }

  // NN init / control
  void initNN(const std::string& model_path, bool use_cuda_request);
  void resetNN();
  bool hasNN() const { return static_cast<bool>(nn_); }

  // 1) Time update
  void predict();

  // 2) NN measurement update — pass RAW inputs [ax,ay,az,wx,wy,wz,cpsi,spsi]
  //    Internally calls the NN, stores (p,q,phi,theta), and updates EKF with [Vn,Ve,r].
  void updateFromIMU(const std::array<double,8>& imu8);

  // 3) GNSS updates
  void updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN);
  void updateHeading(double psi_meas, double R_psi);

  // Legacy 12-state view: [u v w p q r pE pN pD phi theta psi]
  // u,v,w are back-computed from nav-frame vel (Ve, Vn, Vd=0) using latest (phi,theta) from NN and psi from EKF.
  Eigen::VectorXd getState12() const;

private:
  // Continuous dynamics xdot = f(x)
  VecN f(const VecN& x) const;
  MatN A_numeric(const VecN& x, double eps = 1e-6) const;

  // Sub-select NN outputs and apply EKF update with z = [Vn, Ve, r]
  void updateNN(const std::array<double,7>& mu7, const Eigen::Matrix<double,7,7>& Cov7);

private:
  // Filter data
  double h_;     // sample time
  VecN   x_;     // state
  MatN   P_;     // covariance
  MatN   Qd_;    // discrete process noise

  // NN
  std::unique_ptr<NN_INTERFACE_V4> nn_;

  // Cached latest NN extras (used in getState12)
  double nn_p_ = 0.0;
  double nn_q_ = 0.0;
  double nn_phi_ = 0.0;
  double nn_theta_ = 0.0;
};

#endif // NN_EKF_V4_HPP

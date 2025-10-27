#ifndef NN_EKF_V3_HPP
#define NN_EKF_V3_HPP

#pragma once
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>

// ===================================================================================
//                               Neural Net Interface (v3)
// ===================================================================================
//
// TorchScript exports (stateless preferred) take RAW IMU (ax,ay,az,wx,wy,wz)
// and return PHYSICAL outputs. Single-model exports usually return log-variance;
// ensemble exports return variance directly. This interface normalizes that for C++.
//
class NN_INTERFACE_V3 {
public:
  NN_INTERFACE_V3();
  ~NN_INTERFACE_V3();

  // Load TorchScript (single or ensemble). Auto-selects CPU/GPU (if requested & available).
  void load(const std::string& model_path, bool use_cuda_request = true);

  // Optional legacy normalization support (kept for back-compat; v3 TS handles norm internally).
  void set_normalization(const std::array<double,6>& in_mu,
                         const std::array<double,6>& in_std,
                         const std::array<double,6>& out_mu,
                         const std::array<double,6>& out_std);
  bool load_normalization_json(const std::string& json_path);

  // Clear recurrent state (safe always).
  void reset();

  // One-step prediction: imu=[ax,ay,az,wx,wy,wz] (physical units)
  // Returns μ (physical [u,v,w,p,q,r]) and R (6x6 physical covariance; diagonal-filled if needed).
  std::pair<std::array<double,6>, Eigen::Matrix<double,6,6>>
  predict_uvwpqr(const std::array<double,6>& imu) const;

private:
  struct Impl;
  Impl* p_;
};


// ===================================================================================
//                                       EKF (v3)
// ===================================================================================
//
// State (NX = 10):
//   0:u, 1:v, 2:p, 3:q, 4:r, 5:pE, 6:pN, 7:phi, 8:theta, 9:psi
//
// Measurements:
//   - NN (5-dim): [u, v, p, q, r]  (we DROP w)
//   - GNSS pos  : [pE, pN]
//   - Heading   : [psi]
//
class NN_EKF_V3 {
public:
  static constexpr int NX     = 10;
  static constexpr int NY_NN  = 5; // [u v p q r] (w dropped)
  static constexpr int NY_POS = 2; // [pE pN]
  static constexpr int NY_HEAD= 1; // [psi]

  using VecN = Eigen::Matrix<double,NX,1>;
  using MatN = Eigen::Matrix<double,NX,NX>;

  NN_EKF_V3();

  // -------- Config / lifecycle --------
  void setDt(double h);
  void setState(const VecN& x0);
  void setCovariance(const MatN& P0);
  void setProcessNoise(const MatN& Qd);
  const VecN&  state() const;
  const MatN&  covariance() const;

  // Own & manage the neural net inside the EKF
  void initNN(const std::string& model_path,
              bool use_cuda_request = true,
              const std::string& norm_json = "");
  void setNNNormalization(const std::array<double,6>& in_mu,
                          const std::array<double,6>& in_std,
                          const std::array<double,6>& out_mu,
                          const std::array<double,6>& out_std);
  void resetNN();
  bool hasNN() const;

  // -------- Recurring updates --------
  void predict();

  // High-rate IMU → NN infer → EKF measurement update (one call)
  void updateFromIMU(const std::array<double,6>& imu);

  // If you already have NN outputs, call these directly (μ is 6, R is 6x6 or diag):
  void updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                const Eigen::Matrix<double,6,6>& R_uvwpr);
  void updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                const Eigen::Matrix<double,6,1>& var_uvwpr); // diag compat

  // Slow GNSS updates
  void updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN);
  void updateHeading(double psi, double R_psi);

  // Historical name; now returns 10 states (no w, no z):
  // [u v p q r pE pN phi theta psi]
  Eigen::VectorXd getState12() const;

private:
  // Dynamics (body → nav kinematics; w ignored)
  VecN f(const VecN& xs) const;
  MatN A_numeric(const VecN& xs, double eps=1e-7) const;

private:
  // Sampling
  double h_;

  // Filter state
  VecN  x_;
  MatN  P_;

  // Noise
  MatN  Qd_;

  // Embedded NN model
  std::unique_ptr<NN_INTERFACE_V3> nn_;
};

#endif // NN_EKF_V3_HPP

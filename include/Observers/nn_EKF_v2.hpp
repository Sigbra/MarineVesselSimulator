#ifndef NN_EKF_V2_HPP
#define NN_EKF_V2_HPP

#pragma once
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>

// ===================================================================================
//                               Neural Net Interface (v2)
// ===================================================================================
//
// v2 TorchScript exports accept RAW IMU (physical) and return PHYSICAL μ=[u,v,w,p,q,r]
// and an uncertainty head:
//   * single-model TS: (mu_phys, logvar_phys)  -> var = exp(logvar_phys)
//   * ensemble   TS  : (mu_phys, cov_phys)     -> cov already in PHYSICAL units
//
// API keeps JSON normalization loader for backward compatibility; v2 TS already
// handles normalization internally.
//
class NN_INTERFACE_V2 {
public:
  NN_INTERFACE_V2();
  ~NN_INTERFACE_V2();

  // Load TorchScript (single or ensemble). Auto-selects CPU/GPU (if requested & available).
  void load(const std::string& model_path, bool use_cuda_request = true);

  // Normalization setters kept for backward-compat (no-op for v2 TS at runtime).
  void set_normalization(const std::array<double,6>& in_mu,
                         const std::array<double,6>& in_std,
                         const std::array<double,6>& out_mu,
                         const std::array<double,6>& out_std);

  // Try to parse a Python-style stats JSON. Returns true on success.
  bool load_normalization_json(const std::string& json_path);

  // Clear recurrent state (safe always).
  void reset();

  // One-step prediction: imu=[ax,ay,az,wx,wy,wz] (physical units)
  // Returns μ (physical [u,v,w,p,q,r]) and R (6x6 physical covariance; diagonal-filled).
  std::pair<std::array<double,6>, Eigen::Matrix<double,6,6>>
  predict_uvwpqr(const std::array<double,6>& imu) const;

private:
  struct Impl;
  Impl* p_;
};


// ===================================================================================
//                                       EKF (v2)
// ===================================================================================
//
// State (NX=18):
//   0..2:  u,v,w          (body-frame linear vel, unbiased)
//   3..5:  p,q,r          (body-frame angular rates, unbiased)
//   6..8:  pE,pN,pD       (ECEF/Local-ENU position; here ENU: E,N,Down)
//   9..11: phi,theta,psi  (roll, pitch, yaw; ZYX)
//   12..14: b_u,b_v,b_w   (measurement biases for u,v,w)
//   15..17: b_p,b_q,b_r   (measurement biases for p,q,r)
//
// NN measurement (per step):
//   z_NN ≈ [u+b_u, v+b_v, w+b_w, p+b_p, q+b_q, r+b_r] + noise
//
// Bias adaptation gating:
//   - Bias states are only corrected when absolute aiding (GNSS/heading) has been
//     observed recently. We implement this by zeroing the bias columns in H_NN
//     when "no recent aiding" is true (freezing biases).
//
class NN_EKF_V2 {
public:
  static constexpr int NX = 18;
  static constexpr int NY_NN = 6;   // [u v w p q r]
  static constexpr int NY_POS = 2;  // [pE pN]
  static constexpr int NY_POS3 = 3; // [pE pN pD]
  static constexpr int NY_HEAD = 1; // [psi]

  using VecN = Eigen::Matrix<double,NX,1>;
  using MatN = Eigen::Matrix<double,NX,NX>;

  NN_EKF_V2();

  // -------- Config / lifecycle --------
  void setDt(double h);
  void setGravity(double g);
  void setState(const VecN& x0);
  void setCovariance(const MatN& P0);
  void setProcessNoise(const MatN& Qd);
  const VecN&  state() const;
  const MatN&  covariance() const;

  // Bias gating configuration (seconds since last absolute aiding within which
  // bias adaptation is allowed). Default: 2.0 s
  void setBiasAdaptationWindow(double seconds);
  double biasAdaptationWindow() const;

  // Own & manage the neural net inside the EKF (v2 with normalization)
  void initNN(const std::string& model_path,
              bool use_cuda_request = true,
              const std::string& norm_json = "");
  // Optional explicit normalization (kept for compatibility)
  void setNNNormalization(const std::array<double,6>& in_mu,
                          const std::array<double,6>& in_std,
                          const std::array<double,6>& out_mu,
                          const std::array<double,6>& out_std);

  void resetNN();
  bool hasNN() const;

  // -------- Recurring updates --------
  void predict();

  // High-rate: IMU → NN infer → EKF measurement update
  void updateFromIMU(const std::array<double,6>& imu);

  // If you already have NN outputs, call these directly:
  void updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                const Eigen::Matrix<double,6,6>& R_uvwpr);
  void updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                const Eigen::Matrix<double,6,1>& var_uvwpr); // diag compat

  // Slow GNSS updates (absolute aiding) — these also reset the bias gating timer
  void updatePosEN (const Eigen::Vector2d& pEN,  const Eigen::Matrix2d& R_EN);
  void updatePosEND(const Eigen::Vector3d& pEND, const Eigen::Matrix3d& R_END);
  void updateHeading(double psi, double R_psi);

  // Export EKF18-compatible 12-state vector:
  Eigen::VectorXd getState12() const;

private:
  // Dynamics
  VecN f(const VecN& xs) const;
  MatN A_numeric(const VecN& xs, double eps=1e-7) const;

  // Bias gating helper
  bool allowBiasAdaptation() const;

private:
  // Sampling and constants
  double h_;
  double g_;

  // Filter state
  VecN  x_;
  MatN  P_;

  // Noise
  MatN  Qd_;

  // Embedded NN model
  std::unique_ptr<NN_INTERFACE_V2> nn_;

  // Bias gating
  double bias_adapt_window_s_;
  long   steps_since_abs_aid_;   // incremented at predict(), reset on GNSS/heading updates
};

#endif // NN_EKF_V2_HPP

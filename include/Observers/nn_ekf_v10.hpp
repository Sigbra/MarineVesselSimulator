#ifndef NN_EKF_V10_HPP
#define NN_EKF_V10_HPP
#pragma once

// nn_ekf_v10.hpp — 9-state EKF (“NN_qObs_Aided_EKF_v10”) with external attitude (q-Obs)
// + Velocity-only NN TorchScript wrapper (PIMPL) for END-frame velocity aiding.
// v10 difference from v9: adds hydrostatic heave spring–damper to the process model.
// Everything else (interfaces, states, updates, gating) remains identical to v9.

#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>  // Quaterniond

namespace nnqekf_v10
{

// ------------------------------- Aliases -------------------------------
using Vec3  = Eigen::Vector3d;
using Mat3  = Eigen::Matrix3d;
using Mat99 = Eigen::Matrix<double, 9, 9>;
using Mat96 = Eigen::Matrix<double, 9, 6>;

// -------------------------------- State --------------------------------
struct State9_v10 {
  Vec3 p;    // position in END (E,N,D; Down positive)
  Vec3 v;    // velocity in END
  Vec3 b_a;  // accel bias in BODY
  State9_v10(){ p.setZero(); v.setZero(); b_a.setZero(); }
};

// ------------------------------- Config --------------------------------
struct Config_v10 {
  double g               = 9.81;  // gravity magnitude (Down positive in END)
  double sigma_a         = 0.20;  // accel white noise [m/s^2]/√Hz (BODY)
  double sigma_ba_rw     = 0.001; // accel bias random walk [m/s^2]/√Hz
  double tau_ba          = 0.0;   // optional accel-bias leak time constant [s] (0=off)
  double chi2_gate_vec3  = 0.0;   // NIS gate for 3D velocity updates (0=off)
  double chi2_gate_pos3  = 0.0;   // NIS gate for 3D position updates (0=off)
};

// ------------------------------ NN wrapper ------------------------------
class NN_v10 {
public:
  NN_v10();
  ~NN_v10();

  // model_dir: directory or single .pt/.ts file (TorchScript).
  // norm_json: YAML/JSON file with x_mean/x_std/y_mean/y_std (same as v9).
  // use_cuda : request CUDA if available.
  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda);

  // window: last T rows, each 10x1 = [ax ay az qw qx qy qz tau_x tau_y tau_n]^T
  // Outputs: v_nav_mean in END ([vE vN vD]) and covariance Rv_nav (3x3).
  bool infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
             Vec3& v_nav_mean,
             Mat3& Rv_nav);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// --------------------------------- EKF ---------------------------------
class NN_qObs_Aided_EKF_v10 {
public:
  explicit NN_qObs_Aided_EKF_v10(const Config_v10& cfg);

  // External attitude
  void setRotationNavFromBody(const Mat3& R_nb);                 // direct R_nb (BODY->END)
  void setRotationFromQuat(const Eigen::Quaterniond& q_nb);      // uses custom END quaternion

  // Heave equilibrium: choose the Down reference z0 (default 0). Units: meters (END Down +).
  void setHeaveEquilibrium(double z0);

  // Attach NN (window length & stride as in v9)
  void setNN(NN_v10* nn, int seq_len, int stride);

  // Feed one sample to NN buffer (same inputs as training; quaternion normalized)
  void feedNN(const Vec3& accel_b,
              const Eigen::Quaterniond& q_nb,
              double tau_x,
              double tau_y,
              double tau_n);

  // Time update with IMU (BODY) — process model now includes heave spring–damper (v10)
  void propagate(const Vec3& omega_b_meas,
                 const Vec3& accel_b_meas,
                 double dt);

  // Measurement updates (identical to v9)
  bool updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w=1.0);
  bool updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                     const Vec3& omega_b_meas, double w=1.0);
  bool updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w=1.0);
  bool updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w=1.0);

  // Accessors
  const State9_v10& state() const;
  const Mat99&      cov()   const;
  void setState(const State9_v10& x, const Mat99& P);

  // Convenience: legacy 12-state vector [u v w p q r x y z phi theta psi]^T
  Eigen::VectorXd getState12(const Vec3& b_gyro_hat) const;

  // Convenience: 9-vector state (matches v9 signatures some callers expect)
  Eigen::Matrix<double,9,1> getState9() const;

private:
  // Shared EKF helper (unchanged)
  bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                    const Mat3& R, double chi2_gate);

  // NN buffer maintenance (unchanged)
  void nn_prune_();

private:
  // --- configuration & state ---
  Config_v10 cfg_;
  State9_v10 x_;
  Mat99 P_;

  // Attitude (BODY->END) and gravity vector (0,0,g) with Down positive
  Mat3 R_nb_;
  Vec3 g_n_;

  // Last gyro measurement (cached)
  Vec3 last_omega_b_meas_{Vec3::Zero()};

  // --- NN aiding ---
  NN_v10* nn_ = nullptr;
  int nn_seq_len_ = 0;
  int nn_stride_  = 1;
  int nn_count_   = 0;
  std::deque<Eigen::Matrix<double,10,1>> nn_buf_;

  // --- v10 heave spring–damper parameters (RAN “option A”) ---
  // a_D += -k_z_ * (z - z0_) - c_z_ * v_D
  // A(5,2) += -k_z_,  A(5,5) += -c_z_   in the linearized model
  double k_z_ = 0.0;   // [1/s^2]  = (rho*g*2*Cw*L*Beam_pont) / m
  double c_z_ = 0.0;   // [1/s]    = (2*zeta*sqrt(m*K_h)) / m
  double z0_  = 0.0;   // [m]      heave equilibrium (END Down+)

  Eigen::Quaterniond q_last_{1,0,0,0};
  bool q_last_valid_{false};
  bool assume_q_is_nav_to_body_{false};  

};

} // namespace nnqekf_v10

#endif // NN_EKF_V10_HPP

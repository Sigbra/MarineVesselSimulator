#ifndef NN_EKF_V8_HPP
#define NN_EKF_V8_HPP

#pragma once

// nn_ekf_v8.hpp — 9-state EKF (“NN_qObs_Aided_EKF_v8”) with external attitude (q-Obs)
// + Velocity-only NN_v8 TorchScript wrapper (PIMPL) providing prev BODY velocity to NN input.
//
// Differences vs v7:
//   - NN_v8 takes IN_DIM=10: [ax,ay,az, qw,qx,qy,qz, u_prev,v_prev,w_prev]
//   - EKF feeds previous BODY velocity to NN interface.

#include <deque>
#include <memory>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace nnqekf_v8 {

// ---------- handy typedefs ----------
using Vec3  = Eigen::Matrix<double,3,1>;
using Mat3  = Eigen::Matrix<double,3,3>;
using Mat96 = Eigen::Matrix<double,9,6>;
using Mat99 = Eigen::Matrix<double,9,9>;

// ---------- Config_v8 & state ----------
struct Config_v8 {
  double g              = 9.81;   // gravity magnitude (Down positive)
  double sigma_a        = 0.5;    // accel white noise [m/s^2]/√Hz
  double sigma_ba_rw    = 0.01;   // accel bias RW [m/s^2]/√Hz
  double tau_ba         = 0.0;    // bias leak time constant (0=off)
  double chi2_gate_pos3 = 0.0;    // NIS gate for 3D position (0=off)
  double chi2_gate_vec3 = 0.0;    // NIS gate for 3D vectors (0=off)
};

struct State9_v8 {
  Vec3 p;    // position in NAV (END)
  Vec3 v;    // velocity in NAV (END)
  Vec3 b_a;  // accel bias in BODY
};

// ============================== NN_v8 (PIMPL) ==============================
class NN_v8 final {
public:
  NN_v8();
  ~NN_v8();

  // model_dir: directory containing member_XX.pt TorchScript files (or a single .pt)
  // norm_json: YAML/JSON with {x_mean, x_std, y_mean, y_std}; x_* are 10-dim
  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda);

  // window: last T rows, each = [ax,ay,az,qw,qx,qy,qz,u_prev,v_prev,w_prev]^T
  // outputs: mean END velocity (vE,vN,vD) + covariance across ensemble
  bool infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
             Vec3& v_nav_mean,
             Mat3& Rv_nav);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ================================ EKF v8 ===================================
class NN_qObs_Aided_EKF_v8 {
public:
  explicit NN_qObs_Aided_EKF_v8(const Config_v8& cfg);

  // attitude interface
  void setRotationNavFromBody(const Mat3& R_nb);                 // BODY→NAV (END)
  void setRotationFromQuat(const Eigen::Quaterniond& q_nb);      // BODY→NAV (END), custom convention

  // attach NN (sequence length and stride for batched inference)
  void setNN(NN_v8* nn, int seq_len, int stride);

  // feed one IMU + attitude sample to the NN buffer (adds u_prev,v_prev,w_prev internally)
  void feedNN(const Vec3& accel_b, const Eigen::Quaterniond& q_nb);

  // EKF propagation with IMU (accel/gyro), using current R_nb_
  void propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt);

  // measurement updates
  bool updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w=1.0);
  bool updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                     const Vec3& omega_b_meas, double w=1.0);
  bool updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w=1.0); // END aiding
  bool updateNNBodyVel(const Vec3& z_v_b,   const Mat3& Rvb, double w=1.0); // optional BODY aiding

  // accessors
  const State9_v8& state() const;
  const Mat99&  cov()   const;
  void setState(const State9_v8& x, const Mat99& P);

  // convenience exports
  Eigen::Matrix<double,9,1> getState9() const;
  Eigen::VectorXd           getState12(const Vec3& b_gyro_hat) const;

private:
  bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                    const Mat3& R, double chi2_gate);

  void nn_prune_();

private:
  // Config_v8 & filters
  Config_v8 cfg_;
  State9_v8 x_{};
  Mat99  P_{};
  Mat3   R_nb_{ Mat3::Identity() };  // BODY→NAV (END)
  Vec3   g_n_{0,0,9.81};             // Down positive
  Vec3   last_omega_b_meas_{0,0,0};

  // NN interface/buffer (v8)
  NN_v8* nn_ = nullptr;
  int nn_seq_len_ = 0;
  int nn_stride_  = 1;
  int nn_count_   = 0;
  std::deque<Eigen::Matrix<double,10,1>> nn_buf_;

  // cached previous BODY velocity for NN input (u_prev,v_prev,w_prev)
  Vec3 nn_prev_v_b_{0,0,0};
};

} // namespace nnqekf_v8

#endif
#ifndef NN_EKF_V9_HPP
#define NN_EKF_V9_HPP

#pragma once
// nn_ekf_v9.hpp — 9-state EKF (“NN_qObs_Aided_EKF_v9”) with external attitude (q-Obs)
// + Velocity-only NN_v9 TorchScript wrapper (PIMPL) for END-frame velocity aiding.
// v9 difference from v7: NN inputs add 3 scalar taus: tau_x, tau_y, tau_n.
// Inputs per sample: [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n] (10 total).
// No quaternion packing for accel or tau; only the quat is a quaternion.

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <vector>
#include <string>

namespace nnqekf_v9 {

// ---------- handy typedefs ----------
using Vec3  = Eigen::Vector3d;
using Mat3  = Eigen::Matrix3d;
using Mat96 = Eigen::Matrix<double,9,6>;
using Mat99 = Eigen::Matrix<double,9,9>;

// ---------- simple state container ----------
struct State9_v9 {
  Vec3 p = Vec3::Zero();   // position (END)
  Vec3 v = Vec3::Zero();   // velocity (END)
  Vec3 b_a = Vec3::Zero(); // accel bias (BODY)
};

// ---------- EKF config ----------
struct Config_v9 {
  double g = 9.81;            // gravity (Down positive in END)
  double sigma_a = 0.05;      // accel white noise [m/s^2]/√Hz
  double sigma_ba_rw = 1e-3;  // accel bias random walk
  double tau_ba = 0.0;        // bias leak time constant (<=0 disables)
  double chi2_gate_pos3 = -1.0; // 3D pos gate (<=0 disables)
  double chi2_gate_vec3 = -1.0; // 3D vec gate (<=0 disables)
};

// ====================== TorchScript NN wrapper ======================
class NN_v9 {
public:
  NN_v9();
  ~NN_v9();

  // model_dir: dir with member_XX.pt (or a single .pt file)
  // norm_json: JSON/YAML with keys x_mean, x_std, y_mean, y_std
  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda);

  // window: vector of 10x1 rows: [ax,ay,az,qw,qx,qy,qz,tau_x,tau_y,tau_n]
  bool infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
             Vec3& v_nav_mean,
             Mat3& Rv_nav);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ============================ EKF core ==============================
class NN_qObs_Aided_EKF_v9 {
public:
  explicit NN_qObs_Aided_EKF_v9(const Config_v9& cfg);

  // attitude interface
  void setRotationNavFromBody(const Mat3& R_nb);
  void setRotationFromQuat(const Eigen::Quaterniond& q_nb);

  // NN interface (sequence length & stride for batched inference)
  void setNN(NN_v9* nn, int seq_len, int stride);

  // Feed one sample to the NN buffer (accel in BODY, attitude as quaternion, and 3 τ scalars).
  // The NN runs when enough samples are buffered and stride is met.
  void feedNN(const Vec3& accel_b,
              const Eigen::Quaterniond& q_nb,
              double tau_x,
              double tau_y,
              double tau_n);

  // Propagate with IMU (BODY frame) and dt
  void propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt);

  // GNSS updates (END frame)
  bool updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w=1.0);
  bool updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                     const Vec3& omega_b_meas, double w=1.0);

  // Body-velocity pseudo-measurement update (BODY frame)
  bool updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w=1.0);

  // Accessors
  const State9_v9& state() const;
  const Mat99&     cov()   const;
  void setState(const State9_v9& x, const Mat99& P);

  // Convenience getters
  Eigen::Matrix<double,9,1> getState9() const;
  Eigen::VectorXd           getState12(const Vec3& b_gyro_hat) const;

private:
  // Helper: trim NN buffer
  void nn_prune_();

  // EKF update with z_v_nav (END frame) from NN
  bool updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w);

  // generic Kalman update
  bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                    const Mat3& R, double chi2_gate);

private:
  Config_v9 cfg_;
  State9_v9 x_;
  Mat99 P_{Mat99::Identity()};
  Mat3  R_nb_{Mat3::Identity()}; // BODY->NAV (END)
  Vec3  g_n_{0,0,9.81};

  // last gyro (for convenience output)
  Vec3 last_omega_b_meas_ = Vec3::Zero();

  // NN
  NN_v9* nn_ = nullptr;
  int nn_seq_len_ = 0;
  int nn_stride_  = 1;
  int nn_count_   = 0;

  std::deque<Eigen::Matrix<double,10,1>> nn_buf_;
};

} // namespace nnqekf_v9


#endif
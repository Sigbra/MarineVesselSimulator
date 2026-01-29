#pragma once
// nn_ekf_v11.hpp — 9-state EKF + stateful TorchScript velocity NN (v11)

#include <Eigen/Core>
#include <deque>
#include <memory>
#include <vector>
#include <string>
#include "Utilities/calculations.hpp"

namespace nnqekf_v11 {

using Vec3  = Eigen::Vector3d;
using Mat3  = Eigen::Matrix3d;
using Mat99 = Eigen::Matrix<double,9,9>;

// ----------------------- EKF config -----------------------
struct Config_v11 {
  double g              = 9.81;   // gravity magnitude (+Down)
  double sigma_a        = 0.05;   // accel white noise [m/s^2]
  double sigma_ba_rw    = 1e-3;   // accel bias random walk [m/s^2/s]
  double tau_ba         = 0.0;    // accel bias leak time-constant [s] (0 = none)
  double chi2_gate_pos3 = -1.0;   // NIS gate for position (<=0 disables)
  double chi2_gate_vec3 = -1.0;   // NIS gate for 3D vectors (<=0 disables)
};

// ----------------------- EKF state ------------------------
struct State9_v11 {
  Vec3 p{Vec3::Zero()};   // position (END)
  Vec3 v{Vec3::Zero()};   // velocity (END)
  Vec3 b_a{Vec3::Zero()}; // accel bias (BODY) mapped via R_nb in process model
};

// Forward-declare Quat from calculations utilities
using ::Quat;


// ===================== NN v11 wrapper =====================
class NN_v11 {
public:
  NN_v11();
  ~NN_v11();

  // model_dir: directory containing member_XX_onestep_stateful.pt files
  // norm_json: JSON/YAML with keys x_mean, x_std, y_mean, y_std (lengths 10 and 3)
  bool init(const std::string& model_dir, const std::string& norm_json, bool use_cuda);

  // Reset all members' hidden states to zeros via init_state(1)
  void reset_states();

  // Window is kept for API compat; the last row is used ([ax ay az qw qx qy qz tau_x tau_y tau_n])
  bool infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
             Vec3& v_nav_mean, Mat3& Rv_nav);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  enum class QuatInputPolicy { Hemisphere, SignContinuity };

  // Selected automatically from model_dir path in NN_v11::init()
  QuatInputPolicy quatInputPolicy() const;

};

// ===================== EKF v11 ============================
class NN_qObs_Aided_EKF_v11 {
public:
  explicit NN_qObs_Aided_EKF_v11(const Config_v11& cfg);

  // Set attitude directly (BODY -> END)
  void setRotationNavFromBody(const Mat3& R_nb);
  // Or from quaternion (BODY->END).
  void setRotationFromQuat(const Quat& q_nb);

  // State & covariance
  const State9_v11& state() const;
  const Mat99&      cov()   const;
  void setState(const State9_v11& x, const Mat99& P);

  // Force accel-bias estimate to zero (b_a := 0). Does not change covariance.
  void zeroAccelBias() { x_.b_a.setZero(); }

  // Optional: set heave equilibrium (Down) for spring–damper term
  void setHeaveEquilibrium(double z0);

  // Hook up NN and choose streaming cadence (seq_len kept for API; one-step NN uses last sample)
  void setNN(NN_v11* nn, int seq_len, int stride);

  // Feed one NN sample (BODY accel, BODY->END quat, body torques)
  void feedNN(const Vec3& accel_b,
              const Quat& q_nb,
              double tau_x, double tau_y, double tau_n);

  // Updates
  bool updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w=1.0);
  bool updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                     const Vec3& omega_b_meas, double w=1.0);
  bool updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w=1.0);
  bool updateNNBodyVel(const Vec3& z_v_b,  const Mat3& Rvb, double w=1.0);

  // Propagate INS with accel (BODY) and gravity (+Down)
  void propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt);

  // Convenience outputs
  Eigen::Vector3d get_acc_bias_est() const { return x_.b_a; }
  Eigen::Vector3d get_end_vel_est() const { return x_.v;}
  Eigen::Matrix<double,9,1>  getState9()  const;
  Eigen::Matrix<double,12,1> getState12(const Vec3& b_gyro_hat) const;

  // Generic Kalman injection (rarely used directly)
  bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                    const Mat3& R, double chi2_gate=-1.0);

private:
  void nn_prune_();

private:
  static Quat canonicalizeQuat(const Quat& q_in);

  Config_v11 cfg_;
  State9_v11 x_;
  Mat99 P_{Mat99::Identity()};

  Mat3 R_nb_{Mat3::Identity()};   // BODY -> END
  Vec3 g_n_{0,0,9.81};            // +Down

  // Heave spring–damper (optional)
  double k_z_{0.0}, c_z_{0.0}, z0_{0.0};

  // Last gyro for convenience outputs
  Vec3 last_omega_b_meas_{Vec3::Zero()};

  // NN
  NN_v11* nn_{nullptr};
  int nn_seq_len_{1};
  int nn_stride_{1};
  int nn_count_{0};
  std::deque<Eigen::Matrix<double,10,1>> nn_buf_;

  bool nn_warmed_ = false;

private:
  bool nn_use_sign_continuity_ = false;  // derived from NN_v11 model folder name
  bool nn_has_prev_q_nn_ = false;
  Quat nn_prev_q_nn_{1.0, 0.0, 0.0, 0.0};
};

} // namespace nnqekf_v11

// #pragma once
// // nn_ekf_v11.hpp — 9-state EKF + stateful TorchScript velocity NN (v11)

// #include <Eigen/Core>
// #include <Eigen/Geometry>
// #include <deque>
// #include <memory>
// #include <vector>
// #include <string>

// namespace nnqekf_v11 {

// using Vec3  = Eigen::Vector3d;
// using Mat3  = Eigen::Matrix3d;
// using Mat99 = Eigen::Matrix<double,9,9>;

// // ----------------------- EKF config -----------------------
// struct Config_v11 {
//   double g              = 9.81;   // gravity magnitude (+Down)
//   double sigma_a        = 0.05;   // accel white noise [m/s^2]
//   double sigma_ba_rw    = 1e-3;   // accel bias random walk [m/s^2/s]
//   double tau_ba         = 0.0;    // accel bias leak time-constant [s] (0 = none)
//   double chi2_gate_pos3 = -1.0;   // NIS gate for position (<=0 disables)
//   double chi2_gate_vec3 = -1.0;   // NIS gate for 3D vectors (<=0 disables)
// };

// // ----------------------- EKF state ------------------------
// struct State9_v11 {
//   Vec3 p{Vec3::Zero()};   // position (END)
//   Vec3 v{Vec3::Zero()};   // velocity (END)
//   Vec3 b_a{Vec3::Zero()}; // accel bias (BODY) mapped via R_nb in process model
// };

// // ===================== NN v11 wrapper =====================
// class NN_v11 {
// public:
//   NN_v11();
//   ~NN_v11();

//   // model_dir: directory containing member_XX_onestep_stateful.pt files
//   // norm_json: JSON/YAML with keys x_mean, x_std, y_mean, y_std (lengths 10 and 3)
//   bool init(const std::string& model_dir, const std::string& norm_json, bool use_cuda);

//   // Reset all members' hidden states to zeros via init_state(1)
//   void reset_states();

//   // Window is kept for API compat; the last row is used ([ax ay az qw qx qy qz tau_x tau_y tau_n])
//   bool infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
//              Vec3& v_nav_mean, Mat3& Rv_nav);

// private:
//   struct Impl;
//   std::unique_ptr<Impl> impl_;
// };

// // ===================== EKF v11 ============================
// class NN_qObs_Aided_EKF_v11 {
// public:
//   explicit NN_qObs_Aided_EKF_v11(const Config_v11& cfg);

//   // Set attitude directly (BODY -> END)
//   void setRotationNavFromBody(const Mat3& R_nb);
//   // Or from quaternion (BODY->END). If your input is NAV->BODY, pass assume flag via cfg if needed.
//   void setRotationFromQuat(const Eigen::Quaterniond& q_nb);

//   // State & covariance
//   const State9_v11& state() const;
//   const Mat99&      cov()   const;
//   void setState(const State9_v11& x, const Mat99& P);

//   // Optional: set heave equilibrium (Down) for spring–damper term
//   void setHeaveEquilibrium(double z0);

//   // Hook up NN and choose streaming cadence (seq_len kept for API; one-step NN uses last sample)
//   void setNN(NN_v11* nn, int seq_len, int stride);

//   // Feed one NN sample (BODY accel, BODY->END quat, body torques)
//   void feedNN(const Vec3& accel_b,
//               const Eigen::Quaterniond& q_nb,
//               double tau_x, double tau_y, double tau_n);

//   // Updates
//   bool updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w=1.0);
//   bool updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
//                      const Vec3& omega_b_meas, double w=1.0);
//   bool updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w=1.0);
//   bool updateNNBodyVel(const Vec3& z_v_b,  const Mat3& Rvb, double w=1.0);

//   // Propagate INS with accel (BODY) and gravity (+Down)
//   void propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt);

//   // Convenience outputs
//   Eigen::Matrix<double,9,1>  getState9()  const;
//   Eigen::Matrix<double,12,1> getState12(const Vec3& b_gyro_hat) const;

//   // Generic Kalman injection (rarely used directly)
//   bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
//                     const Mat3& R, double chi2_gate=-1.0);

// private:
//   void nn_prune_();
//   static Eigen::Quaterniond canonicalizeQuat(const Eigen::Quaterniond& q_in);

// private:
//   Config_v11 cfg_;
//   State9_v11 x_;
//   Mat99 P_{Mat99::Identity()};

//   Mat3 R_nb_{Mat3::Identity()};   // BODY -> END
//   Vec3 g_n_{0,0,9.81};            // +Down

//   // Heave spring–damper (optional)
//   double k_z_{0.0}, c_z_{0.0}, z0_{0.0};

//   // Last gyro for convenience outputs
//   Vec3 last_omega_b_meas_{Vec3::Zero()};

//   // NN
//   NN_v11* nn_{nullptr};
//   int nn_seq_len_{1};
//   int nn_stride_{1};
//   int nn_count_{0};
//   std::deque<Eigen::Matrix<double,10,1>> nn_buf_;
// };

// } // namespace nnqekf_v11
#pragma once
// nn_ekf_v12.hpp — 9-state EKF + stateful TorchScript velocity NN (v12, Euler input)
//
// v12 NN interface (Euler-only, wrapped angles):
//  - EKF buffers a canonical 10-vector per step (API kept stable):
//      [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n, 0]
//  - NN input dimension is inferred from norm_stats.json (x_mean length) and is expected to be:
//      D=9 -> [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n]
//  - Angles are wrapped to [-pi, +pi) using ssa() (mapped from (-pi, pi]) before normalization.
//  - Streaming / warm-start semantics match v11:
//      * do not call infer() until nn_seq_len samples exist
//      * warm-start once by feeding last nn_seq_len window
//      * thereafter feed only newest sample every tick
//      * apply EKF update only on stride

#include <Eigen/Core>
#include <deque>
#include <memory>
#include <vector>
#include <string>
#include "Utilities/calculations.hpp"

namespace nnqekf_v12 {

using Vec3  = Eigen::Vector3d;
using Mat3  = Eigen::Matrix3d;
using Mat99 = Eigen::Matrix<double,9,9>;

// ----------------------- EKF config -----------------------
struct Config_v12 {
  double g              = 9.81;   // gravity magnitude (+Down)
  double sigma_a        = 0.05;   // accel white noise [m/s^2]
  double sigma_ba_rw    = 1e-3;   // accel bias random walk [m/s^2/s]
  double tau_ba         = 0.0;    // accel bias leak time-constant [s] (0 = none)
  double chi2_gate_pos3 = -1.0;   // NIS gate for position (<=0 disables)
  double chi2_gate_vec3 = -1.0;   // NIS gate for 3D vectors (<=0 disables)
};

// ----------------------- EKF state ------------------------
struct State9_v12 {
  Vec3 p{Vec3::Zero()};   // position (END)
  Vec3 v{Vec3::Zero()};   // velocity (END)
  Vec3 b_a{Vec3::Zero()}; // accel bias (BODY) mapped via R_nb in process model
};

// Forward-declare Quat from calculations utilities
using ::Quat;


// ===================== NN v12 wrapper =====================
class NN_v12 {
public:
  NN_v12();
  ~NN_v12();

  // model_dir: directory containing member_XX_onestep_stateful.pt files
  // norm_json: JSON/YAML with keys x_mean, x_std, y_mean, y_std (lengths D and 3)
  // v12 expects D=9 for Euler-only inputs.
  bool init(const std::string& model_dir, const std::string& norm_json, bool use_cuda);

  // Reset all members' hidden states (clears cached hidden tensors so next infer() re-inits via init_state(1))
  void reset_states();

  // Window is kept for API compat; v12 uses Euler inputs packed into the row:
  //   row = [ax ay az phi theta psi tau_x tau_y tau_n 0]
  // The implementation maps to the model input dimension D=9.
  bool infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
             Vec3& v_nav_mean, Mat3& Rv_nav);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};


// ===================== EKF v12 ============================
class NN_qObs_Aided_EKF_v12 {
public:
  explicit NN_qObs_Aided_EKF_v12(const Config_v12& cfg);

  // Set attitude directly (BODY -> END)
  void setRotationNavFromBody(const Mat3& R_nb);
  // Or from quaternion (BODY->END).
  void setRotationFromQuat(const Quat& q_nb);

  // State & covariance
  const State9_v12& state() const;
  const Mat99&      cov()   const;
  void setState(const State9_v12& x, const Mat99& P);

  // Force accel-bias estimate to zero (b_a := 0). Does not change covariance.
  void zeroAccelBias() { x_.b_a.setZero(); }

  // Optional: set heave equilibrium (Down) for spring–damper term
  void setHeaveEquilibrium(double z0);

  // Hook up NN and choose streaming cadence.
  // Semantics match v11: warm-start once on last seq_len samples, then 1-step each tick.
  void setNN(NN_v12* nn, int seq_len, int stride);

  // Feed one NN sample (BODY accel, BODY->END quat, body torques).
  // v12 derives Euler angles from q_nb (END convention) and wraps via ssa() to [-pi, +pi).
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
  Eigen::Vector3d get_end_vel_est() const { return x_.v;  }
  Eigen::Matrix<double,9,1>  getState9()  const;
  Eigen::Matrix<double,12,1> getState12(const Vec3& b_gyro_hat) const;

  // Generic Kalman injection (rarely used directly)
  bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                    const Mat3& R, double chi2_gate=-1.0);

private:
  void nn_prune_();

private:
  static Quat canonicalizeQuat(const Quat& q_in);

  Config_v12 cfg_;
  State9_v12 x_;
  Mat99 P_{Mat99::Identity()};

  Mat3 R_nb_{Mat3::Identity()};   // BODY -> END
  Vec3 g_n_{0,0,9.81};            // +Down

  // Heave spring–damper (optional)
  double k_z_{0.0}, c_z_{0.0}, z0_{0.0};

  // Last gyro for convenience outputs
  Vec3 last_omega_b_meas_{Vec3::Zero()};

  // NN streaming
  NN_v12* nn_{nullptr};
  int nn_seq_len_{1};
  int nn_stride_{1};
  int nn_count_{0};
  std::deque<Eigen::Matrix<double,10,1>> nn_buf_;
  bool nn_warmed_{false};
};

} // namespace nnqekf_v12

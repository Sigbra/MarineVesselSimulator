#pragma once
// nn_ekf_v12.hpp — 9-state EKF + stateful TorchScript velocity NN (v12, Euler-only wrapped angles)

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
  double sigma_a        = 5e-3;   // accel white noise [m/s^2]
  double sigma_ba_rw    = 3e-6;   // accel bias random walk [m/s^2/s]
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
  //
  // norm_json: JSON/YAML with keys x_mean, x_std, y_mean, y_std.
  //  - v12 setup (Euler-only, wrapped): x_mean/x_std length MUST be 9, with ordering:
  //      [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n]
  //    where phi/theta/psi are wrapped to [-pi, +pi) (±180°) in the dataset and at runtime.
  //  - y_mean/y_std must be length 3 (END velocity: vE,vN,vD)
  bool init(const std::string& model_dir, const std::string& norm_json, bool use_cuda);

  // Reset all members' hidden states to zeros via init_state(1)
  void reset_states();

  // Window is kept for API compatibility (EKF buffers 10-wide rows).
  //
  // The EKF buffers a canonical 10-vector per step:
  //   [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n, 0]
  //
  // Internally, the NN uses ONLY Euler (D=9):
  //   [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n]
  // with angles wrapped to [-pi, +pi).
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
  // Note: v12 NN input does not use quaternion; this is still used by the EKF process model.
  void setRotationFromQuat(const Quat& q_nb);

  // State & covariance
  const State9_v12& state() const;
  const Mat99&      cov()   const;
  void setState(const State9_v12& x, const Mat99& P);

  void zeroAccelBias() { x_.b_a.setZero(); }

  // Optional: set heave equilibrium (Down) for spring–damper term
  void setHeaveEquilibrium(double z0);

  // Hook up NN and choose streaming cadence
  // seq_len: warm-start history length for GRU hidden state
  // stride:  apply EKF update every N samples (still steps NN every tick after warm-start)
  void setNN(NN_v12* nn, int seq_len, int stride);

  // Feed one NN sample:
  //  - accel_b: BODY specific force (or accel measurement, consistent with training)
  //  - q_nb:    BODY->END quaternion (kept for API compatibility; NN features derived from R_nb_)
  //  - tau_*:   BODY forces/torques used as NN inputs
  //
  // v12 NN features are Euler-based:
  //  - Euler angles (phi,theta,psi) are derived from the EKF attitude R_nb_ (END convention)
  //  - phi/theta/psi are wrapped to [-pi, +pi) before buffering
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

  // NN
  NN_v12* nn_{nullptr};
  int nn_seq_len_{1};
  int nn_stride_{1};
  int nn_count_{0};

  // Buffered canonical row (10):
  //   [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n, 0]
  // where angles are wrapped to [-pi, +pi).
  std::deque<Eigen::Matrix<double,10,1>> nn_buf_;

  bool nn_warmed_ = false;   // becomes true after we warm-start the GRU once
};

} // namespace nnqekf_v12
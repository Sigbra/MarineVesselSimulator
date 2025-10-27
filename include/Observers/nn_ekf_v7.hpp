#ifndef NN_EKF_V7_HPP
#define NN_EKF_V7_HPP

#pragma once
// NN_qObs_Aided_EKF — 9-state EKF aided by external quaternion observer (q-Obs)
// and a velocity-only neural network (NN_v7) that outputs END-frame velocity.
//
// Frames: BODY (x fwd, y starboard, z down), NAV/END (E, N, D; Down positive)
// State:  x = [ p^n (3); v^n (3); b^b_acc (3) ] — attitude is provided externally
//
// Measurements supported:
//  - GNSS antenna positions (lever arms)            -> updateGnssPos
//  - GNSS velocities (e.g., Doppler) at antenna     -> updateGnssVel
//  - NN END velocity (ensemble mean + covariance)   -> feedNN + updateNNVelNav (internal)

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <memory>
#include <vector>

namespace nnqekf {

using Vec3  = Eigen::Vector3d;
using Mat3  = Eigen::Matrix3d;
using Mat96 = Eigen::Matrix<double,9,6>;
using Mat99 = Eigen::Matrix<double,9,9>;

struct Config {
  double g = 9.81;            // gravity magnitude (Down positive in NAV)
  double sigma_a = 0.05;      // accel white noise [m/s^2]
  double sigma_ba_rw = 1e-3;  // accel bias random walk [m/s^2/sqrt(s)]
  double tau_ba = 0.0;        // accel bias leak time constant [s]; 0 => pure RW
  double chi2_gate_pos3 = -1.0; // NIS gate for 3D position; negative disables
  double chi2_gate_vec3 = -1.0; // NIS gate for 3D vectors (vel/body-vel); negative disables
};

struct State9 {
  Vec3 p = Vec3::Zero();   // position (E,N,D)
  Vec3 v = Vec3::Zero();   // velocity (E,N,D)
  Vec3 b_a = Vec3::Zero(); // accel bias (BODY)
};

// ------------------------------ NN_v7 ------------------------------
// Velocity-only NN wrapper (PIMPL). Torch/YAML are kept in the .cpp.
// Inputs per timestep:  [ax, ay, az, qw, qx, qy, qz]
// Output (END frame):   [vE, vN, vD]  + ensemble covariance Rv
class NN_v7 {
public:
  NN_v7();
  ~NN_v7();

  // model_dir: folder with member_XX.pt (TorchScript). Fallback: any *.pt excluding ensemble*.
  // norm_json: JSON/YAML with x_mean/x_std (7) and y_mean/y_std (3) used by the training script.
  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda);

  // window: last T rows of [ax, ay, az, qw, qx, qy, qz]
  // v_nav_mean: ensemble mean of [vE, vN, vD]
  // Rv_nav:     3x3 sample covariance from ensemble spread (with a small floor)
  bool infer(const std::vector<Eigen::Matrix<double,7,1>>& window,
             Vec3& v_nav_mean,
             Mat3& Rv_nav);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// -------------------------- NN_qObs_Aided_EKF --------------------------
class NN_qObs_Aided_EKF {
public:
  explicit NN_qObs_Aided_EKF(const Config& cfg = Config());

  // External attitude interface (Rotation/quat observer supplies this each step)
  void setRotationNavFromBody(const Mat3& R_nb);
  void setRotationFromQuat(const Eigen::Quaterniond& q_nb); // BODY->NAV quaternion in END convention

  // ---------- NN integration (END velocity aiding) ----------
  // Provide NN handle + sequence length and stride for inference triggering.
  void setNN(NN_v7* nn, int seq_len, int stride);

  // Feed one timestep: accelerometer (BODY) and BODY->NAV quaternion.
  // When the buffer holds seq_len samples, an NN inference is triggered every 'stride'.
  // The resulting [vE,vN,vD] and covariance are fused with updateNNVelNav().
  void feedNN(const Vec3& accel_b, const Eigen::Quaterniond& q_nb);

  // Directly fuse an END velocity measurement (e.g., from your own source).
  bool updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w = 1.0);

  // GNSS position update for a given antenna with lever arm r_body (BODY)
  bool updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w = 1.0);

  // GNSS velocity (e.g., Doppler) for antenna point with lever arm and body rates
  bool updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                     const Vec3& omega_b_meas, double w = 1.0);

  // NN CoM body velocity (u,v,w) update — if you ever have such a measurement (not used by NN_v7)
  bool updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w = 1.0);

  // Accessors
  const State9& state() const;
  const Mat99&  cov()   const;
  void setState(const State9& x, const Mat99& P);

  // Propagation with IMU (gyro cached for optional GNSS velocity lever-arm term)
  void propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt);

  // Optional convenience: legacy 12-state vector
  // [u v w p q r x y z phi theta psi]^T  (BODY rates are bias-corrected via provided b_gyro_hat)
  Eigen::Matrix<double,9,1> getState9() const;
  Eigen::VectorXd getState12(const Vec3& b_gyro_hat = Vec3::Zero()) const;

private:
  bool kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                    const Mat3& R, double chi2_gate);

  // Helper for NN buffer size
  void nn_prune_();

private:
  Config cfg_{};
  State9 x_{};
  Mat99  P_{};

  Mat3 R_nb_{};      // BODY->NAV rotation (from q-Obs/Rotation)
  Vec3 g_n_{};       // gravity [0,0,g] in NAV (Down +)
  Vec3 last_omega_b_meas_ = Vec3::Zero();

  // NN integration
  NN_v7* nn_ = nullptr;
  int nn_seq_len_ = 0;
  int nn_stride_  = 1;
  int nn_count_   = 0;
  std::deque<Eigen::Matrix<double,7,1>> nn_buf_;
};

} // namespace nnqekf

#endif // NN_EKF_V7_HPP

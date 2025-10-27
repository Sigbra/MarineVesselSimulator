#ifndef NN_EKF_V6_HPP
#define NN_EKF_V6_HPP

#pragma once
#include <deque>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace mekf {

// --- handy aliases ---
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
template<int R,int C> using Mat = Eigen::Matrix<double,R,C>;
template<int N>      using Vec = Eigen::Matrix<double,N,1>;

// ======================= State & Config =======================

struct State {
  Vec3 p   = Vec3::Zero();                  // position in END (E,N,D)
  Vec3 v   = Vec3::Zero();                  // velocity in END (E,N,D)
  Eigen::Quaterniond q_nb = Eigen::Quaterniond::Identity(); // BODY->NAV(END)
  Vec3 b_g = Vec3::Zero();                  // gyro bias (BODY)
  Vec3 b_a = Vec3::Zero();                  // accel bias (BODY)
};

struct Config {
  // Gravity (Down positive)
  double g = 9.81;

  // IMU axis sign correction (BODY). Use (+1,+1,+1) if IMU matches the simulator.
  Vec3 sign_acc  = Vec3(1,1,1);
  Vec3 sign_gyro = Vec3(1,1,1);

  // Continuous-time noise (std dev)
  double sigma_g     = 1e-3;  // rad/sqrt(s) gyro white noise
  double sigma_a     = 5e-2;  // m/s^1.5 accel white noise
  double sigma_bg_rw = 1e-5;  // rad/s^1.5 gyro bias RW
  double sigma_ba_rw = 1e-4;  // m/s^2/sqrt(s) accel bias RW

  // Optional first-order bias decay (time constants, seconds). <=0 disables.
  double tau_bg = -1.0;
  double tau_ba = -1.0;

  // NN windowing
  int nn_seq_len = 200;
  int nn_stride  = 1;

  // Gating (set <0 to disable)
  double chi2_gate_pos3 = -1.0;
  double chi2_gate_vec3 = -1.0;

  // Bias management
  bool estimate_bias = true;
  bool clamp_bias    = false;
  Vec3 bias_limit_gyro = Vec3(0.01, 0.01, 0.01);
  Vec3 bias_limit_acc  = Vec3(0.05, 0.05, 0.05);

  double gnss_dt_hint = 0.0;          // [s] if >0, use as GNSS sample period to build pseudo Doppler
  double gnss_vel_deriv_q = 0.05;     // [m^2/s^2] floor added to Rvel from finite-difference
  bool   use_gnss_pseudo_velocity = true; // enable vel update from position differencing
};

// ======================= Torch-free NN Interface (PIMPL) =======================

class QuatVelAttNN {
public:
  QuatVelAttNN();
  ~QuatVelAttNN();
  QuatVelAttNN(QuatVelAttNN&&) noexcept;
  QuatVelAttNN& operator=(QuatVelAttNN&&) noexcept;

  // Load models + normalization (TorchScript files; directory or single file)
  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda);

  // Run inference on a window of [p,q,r,ax,ay,az,cosψ,sinψ] (T=seq_len)
  // Returns v_mean (END), q_mean (BODY->END), and covariances Rv, Rq
  bool infer(const std::vector<Eigen::Matrix<double,8,1>>& window,
             Vec3& v_mean, Eigen::Quaterniond& q_mean,
             Mat3& Rv, Mat3& Rq);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ============================== MEKF ==============================

class MEKF {
public:
  using NNInfer = std::function<bool(const std::vector<Eigen::Matrix<double,8,1>>&,
                                     Vec3&, Eigen::Quaterniond&, Mat3&, Mat3&)>;

  explicit MEKF(const Config& cfg);

  // Access
  const State& state() const;
  const Mat<15,15>& covariance() const;

  // Setters
  void setState(const State& x, const Mat<15,15>& P);
  void enableBiasEstimation(bool on) { cfg_.estimate_bias = on; }
  void clampBias(bool on)            { cfg_.clamp_bias = on; }
  void setBiasLimits(const Vec3& gyro_lim, const Vec3& acc_lim) {
    cfg_.bias_limit_gyro = gyro_lim.cwiseAbs();
    cfg_.bias_limit_acc  = acc_lim.cwiseAbs();
  }

  // NN hookup (either provide a callback or wire the QuatVelAttNN)
  void setNNInfer(NNInfer cb, int seq_len=200, int stride=1);
  void setNN(QuatVelAttNN* nn, int seq_len=200, int stride=1);
  void clearNNInfer();

  // Time update with raw IMU (BODY): omega_meas=[p,q,r], acc_meas=specific force
  void propagate(const Vec3& omega_meas_body,
                 const Vec3& acc_meas_body,
                 double dt);

  // GNSS updates
  // Position of an antenna measured in NAV (END), with known lever arm in BODY.
  bool updateGnssPos(const Vec3& z_nav, const Mat3& Rpos, const Vec3& r_body, double w=1.0);
  bool updateGnssVel(const Vec3& z_v_nav,const Mat3& Rvel,const Vec3& r_body, const Vec3& omega_meas_body, double w);
  bool updateGnss(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w=1.0);


  // Heading/attitude-only correction from antenna baseline (z_nav in END, b_body in BODY).
  bool updateGnssBaseline(const Vec3& z_nav, const Mat3& Rz, const Vec3& b_body, double w=1.0);

  // NN updates (END velocity and BODY->END attitude)
  bool updateNNVelocity(const Vec3& z_v_n, const Mat3& Rv, double gate_chi2=-1.0);
  bool updateNNAttitude(const Eigen::Quaterniond& q_nn, const Mat3& Rq, double gate_chi2=-1.0);

  // Legacy export: [u v w p q r x y z phi theta psi]^T
  Eigen::Matrix<double,12,1> getState12() const;

  // Attitude accessors
  Eigen::Quaterniond attitudeQuat() const;        // q_nb (BODY->END)
  Eigen::Matrix3d    rotationNavFromBody() const; // R_nb (BODY->END)

  // Dataset Euler → quaternion (uses our END convention)
  static Eigen::Quaterniond quatFromEulerDataset(double phi, double th, double psi);

  // Heading convention: 0 = North, +CW (radians)
  static double yawFromQuatEND(const Eigen::Quaterniond& q);

  // --- “Our” END DCM builders/parsers (never use Eigen's default for this) ---
  static Mat3  RnbFromEuler(double phi, double th, double psi); // ZYX, rows = [E;N;D]
  static Mat3  RnbFromQuatCustom(const Eigen::Quaterniond& q);  // BODY->END using our convention
  static double yawFromRnb(const Mat3& Rnb);                    // atan2(R(0,0), R(1,0))

private:
  // Small helpers
  static Mat3 skew(const Vec3& a);
  static Eigen::Quaterniond quatExp(const Vec3& phi);
  static Vec3 quatLogVec(const Eigen::Quaterniond& q);
  static Mat3 leftJacobianSO3(const Vec3& phi);

  // NN window helpers
  void resetNNBuffer();
  bool nnWindowReady() const;
  void pushNNSample(const Vec3& imu_omega_meas, const Vec3& imu_acc_meas);
  void buildNNWindow(std::vector<Eigen::Matrix<double,8,1>>& out) const;
  bool maybeRunNNAndFuse();

  // KF internals
  void kalmanUpdate(const Vec3& r, const Mat<3,15>& H, const Mat3& R);
  bool gate(const Vec3& r, const Mat<3,15>& H, const Mat3& R, double chi2_thresh) const;
  static void josephUpdate(Mat<15,15>& P, const Mat<15,3>& K, const Mat<3,15>& H, const Mat3& R);
  static Mat<15,15> resetGamma(const Vec3& dtheta);

private:
  Config cfg_;
  State  x_;
  Mat<15,15> P_;

  // last raw IMU (for reporting p,q,r in getState12 and NN features)
  Vec3 last_imu_omega_meas_ = Vec3::Zero();
  Vec3 last_imu_acc_meas_   = Vec3::Zero();

  // NN windowing
  std::deque<Eigen::Matrix<double,8,1>> nn_buf_;
  int nn_stride_counter_ = 0;
  NNInfer nn_infer_ = nullptr;

  //Gnss update
  bool has_prev_gnss_pos_ = false;
  Vec3 prev_gnss_pos_nav_ = Vec3::Zero();
  Mat3 prev_Rpos_ = Mat3::Zero();
};

} // namespace mekf

#endif
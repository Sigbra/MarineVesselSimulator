// nn_ekf_v10.cpp — 9-state EKF (“NN_qObs_Aided_EKF_v10”) with external attitude (q-Obs)
// + Velocity-only NN_v10 TorchScript wrapper (PIMPL) for END-frame velocity aiding.
// v10 adds hydrostatic heave spring–damper to the process model,
// quaternion sign-continuity for NN input, yaw-rate-based covariance inflation
// for NN velocity updates, and optional IMU lever-arm compensation.

#include "Observers/nn_ekf_v10.hpp"
#include "Utilities/calculations.hpp"   // RnbFromQuatCustom(...)

#include <cmath>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cstring>

// TorchScript + YAML (kept out of header via PIMPL)
#include <torch/script.h>
#include <yaml-cpp/yaml.h>

namespace nnqekf_v10 {

// ----------------------- small helpers -----------------------
static inline double rad2deg(double r){ return r * 180.0 / M_PI; }

// END-convention yaw: ψ = atan2(R(0,0), R(1,0))
static inline double yaw_from_Rnb_END(const Mat3& R_nb){
  const double r10 = R_nb(1,0);
  const double r00 = R_nb(0,0);
  return std::atan2(r00, r10);
}

// Enforce quaternion sign continuity for NN inputs
static inline Eigen::Quaterniond enforceQuatContinuity(const Eigen::Quaterniond& q_in){
  static Eigen::Quaterniond q_last(1,0,0,0);
  Eigen::Quaterniond q = q_in.normalized();
  if (q.w()*q_last.w() + q.x()*q_last.x() + q.y()*q_last.y() + q.z()*q_last.z() < 0.0){
    q.coeffs() *= -1.0;
  }
  q_last = q;
  return q;
}

// ============================================================
//                       NN_v10 (PIMPL)
// ============================================================
struct NN_v10::Impl {
  // device
  bool use_cuda_ = true;
  torch::Device device_{torch::kCPU};

  // ensemble
  std::vector<torch::jit::script::Module> members_;

  // norms
  static constexpr int IN_DIM = 10; // [ax,ay,az,qw,qx,qy,qz, tau_x, tau_y, tau_n]
  static constexpr int V_DIM  = 3;  // [vE,vN,vD]

  std::array<double,IN_DIM> x_mean_{}, x_std_{};
  std::array<double,V_DIM>  y_mean_{}, y_std_{};
  torch::Tensor x_mean_t_, x_std_t_, y_mean_t_, y_std_t_;

  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda)
  {
#ifdef TORCH_CUDA_AVAILABLE
    use_cuda_ = use_cuda && torch::cuda::is_available();
#else
    use_cuda_ = false;
#endif
    device_ = use_cuda_ ? torch::kCUDA : torch::kCPU;

    // ---- load norms ----
    YAML::Node ns;
    {
      namespace fs = std::filesystem;
      fs::path p(norm_json);
#ifdef MVS_PROJECT_ROOT
      if (!p.is_absolute()) {
        fs::path alt = fs::path(MVS_PROJECT_ROOT) / p;
        if (fs::exists(alt)) p = alt;
      }
#endif
      try {
        ns = YAML::LoadFile(p.string());
      } catch (const YAML::BadFile&) {
        std::cerr << "[NN_v10] Failed to open norm_stats: " << norm_json << "\n";
        return false;
      }
    }
    auto x_mean = ns["x_mean"]; auto x_std = ns["x_std"];
    auto y_mean = ns["y_mean"]; auto y_std = ns["y_std"];
    if (!x_mean || !x_std || !y_mean || !y_std) {
      std::cerr << "[NN_v10] norm_json missing keys x_mean/x_std/y_mean/y_std\n";
      return false;
    }
    for (int i=0;i<IN_DIM;++i){ x_mean_[i]=x_mean[i].as<double>(); x_std_[i]=x_std[i].as<double>(); }
    for (int i=0;i<V_DIM;++i){  y_mean_[i]=y_mean[i].as<double>(); y_std_[i]=y_std[i].as<double>(); }

    x_mean_t_ = torch::from_blob(x_mean_.data(), {IN_DIM}, torch::kDouble).clone().to(device_);
    x_std_t_  = torch::from_blob(x_std_.data(),  {IN_DIM}, torch::kDouble).clone().to(device_);
    y_mean_t_ = torch::from_blob(y_mean_.data(), {V_DIM},  torch::kDouble).clone().to(device_);
    y_std_t_  = torch::from_blob(y_std_.data(),  {V_DIM},  torch::kDouble).clone().to(device_);

    // ---- collect TorchScript members ----
    namespace fs = std::filesystem;
    fs::path md(model_dir);
#ifdef MVS_PROJECT_ROOT
    if (!md.is_absolute()) {
      fs::path alt = fs::path(MVS_PROJECT_ROOT) / md;
      if (fs::exists(alt)) md = alt;
    }
#endif
    if (!fs::exists(md)) {
      std::cerr << "[NN_v10] Model path does not exist: " << md << "\n";
      return false;
    }

    auto is_pt = [](const fs::path& p){
      std::string ext = p.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      return (ext == ".pt" || ext == ".ts");
    };

    std::vector<std::string> files;
    if (fs::is_directory(md)) {
      // Prefer member_XX.pt
      for (auto& e : fs::directory_iterator(md)) {
        if (!e.is_regular_file()) continue;
        const auto& p = e.path();
        if (!is_pt(p)) continue;
        const auto name = p.filename().string();
        if (name.rfind("member_", 0) == 0) files.push_back(p.string());
      }
      // Fallback: any *.pt except ensemble*
      if (files.empty()) {
        for (auto& e : fs::directory_iterator(md)) {
          if (!e.is_regular_file()) continue;
          const auto& p = e.path();
          if (!is_pt(p)) continue;
          const auto name = p.filename().string();
          if (name.find("ensemble") != std::string::npos) continue;
          files.push_back(p.string());
        }
      }
    } else if (fs::is_regular_file(md) && is_pt(md)) {
      files.push_back(md.string());
    }

    if (files.empty()) {
      std::cerr << "[NN_v10] No TorchScript files found at: " << md << "\n";
      return false;
    }
    std::sort(files.begin(), files.end());

    try {
      members_.clear();
      members_.reserve(files.size());
      for (const auto& f : files) {
        auto m = torch::jit::load(f, device_);
        m.eval();
        members_.push_back(std::move(m));
        std::cerr << "[NN_v10] Loaded " << f << (use_cuda_ ? " (CUDA)\n" : " (CPU)\n");
      }
    } catch (const c10::Error& e) {
      std::cerr << "[NN_v10] Failed to load model: " << e.what() << "\n";
      return false;
    }

    if (members_.size() == 1) {
      std::cerr << "[NN_v10] WARNING: only 1 model loaded; covariance will be near floor.\n";
    } else {
      std::cerr << "[NN_v10] Ensemble size: " << members_.size() << "\n";
    }
    return true;
  }

  bool infer(const std::vector<Eigen::Matrix<double,IN_DIM,1>>& window,
             Vec3& v_nav_mean,
             Mat3& Rv_nav)
  {
    if (members_.empty() || window.empty()) return false;
    const int T = (int)window.size();

    torch::NoGradGuard ng;

    // Input [1, T, IN_DIM] (double for norms; cast to float for model)
    torch::Tensor x = torch::empty({1, T, IN_DIM},
        torch::TensorOptions().dtype(torch::kDouble).device(device_));
    {
      auto xa = x.accessor<double,3>();
      for (int t=0; t<T; ++t) {
        for (int j=0; j<IN_DIM; ++j) xa[0][t][j] = window[t](j);
      }
    }
    x = (x - x_mean_t_.view({1,1,IN_DIM})) / x_std_t_.view({1,1,IN_DIM});

    const int M = (int)members_.size();
    std::vector<Vec3> preds; preds.reserve(M);

    for (int m=0; m<M; ++m) {
      auto out_any = members_[m].forward({ x.to(torch::kFloat) });
      torch::Tensor y = out_any.toTensor();     // [1,T,3] = [vE,vN,vD]
      if (y.dim()!=3 || y.size(0)!=1 || y.size(2)!=3) {
        std::cerr << "[NN_v10] Bad output shape from member " << m << ": " << y.sizes() << "\n";
        return false;
      }
      torch::Tensor last = y.index({0, T-1});   // [3]
      auto ycpu = last.to(torch::kDouble).to(torch::kCPU);
      double vv[3]; std::memcpy(vv, ycpu.data_ptr<double>(), 3*sizeof(double));

      Vec3 v;
      for (int k=0;k<3;++k) v(k) = vv[k]*y_std_[k] + y_mean_[k]; // de-normalize
      preds.push_back(v);
    }

    // mean
    v_nav_mean.setZero();
    for (const auto& v : preds) v_nav_mean += v;
    v_nav_mean /= std::max(1, M);

    // sample covariance + small floor
    Eigen::Matrix<double,3,Eigen::Dynamic> D(3, M);
    for (int i=0;i<M;++i) D.col(i) = preds[i] - v_nav_mean;
    Rv_nav = (D * D.transpose()) / std::max(1, M-1);
    Rv_nav += 1e-6 * Eigen::Matrix3d::Identity();

    return true;
  }
};

NN_v10::NN_v10()  : impl_(std::make_unique<Impl>()) {}
NN_v10::~NN_v10() = default;

bool NN_v10::init(const std::string& model_dir,
                 const std::string& norm_json,
                 bool use_cuda) {
  return impl_->init(model_dir, norm_json, use_cuda);
}

bool NN_v10::infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
                  Vec3& v_nav_mean,
                  Mat3& Rv_nav) {
  return impl_->infer(window, v_nav_mean, Rv_nav);
}

// ============================================================
//                       EKF v10
// ============================================================

NN_qObs_Aided_EKF_v10::NN_qObs_Aided_EKF_v10(const Config_v10& cfg) : cfg_(cfg) {
  P_.setIdentity();
  P_ *= 1e-2;                 // mild initial uncertainty
  R_nb_.setIdentity();        // BODY->END
  g_n_ = Vec3(0,0,cfg_.g);    // Down positive

  // -------- Heave spring–damper constants from RAN (option A) --------
  // Geometry & fluid (RAN defaults)
  const double rho        = 1025.0;   // seawater density
  const double L          = 5.46;     // hull length
  const double Beam_pont  = 0.50;     // single pontoon beam
  const double Cw_pont    = 0.80;     // waterplane coefficient
  const double m          = 850.0;    // mass (kg) — RAN nominal
  const double zeta       = 0.30;     // damping ratio (for heave)

  // Waterplane area of two pontoons (each: Cw*L*Beam_pont)
  const double Aw_pont = Cw_pont * L * Beam_pont;
  const double K_h     = rho * cfg_.g * (2.0 * Aw_pont);          // N/m
  const double C_h     = 2.0 * zeta * std::sqrt(m * K_h);         // N·s/m

  // Per-mass coefficients used in accel (a_D += -k_z*(z-z0) - c_z*v_D)
  k_z_  = K_h / m;            // [1/s^2]
  c_z_  = C_h / m;            // [1/s]
  z0_   = 0.0;                // equilibrium Down (can be overridden)
}

void NN_qObs_Aided_EKF_v10::setHeaveEquilibrium(double z0){ z0_ = z0; }

void NN_qObs_Aided_EKF_v10::setRotationNavFromBody(const Mat3& R_nb) { R_nb_ = R_nb; }

void NN_qObs_Aided_EKF_v10::setRotationFromQuat(const Eigen::Quaterniond& q_in)
{
    // 1) Normalize
    Eigen::Quaterniond q = q_in.normalized();

    // 2) If your source provides NAV->BODY (common), convert to BODY->NAV
    if (assume_q_is_nav_to_body_) {
        q = q.conjugate();
    }

    // 3) Enforce hemisphere continuity (avoid q <-> -q jumps at ~180°)
    if (q_last_valid_ && q.dot(q_last_) < 0.0) {
        q.coeffs() *= -1.0;  // flip all 4 components
    }
    q_last_ = q;
    q_last_valid_ = true;

    // 4) Build R_nb_ using your custom END convention (Body -> END)
    R_nb_ = RnbFromQuatCustom(q);
}

void NN_qObs_Aided_EKF_v10::setNN(NN_v10* nn, int seq_len, int stride) {
  nn_ = nn;
  nn_seq_len_ = std::max(1, seq_len);
  nn_stride_  = std::max(1, stride);
  nn_count_   = 0;
  nn_buf_.clear();
}

void NN_qObs_Aided_EKF_v10::nn_prune_() {
  const int MAX_KEEP = std::max(nn_seq_len_, 4);
  while ((int)nn_buf_.size() > MAX_KEEP) nn_buf_.pop_front();
}

// Feed one sample to NN buffer; accel and tau are scalars, quaternion is normalized + sign-continuous.
void NN_qObs_Aided_EKF_v10::feedNN(const Vec3& accel_b,
                                   const Eigen::Quaterniond& q_nb,
                                   double tau_x,
                                   double tau_y,
                                   double tau_n)
{
  if (!nn_ || nn_seq_len_ <= 0) return;

  // 1) Enforce quaternion sign continuity (NN sees raw quaternion components)
  const Eigen::Quaterniond qn = enforceQuatContinuity(q_nb);

  // 2) Buffer the latest input row [ax, ay, az, qw, qx, qy, qz, tau_x, tau_y, tau_n]
  Eigen::Matrix<double,10,1> row;
  row << accel_b.x(), accel_b.y(), accel_b.z(),
         qn.w(), qn.x(), qn.y(), qn.z(),
         tau_x, tau_y, tau_n;

  nn_buf_.push_back(row);
  nn_prune_();

  // 3) Only infer when we have a full window and hit the stride
  nn_count_++;
  if ((int)nn_buf_.size() < nn_seq_len_) return;
  if ((nn_count_ % nn_stride_) != 0)     return;

  // Last nn_seq_len_ samples (time order preserved)
  std::vector<Eigen::Matrix<double,10,1>> window;
  window.reserve(nn_seq_len_);
  auto it = nn_buf_.end();
  for (int i = 0; i < nn_seq_len_; ++i) { --it; }
  for (int i = 0; i < nn_seq_len_; ++i, ++it) window.push_back(*it);

  // 4) NN inference → mean END velocity + covariance
  Vec3 v_nav_mean;   // [vE, vN, vD]
  Mat3 Rv_nav;
  if (!nn_->infer(window, v_nav_mean, Rv_nav)) return;

  // 5) Fuse directly in END (no E/N swap)
  (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1.0);
}

bool NN_qObs_Aided_EKF_v10::updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w)
{
  // residual: z - h(x) with h(x)=v^n
  const Vec3 res = z_v_nav - x_.v;

  // H = [0_{3x3}  I_{3x3}  0_{3x3}]
  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3).setIdentity();

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rv / wt;
  const Mat3 S = (H * P_ * H.transpose()) + R;

  // Optional NIS gate
  if (cfg_.chi2_gate_vec3 > 0.0) {
    const double nis = res.transpose() * S.inverse() * res;
    if (nis >= cfg_.chi2_gate_vec3) return false;
  }

  // Full Kalman gain (9x3)
  const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

  // ---------------- Masked state update ----------------
  // Build J that updates only velocity rows (3..5). Position (0..2) and bias (6..8) frozen.
  Eigen::Matrix<double,9,9> J = Eigen::Matrix<double,9,9>::Zero();
  J.block<3,3>(3,3).setIdentity();   // update v only

  // Apply masked increment
  const Eigen::Matrix<double,9,1> Jdx = J * dx;
  // x_.p unchanged
  x_.v   += Jdx.segment<3>(3);
  // x_.b_a unchanged

  // ---------------- Consistent masked covariance update (Joseph form) ----------------
  const Eigen::Matrix<double,9,9> I = Eigen::Matrix<double,9,9>::Identity();
  const Eigen::Matrix<double,9,9> IKH = I - (J * K * H);
  P_ = IKH * P_ * IKH.transpose() + (J * K) * R * (J * K).transpose();

  return true;
}

// bool NN_qObs_Aided_EKF_v10::updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w) {
//   // residual: z - h(x) with h(x)=v^n
//   const Vec3 res = z_v_nav - x_.v;

//   // H = [0_{3x3}  I_{3x3}  0_{3x3}]
//   Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
//   H.block<3,3>(0,3).setIdentity();

//   // Base scaling from 'w'
//   const double wt = (std::isfinite(w) && w > 0.0) ? w : 1.0;
//   Mat3 R_eff = Rv / wt;

//   // --- NEW: inflate covariance during aggressive yaw turns ---
//   // Threshold ~0.4 rad/s (~23 deg/s). Cap inflation at 5x.
//   const double r = last_omega_b_meas_.z();
//   const double r_thresh = 0.4;
//   const double inflate = 1.0 + std::min(4.0, std::abs(r) / r_thresh);
//   R_eff *= inflate;

//   const Mat3 S = (H * P_ * H.transpose()) + R_eff;

//   // Optional NIS gate (unchanged)
//   if (cfg_.chi2_gate_vec3 > 0.0) {
//     const double nis = res.transpose() * S.inverse() * res;
//     if (nis >= cfg_.chi2_gate_vec3) return false;
//   }

//   const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
//   const Eigen::Matrix<double,9,1> dx = K * res;

//   // Inject nominal
//   x_.p   += dx.segment<3>(0);
//   x_.v   += dx.segment<3>(3);
//   x_.b_a += dx.segment<3>(6);

//   // Joseph stabilized covariance update
//   const Mat99 I = Mat99::Identity();
//   const Mat99 IKH = I - K * H;
//   P_ = IKH * P_ * IKH.transpose() + K * R_eff * K.transpose();

//   return true;
// }

const State9_v10& NN_qObs_Aided_EKF_v10::state() const { return x_; }
const Mat99&     NN_qObs_Aided_EKF_v10::cov()   const { return P_; }

void NN_qObs_Aided_EKF_v10::setState(const State9_v10& x, const Mat99& P){ x_ = x; P_ = P; }

void NN_qObs_Aided_EKF_v10::propagate(const Vec3& omega_b_meas,
                                      const Vec3& accel_b_meas,
                                      double dt)
{
  if (!(dt > 0.0)) return;                       // guard
  last_omega_b_meas_ = omega_b_meas;

  // ---------- Specific force in BODY ----------
#ifdef ENABLE_IMU_LEVER_ARM
  static Vec3 omega_prev = omega_b_meas;
  const Vec3 alpha_b = (omega_b_meas - omega_prev) / std::max(1e-6, dt);
  omega_prev = omega_b_meas;

  // IMU lever arm r_IMU - r_CG in BODY (m) — set your actual value:
  const Vec3 r_ib_b(0.0, 0.0, 0.0);

  // Remove rotational acceleration at IMU location: a_corr = a_meas - (α×r + ω×(ω×r))
  const Vec3 a_corr = accel_b_meas
                    - (alpha_b.cross(r_ib_b) + omega_b_meas.cross(omega_b_meas.cross(r_ib_b)));

  const Vec3 f_b = a_corr - x_.b_a;            // specific force minus bias
#else
  const Vec3 f_b = accel_b_meas - x_.b_a;      // if accel_b_meas is specific force (incl. gravity removed by sensor)
#endif

  // ---------- Acceleration in END (Down-positive gravity) ----------
  Vec3 a_n = R_nb_ * f_b + g_n_;               // g_n_ ~ [0,0,+9.81] in END

  // Optional heave spring–damper (Down axis)
  a_n.z() += -k_z_ * (x_.p.z() - z0_) - c_z_ * x_.v.z();

  // ---------- Nominal state propagation (constant-acceleration step) ----------
  // Exact for constant a over dt:
  x_.p += x_.v * dt + 0.5 * a_n * dt * dt;     // uses v_k
  x_.v += a_n * dt;                             // to v_{k+1}

  // ---------- Linearized error model: δx=[δp, δv, δb_a] (9x9) ----------
  Mat99 A = Mat99::Zero();
  // δṗ = δv
  A.block<3,3>(0,3).setIdentity();
  // δv̇ ≈ -R_nb_ δb_a (through f_b = a_meas_specific - b_a)
  A.block<3,3>(3,6) = -R_nb_;
  // Heave coupling (Down axis)
  A(5,2) += -k_z_;    // ∂(v̇_D)/∂z
  A(5,5) += -c_z_;    // ∂(v̇_D)/∂v_D
  // Bias first-order leak (optional)
  if (cfg_.tau_ba > 0.0) {
    A.block<3,3>(6,6) = -(1.0 / cfg_.tau_ba) * Mat3::Identity();
  }

  // ---------- Discretization ----------
  const Mat99 I9 = Mat99::Identity();

  // 2nd-order Ad (improves long-run numeric behavior)
  const Mat99 Ad = I9 + A * dt + 0.5 * (A * A) * (dt * dt);

  // Noise mapping (continuous): w = [n_a_body; n_ba_rw]
  // We’ll build a *discrete* Qd with the standard integrated blocks so P grows correctly.
  // Accel white noise in BODY:
  Mat3 Qa_body = (cfg_.sigma_a * cfg_.sigma_a) * Mat3::Identity();
  // Map into END this step:
  Mat3 S = R_nb_ * Qa_body * R_nb_.transpose();     // if Qa isotropic, S == σ_a^2 * I

  // Discrete process noise for [p; v] driven by accel white noise:
  const double dt2 = dt * dt, dt3 = dt2 * dt;
  Mat99 Qd = Mat99::Zero();
  // Position–position
  Qd.block<3,3>(0,0) = (dt3 / 3.0) * S;
  // Position–velocity and its transpose
  Mat3 Qpv = (dt2 / 2.0) * S;
  Qd.block<3,3>(0,3) = Qpv;
  Qd.block<3,3>(3,0) = Qpv.transpose();
  // Velocity–velocity
  Qd.block<3,3>(3,3) = dt * S;

  // Bias random walk (discrete): Q_ba = σ_ba_rw^2 * dt * I
  Qd.block<3,3>(6,6) += (cfg_.sigma_ba_rw * cfg_.sigma_ba_rw) * dt * Mat3::Identity();

  // If you keep the bias leak in A, the exact Qd for that subblock differs slightly;
  // this approximation is fine unless τ is very small vs dt.

  // ---------- Covariance time update ----------
  P_ = Ad * P_ * Ad.transpose() + Qd;
}

// void NN_qObs_Aided_EKF_v10::propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt) {
//   last_omega_b_meas_ = omega_b_meas; // cached for gating, etc.

//   // -------- Optional lever-arm compensation (BODY) --------
//   // Define ENABLE_IMU_LEVER_ARM (e.g., add -DENABLE_IMU_LEVER_ARM to your compile flags)
//   // and set r_ib_b below to your IMU position w.r.t. CG (BODY frame, meters).
// #ifdef ENABLE_IMU_LEVER_ARM
//   static Vec3 omega_prev = omega_b_meas;
//   const Vec3 alpha_b = (omega_b_meas - omega_prev) / std::max(1e-6, dt);
//   omega_prev = omega_b_meas;

//   // Set your lever arm here (BODY): r_IMU - r_CG
//   const Vec3 r_ib_b(0.0, 0.0, 0.0); // TODO: fill with your IMU offset [m]

//   // Remove rotational kinematics: a_corr = a_meas - (α×r + ω×(ω×r))
//   const Vec3 a_corr = accel_b_meas
//                     - (alpha_b.cross(r_ib_b) + omega_b_meas.cross(omega_b_meas.cross(r_ib_b)));

//   const Vec3 f_b = a_corr - x_.b_a;       // specific force minus bias
// #else
//   const Vec3 f_b = accel_b_meas - x_.b_a; // default (no lever arm)
// #endif

//   // Acceleration in END with Down-positive gravity
//   Vec3 a_n = R_nb_ * f_b + g_n_;

//   // -------- v10: add hydrostatic heave spring–damper to a_n (Down axis) --------
//   // a_D += -k_z_ * (z - z0_) - c_z_ * v_D
//   a_n.z() += -k_z_ * (x_.p.z() - z0_) - c_z_ * x_.v.z();

//   // Nominal state propagation
//   x_.p += x_.v * dt + 0.5 * a_n * dt * dt;
//   x_.v += a_n * dt;

//   // Linearized error propagation: δx=[δp, δv, δb_a]
//   Mat99 A = Mat99::Zero();
//   A.block<3,3>(0,3).setIdentity();            // δp_dot = δv
//   A.block<3,3>(3,6) = - R_nb_;                // δv_dot ≈ -R_nb δb_a

//   // v10 additions for heave dynamics (Down axis indices: p.z()→row 2, v.z()→row 5)
//   A(5,2) += -k_z_;    // ∂(v̇_D)/∂z = -k_z_
//   A(5,5) += -c_z_;    // ∂(v̇_D)/∂v_D = -c_z_

//   if (cfg_.tau_ba > 0.0) {
//     A.block<3,3>(6,6) = -(1.0/cfg_.tau_ba) * Mat3::Identity(); // optional bias leak
//   }

//   // Discretize (Euler)
//   const Mat99 Ad = Mat99::Identity() + A * dt;

//   // Process-noise mapping (continuous): w = [n_a (BODY); n_ba_rw]
//   Mat96 E = Mat96::Zero();
//   E.block<3,3>(3,0) = R_nb_;                  // accel white noise -> δv
//   E.block<3,3>(6,3) = Mat3::Identity();       // bias random walk -> δb_a

//   Eigen::Matrix<double,6,6> Qc = Eigen::Matrix<double,6,6>::Zero();
//   Qc.block<3,3>(0,0) = (cfg_.sigma_a     * cfg_.sigma_a)     * Mat3::Identity();
//   Qc.block<3,3>(3,3) = (cfg_.sigma_ba_rw * cfg_.sigma_ba_rw) * Mat3::Identity();

//   const Mat99 Qd = (E * Qc * E.transpose()) * dt;

//   // Covariance time update
//   P_ = Ad * P_ * Ad.transpose() + Qd;
// }



bool NN_qObs_Aided_EKF_v10::updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w){
  const Vec3 h  = x_.p + R_nb_ * r_body; // predicted antenna position
  const Vec3 res = z_nav_ant - h;        // residual

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,0) = Mat3::Identity();  // d/dp

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  return kalmanUpdate(res, H, Rpos / wt, cfg_.chi2_gate_pos3);
}

bool NN_qObs_Aided_EKF_v10::updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                                         const Vec3& omega_b_meas, double w){
  const Vec3 v_ant_pred = x_.v + R_nb_ * (omega_b_meas.cross(r_body));
  const Vec3 res = z_v_nav_ant - v_ant_pred;

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = Mat3::Identity(); // d/dv

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  return kalmanUpdate(res, H, Rvel / wt, cfg_.chi2_gate_vec3);
}

bool NN_qObs_Aided_EKF_v10::updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w){
  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b_pred = R_bn * x_.v;   // predicted body velocity at CG
  const Vec3 res = z_v_b - v_b_pred;   // residual in BODY

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = R_bn;            // d(R_bn v_n)/d v_n = R_bn

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  return kalmanUpdate(res, H, Rvb / wt, cfg_.chi2_gate_vec3);
}

Eigen::Matrix<double,9,1> NN_qObs_Aided_EKF_v10::getState9() const {
  Eigen::Matrix<double,9,1> x9;
  x9.segment<3>(0) = x_.p;
  x9.segment<3>(3) = x_.v;
  x9.segment<3>(6) = x_.b_a;
  return x9;
}

// Optional convenience: 12-state legacy vector [u v w p q r x y z phi theta psi]^T
Eigen::VectorXd NN_qObs_Aided_EKF_v10::getState12(const Vec3& b_gyro_hat) const {
  Eigen::Matrix<double,12,1> x12;

  // BODY velocity from NAV
  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b  = R_bn * x_.v; // [u v w]

  // BODY rates (bias-corrected if provided)
  const Vec3 omega_b = last_omega_b_meas_ - b_gyro_hat;

  // ZYX Euler from custom END R_nb (rows [E;N;D])
  auto clamp1 = [](double s){ return std::max(-1.0, std::min(1.0, s)); };
  const double r20 = R_nb_(2,0);
  const double r21 = R_nb_(2,1);
  const double r22 = R_nb_(2,2);
  const double r10 = R_nb_(1,0);
  const double r00 = R_nb_(0,0);

  const double theta = -std::asin(clamp1(r20));  // pitch about +E/NED-y
  const double phi   = std::atan2(r21, r22);     // roll  about +E/NED-x
  const double psi   = std::atan2(r00, r10);     // heading from North toward East

  x12 << v_b.x(), v_b.y(), v_b.z(),
          omega_b.x(), omega_b.y(), omega_b.z(),
          x_.p.x(), x_.p.y(), x_.p.z(),
          phi, theta, psi;
  return x12;
}

bool NN_qObs_Aided_EKF_v10::kalmanUpdate(const Vec3& r,
                                         const Eigen::Matrix<double,3,9>& H,
                                         const Mat3& R, double chi2_gate)
{
  // Innovation covariance + tiny jitter for numeric safety
  Mat3 S = (H * P_ * H.transpose()) + R;
  S(0,0) += 1e-12; S(1,1) += 1e-12; S(2,2) += 1e-12;

  // Gate with a robust solve (no explicit inverse)
  Eigen::LDLT<Mat3> ldltS(S);
  if (ldltS.info() != Eigen::Success) return false;
  const double nis = r.transpose() * ldltS.solve(r);
  if (chi2_gate > 0.0 && nis >= chi2_gate) return false;

  // K = P Hᵀ S⁻¹ via solve
  const Eigen::Matrix<double,9,3> PHt = P_ * H.transpose();
  const Eigen::Matrix<double,9,3> K   = ldltS.solve(PHt.transpose()).transpose();

  Eigen::Matrix<double,9,1> dx = K * r;

  // --- Safety: limit velocity correction from a position measurement ---
  // (Prevents a single crazy Δv. Tune these numbers to your dynamics.)
  Eigen::Vector3d dv = dx.segment<3>(3);
  const double dv_max = 0.5; // m/s per update (example)
  const double n = dv.norm();
  if (n > dv_max) {
    dv *= (dv_max / (n + 1e-12));
    dx.segment<3>(3) = dv;
  }

  // Inject
  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  // Joseph form keeps P PSD even with saturation
  const Mat99 I  = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
  return true;
}


// bool NN_qObs_Aided_EKF_v10::kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
//                                         const Mat3& R, double chi2_gate){
//   const Mat3 S = (H * P_ * H.transpose()) + R;

//   // Optional NIS gating
//   if (chi2_gate > 0.0) {
//     const double nis = r.transpose() * S.inverse() * r;
//     if (nis >= chi2_gate) return false;
//   }

//   const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
//   const Eigen::Matrix<double,9,1> dx = K * r;

//   // Inject nominal
//   x_.p   += dx.segment<3>(0);
//   x_.v   += dx.segment<3>(3);
//   x_.b_a += dx.segment<3>(6);

//   // Joseph stabilized covariance update
//   const Mat99 I = Mat99::Identity();
//   const Mat99 IKH = I - K * H;
//   P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

//   return true;
// }

} // namespace nnqekf_v10

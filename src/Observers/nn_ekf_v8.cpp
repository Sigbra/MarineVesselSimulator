// nn_ekf_v8.cpp — 9-state EKF (“NN_qObs_Aided_EKF_v8”) with external attitude (q-Obs)
// + Velocity-only NN_v8 TorchScript wrapper (PIMPL) providing prev BODY velocity to NN input.

#include "Observers/nn_ekf_v8.hpp"
#include "Utilities/calculations.hpp"   // RnbFromQuatCustom(...)

#include <cmath>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <deque>

// TorchScript + YAML (kept out of header via PIMPL)
#include <torch/script.h>
#include <yaml-cpp/yaml.h>

namespace nnqekf_v8 {

// ----------------------- small debug helpers -----------------------
static inline double rad2deg(double r){ return r * 180.0 / M_PI; }

static inline double yaw_from_Rnb_END(const Mat3& R_nb) {
  // END convention used in your codebase: psi = atan2(r00, r10)
  const double r10 = R_nb(1,0);
  const double r00 = R_nb(0,0);
  return std::atan2(r00, r10);
}

static void print_vec3(const char* name, const Vec3& v){
  std::cerr << name << " = [" << v.x() << ", " << v.y() << ", " << v.z() << "]\n";
}

// ============================== NN_v8 (PIMPL) ==============================
// NOTE: Only change from v7 wrapper is IN_DIM=10 and the caller now supplies
//       u_prev, v_prev, w_prev in the per-timestep input row.

struct NN_v8::Impl {
  // device
  bool use_cuda_ = false;
  torch::Device device_{torch::kCPU};

  // ensemble
  std::vector<torch::jit::script::Module> members_;

  // norms
  static constexpr int IN_DIM = 10; // [ax,ay,az,qw,qx,qy,qz, u_prev,v_prev,w_prev]
  static constexpr int V_DIM  = 3;  // network outputs 3 channels (kept as in v7 path)

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
        std::cerr << "[NN_v8] Failed to open norm_stats: " << norm_json << "\n";
        return false;
      }
    }
    auto x_mean = ns["x_mean"]; auto x_std = ns["x_std"];
    auto y_mean = ns["y_mean"]; auto y_std = ns["y_std"];
    if (!x_mean || !x_std || !y_mean || !y_std) {
      std::cerr << "[NN_v8] norm_json missing keys x_mean/x_std/y_mean/y_std\n";
      return false;
    }
    // Expect 10-dim inputs now
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
      std::cerr << "[NN_v8] Model path does not exist: " << md << "\n";
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
      std::cerr << "[NN_v8] No TorchScript files found at: " << md << "\n";
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
        std::cerr << "[NN_v8] Loaded " << f << (use_cuda_ ? " (CUDA)\n" : " (CPU)\n");
      }
    } catch (const c10::Error& e) {
      std::cerr << "[NN_v8] Failed to load model: " << e.what() << "\n";
      return false;
    }

    if (members_.size() == 1) {
      std::cerr << "[NN_v8] WARNING: only 1 model loaded; covariance will be near floor.\n";
    } else {
      std::cerr << "[NN_v8] Ensemble size: " << members_.size() << "\n";
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
    torch::Tensor x = torch::empty({1, T, IN_DIM}, torch::TensorOptions().dtype(torch::kDouble).device(device_));
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
      torch::Tensor y = out_any.toTensor();     // [1,T,3] (as in v7 path)
      if (y.dim()!=3 || y.size(0)!=1 || y.size(2)!=3) {
        std::cerr << "[NN_v8] Bad output shape from member " << m << ": " << y.sizes() << "\n";
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

    // ---------- covariance: single-model fallback, floor, inflation ----------
    Eigen::Matrix<double,3,Eigen::Dynamic> D(3, M);
    for (int i=0;i<M;++i) D.col(i) = preds[i] - v_nav_mean;

    if (M <= 1) {
      // Single model: fixed covariance (e.g., 0.4 m/s std on each axis)
      const double s_single = 0.4;
      Rv_nav = (s_single*s_single) * Eigen::Matrix3d::Identity();
    } else {
      // Sample covariance
      Rv_nav = (D * D.transpose()) / std::max(1, M-1);

      // Add floor to avoid overconfidence (e.g., 0.3 m/s std)
      const double s_floor = 0.3;
      Rv_nav += (s_floor*s_floor) * Eigen::Matrix3d::Identity();

      // Global inflation to further soften measurement authority
      const double alpha = 4.0;  // 4× covariance
      Rv_nav *= alpha;
    }

    return true;
  }
};

NN_v8::NN_v8()  : impl_(std::make_unique<Impl>()) {}
NN_v8::~NN_v8() = default;

bool NN_v8::init(const std::string& model_dir,
                 const std::string& norm_json,
                 bool use_cuda) {
  return impl_->init(model_dir, norm_json, use_cuda);
}

bool NN_v8::infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
                  Vec3& v_nav_mean,
                  Mat3& Rv_nav) {
  return impl_->infer(window, v_nav_mean, Rv_nav);
}

// ================================ EKF ================================

NN_qObs_Aided_EKF_v8::NN_qObs_Aided_EKF_v8(const Config_v8& cfg) : cfg_(cfg) {
  P_.setIdentity();
  P_ *= 1e-2;                 // mild initial uncertainty
  R_nb_.setIdentity();        // BODY->NAV
  g_n_ = Vec3(0,0,cfg_.g);    // Down positive
  nn_prev_v_b_.setZero();     // prev BODY velocity for NN input (u_prev,v_prev,w_prev)
}

void NN_qObs_Aided_EKF_v8::setRotationNavFromBody(const Mat3& R_nb) { R_nb_ = R_nb; }

void NN_qObs_Aided_EKF_v8::setRotationFromQuat(const Eigen::Quaterniond& q_nb) {
  R_nb_ = RnbFromQuatCustom(q_nb);  // custom END convention
}

void NN_qObs_Aided_EKF_v8::setNN(NN_v8* nn, int seq_len, int stride) {
  nn_ = nn;
  nn_seq_len_ = std::max(1, seq_len);
  nn_stride_  = std::max(1, stride);
  nn_count_   = 0;
  nn_buf_.clear();
}

void NN_qObs_Aided_EKF_v8::nn_prune_() {
  const int MAX_KEEP = std::max(nn_seq_len_, 4);
  while ((int)nn_buf_.size() > MAX_KEEP) nn_buf_.pop_front();
}

// Noth East Vel from NN swapped to match convention.
// Now we also feed (u_prev,v_prev,w_prev) into the NN input row.
void NN_qObs_Aided_EKF_v8::feedNN(const Vec3& accel_b, const Eigen::Quaterniond& q_nb)
{
  if (!nn_ || nn_seq_len_ <= 0) return;

  // 0) Compute BODY velocity estimate to use as "previous" for the NEXT call
  const Mat3 R_bn_now = R_nb_.transpose();
  const Vec3 v_b_now = R_bn_now * x_.v; // EKF-estimated BODY velocity at current instant

  // 1) Buffer the latest input row [ax, ay, az, qw, qx, qy, qz, u_prev, v_prev, w_prev]
  Eigen::Quaterniond qn = q_nb.normalized();
  Eigen::Matrix<double,10,1> row;
  row << accel_b.x(), accel_b.y(), accel_b.z(),
         qn.w(), qn.x(), qn.y(), qn.z(),
         nn_prev_v_b_.x(), nn_prev_v_b_.y(), nn_prev_v_b_.z();
  nn_buf_.push_back(row);
  nn_prune_();

  // 2) Update the cached prev BODY velocity for the *next* call
  nn_prev_v_b_ = v_b_now;

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

  // 4) NN inference → BODY velocity + covariance
  Vec3 v_b_mean;   // [u, v, w] in BODY
  Mat3 Rvb;
  if (!nn_->infer(window, v_b_mean, Rvb)) return;

  // 5) Apply the EKF velocity update in BODY
  (void)updateNNBodyVel(v_b_mean, Rvb, /*w=*/0.5);
}

bool NN_qObs_Aided_EKF_v8::updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w) {
  // residual: z - h(x) with h(x)=v^n
  const Vec3 res = z_v_nav - x_.v;

  // H = [0_{3x3}  I_{3x3}  0_{3x3}]
  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3).setIdentity();

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rv / wt;
  const Mat3 S = (H * P_ * H.transpose()) + R;

  // NIS & gate
  const double nis = res.transpose() * S.inverse() * res;

  // ---- ENABLE GATING (if configured) ----
  if (cfg_.chi2_gate_vec3 > 0.0 && nis >= cfg_.chi2_gate_vec3) {
    return false;
  }

  const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

  // Inject nominal
  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  // Joseph stabilized covariance update
  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
  return true;
}

const State9_v8& NN_qObs_Aided_EKF_v8::state() const { return x_; }
const Mat99&  NN_qObs_Aided_EKF_v8::cov()   const { return P_; }

void NN_qObs_Aided_EKF_v8::setState(const State9_v8& x, const Mat99& P){ x_ = x; P_ = P; }

void NN_qObs_Aided_EKF_v8::propagate(const Vec3& omega_b_meas, const Vec3& accel_b_meas, double dt) {
  last_omega_b_meas_ = omega_b_meas; // cached for optional velocity updates

  // Specific force minus estimated bias (BODY)
  const Vec3 f_b = accel_b_meas - x_.b_a;

  // Acceleration in NAV with g in Down (+)
  const Vec3 a_n = R_nb_ * f_b + g_n_;

  // Nominal state propagation
  x_.v += a_n * dt;
  x_.p += x_.v * dt + 0.5 * a_n * dt * dt;

  // Linearized error propagation: δx=[δp, δv, δb_a]
  Mat99 A = Mat99::Zero();
  A.block<3,3>(0,3).setIdentity();            // δp_dot = δv
  A.block<3,3>(3,6) = - R_nb_;                // δv_dot ≈ -R_nb δb_a
  if (cfg_.tau_ba > 0.0) {
    A.block<3,3>(6,6) = -(1.0/cfg_.tau_ba) * Mat3::Identity(); // optional bias leak
  }

  // Discretize (Euler)
  const Mat99 Ad = Mat99::Identity() + A * dt;

  // Process-noise mapping (continuous): w = [n_a (BODY); n_ba_rw]
  Mat96 E = Mat96::Zero();
  E.block<3,3>(3,0) = R_nb_;                  // accel white noise -> δv
  E.block<3,3>(6,3) = Mat3::Identity();       // bias random walk -> δb_a

  Eigen::Matrix<double,6,6> Qc = Eigen::Matrix<double,6,6>::Zero();
  Qc.block<3,3>(0,0) = (cfg_.sigma_a     * cfg_.sigma_a)     * Mat3::Identity();
  Qc.block<3,3>(3,3) = (cfg_.sigma_ba_rw * cfg_.sigma_ba_rw) * Mat3::Identity();

  const Mat99 Qd = (E * Qc * E.transpose()) * dt;

  // Covariance time update
  P_ = Ad * P_ * Ad.transpose() + Qd;
}

bool NN_qObs_Aided_EKF_v8::updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w){
  const Vec3 h  = x_.p + R_nb_ * r_body; // predicted antenna position
  const Vec3 res = z_nav_ant - h;        // residual

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,0) = Mat3::Identity();  // d/dp

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  return kalmanUpdate(res, H, Rpos / wt, cfg_.chi2_gate_pos3);
}

bool NN_qObs_Aided_EKF_v8::updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                                      const Vec3& omega_b_meas, double w){
  const Vec3 v_ant_pred = x_.v + R_nb_ * (omega_b_meas.cross(r_body));
  const Vec3 res = z_v_nav_ant - v_ant_pred;

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = Mat3::Identity(); // d/dv

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  return kalmanUpdate(res, H, Rvel / wt, cfg_.chi2_gate_vec3);
}

bool NN_qObs_Aided_EKF_v8::updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w){
  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b_pred = R_bn * x_.v;   // predicted body velocity at CG
  const Vec3 res = z_v_b - v_b_pred;   // residual in BODY

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = R_bn;            // d(R_bn v_n)/d v_n = R_bn

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  return kalmanUpdate(res, H, Rvb / wt, cfg_.chi2_gate_vec3);
}

Eigen::Matrix<double,9,1> NN_qObs_Aided_EKF_v8::getState9() const {
  Eigen::Matrix<double,9,1> x;
  x.segment<3>(0) = x_.p;
  x.segment<3>(3) = x_.v;
  x.segment<3>(6) = x_.b_a;
  return x;
}

Eigen::VectorXd NN_qObs_Aided_EKF_v8::getState12(const Vec3& b_gyro_hat) const {
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

bool NN_qObs_Aided_EKF_v8::kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                                     const Mat3& R, double chi2_gate){
  const Mat3 S = (H * P_ * H.transpose()) + R;

  // Optional NIS gating
  if (chi2_gate > 0.0) {
    const double nis = r.transpose() * S.inverse() * r;
    if (nis >= chi2_gate) return false;
  }

  const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * r;

  // Inject nominal
  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  // Joseph stabilized covariance update
  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

} // namespace nnqekf_v8

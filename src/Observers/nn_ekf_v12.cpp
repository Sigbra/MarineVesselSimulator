// nn_ekf_v12.cpp — 9-state EKF (“NN_qObs_Aided_EKF_v12”) with external attitude (q-Obs)
// + Velocity-only NN_v12 TorchScript wrapper (STATEFUL, PIMPL) for END-frame velocity aiding.
//
// v12 Euler NN interface (Euler-only, wrapped angles):
//  - EKF buffers a canonical 10-vector per step (API remains unchanged):
//      [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n, 0]
//  - NN input dimension D is inferred from norm_stats.json (x_mean length) and MUST be:
//      D=9 -> [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n]
//  - All angles (phi/theta/psi) are wrapped to [-pi, +pi) (±180°) before normalization.
//
// This implementation is intended to be as close as possible to the v11 behavior,
// with only necessary changes for Euler-only inputs

#include "Observers/nn_ekf_v12.hpp"
#include "Utilities/calculations.hpp"  

#include <torch/script.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cctype>

namespace nnqekf_v12 {

// ----------------------- debug helpers -----------------------
#ifdef EKF_DEBUG
#ifndef DBG_PRINT_EVERY
#define DBG_PRINT_EVERY 200
#endif
#ifndef DBG_YAW_NEAR_DEG
#define DBG_YAW_NEAR_DEG 15
#endif
#ifdef DBG_DISABLE_LEVER_ARM
static constexpr bool kDbgDisableLeverArm = true;
#else
static constexpr bool kDbgDisableLeverArm = false;
#endif

static uint64_t g_dbg_counter = 0;

static inline void dbgHeader(const char* tag){
  std::cerr << "[EKF][" << tag << "] ";
}

static inline void dbgPrintVec(const char* name, const Vec3& v){
  std::cerr << name << "=[" << std::fixed << std::setprecision(4)
            << v.x() << "," << v.y() << "," << v.z() << "]";
}

static inline void dbgPrintQuat(const char* name, const Quat& q){
  std::cerr << name << "=[w:" << std::setprecision(6) << q.w
            << " x:" << q.x << " y:" << q.y << " z:" << q.z << "]";
}

static inline void dbgPrintMat3Row(const Mat3& R, int r, const char* prefix){
  std::cerr << prefix << "[" << r << "] "
            << "[" << R(r,0) << " " << R(r,1) << " " << R(r,2) << "]";
}

static inline double rad2deg(double r){ return r * 180.0 / M_PI; }

static inline bool near180(double yaw_rad){
  const double d = std::fabs(std::remainder(yaw_rad - M_PI, 2*M_PI));
  return rad2deg(d) <= DBG_YAW_NEAR_DEG;
}

static void check_R_sanity(const Mat3& R, const char* tag){
  const double det = R.determinant();
  const double rn0 = R.row(0).norm();
  const double rn1 = R.row(1).norm();
  const double rn2 = R.row(2).norm();
  const double cn0 = R.col(0).norm();
  const double cn1 = R.col(1).norm();
  const double cn2 = R.col(2).norm();
  const double o01 = std::fabs(R.row(0).dot(R.row(1)));
  const double o02 = std::fabs(R.row(0).dot(R.row(2)));
  const double o12 = std::fabs(R.row(1).dot(R.row(2)));

  if (std::fabs(det-1.0) > 5e-3 || o01>5e-3 || o02>5e-3 || o12>5e-3 ||
      std::fabs(rn0-1.0)>5e-3 || std::fabs(rn1-1.0)>5e-3 || std::fabs(rn2-1.0)>5e-3 ||
      std::fabs(cn0-1.0)>5e-3 || std::fabs(cn1-1.0)>5e-3 || std::fabs(cn2-1.0)>5e-3){
    dbgHeader(tag);
    std::cerr << "R sanity FAIL det=" << det
              << " rowNorms=["<<rn0<<","<<rn1<<","<<rn2<<"]"
              << " colNorms=["<<cn0<<","<<cn1<<","<<cn2<<"]"
              << " rowDots=["<<o01<<","<<o02<<","<<o12<<"]\n";
    dbgHeader(tag); dbgPrintMat3Row(R,0,"R"); std::cerr << "\n";
    dbgHeader(tag); dbgPrintMat3Row(R,1,"R"); std::cerr << "\n";
    dbgHeader(tag); dbgPrintMat3Row(R,2,"R"); std::cerr << "\n";
  }
}

#else
static inline double rad2deg(double r){ return r * 180.0 / M_PI; }
#endif

// END-convention yaw: ψ = atan2(R(0,0), R(1,0))
static inline double yaw_from_Rnb_END(const Mat3& R_nb){
  const double r10 = R_nb(1,0);
  const double r00 = R_nb(0,0);
  return std::atan2(r00, r10);
}

// Wrap angle to [-pi, +pi) using Utilities::ssa() which returns (-pi, pi].
// Map +pi -> -pi to match Python wrap: (a+pi)%(2pi)-pi.
static inline double wrap_pm_pi_ssa(double a)
{
  a = ssa(a);                  // (-pi, pi]
  if (a >= M_PI) a -= 2*M_PI;  // -> [-pi, pi)
  return a;
}

// Deterministic, stateless quaternion canonicalization to remove ±q ambiguity.
Quat NN_qObs_Aided_EKF_v12::canonicalizeQuat(const Quat& q_in) {
  Quat q = q_in;
  const double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
  if (n > 0.0) { q.w/=n; q.x/=n; q.y/=n; q.z/=n; }
  if (q.w < 0.0 || (std::abs(q.w) <= 1e-12 && q.z < 0.0)) {
    q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z;
  }
  return q;
}

// ============================================================
//                       NN_v12 (PIMPL)
// ============================================================
struct NN_v12::Impl {
  // device
  bool use_cuda_ = true;
  torch::Device device_{torch::kCPU};

  // ensemble modules & hidden states
  std::vector<torch::jit::script::Module> members_;
  std::vector<torch::Tensor> h_;  // per-member [L,1,H], persisted across calls

  // ---- buffered EKF row dimension (fixed by header API) ----
  static constexpr int IN_ROW_DIM = 10; // [ax,ay,az,phi,theta,psi,tau_x,tau_y,tau_n,0]
  static constexpr int V_DIM      = 3;  // [vE,vN,vD]

  // ---- model input dimension (from norm_stats.json) ----
  int in_dim_ = 0; // v12 expects 9

  // norms (size in_dim_)
  std::vector<double> x_mean_, x_std_;
  std::array<double,V_DIM>  y_mean_{}, y_std_{};
  torch::Tensor x_mean_t_, x_std_t_, y_mean_t_, y_std_t_;

  void reset_hidden() { h_.clear(); }

  // Some TorchScript exports do NOT provide init_state().
  // We infer GRU hidden-state shape from weight_hh_l{k} parameters and build zeros.
  torch::Tensor make_h0_from_gru_weights(const torch::jit::script::Module& mod, int batch) const
  {
    int max_layer = -1;
    bool has_reverse = false;
    int64_t hidden_size = -1;

    for (const auto& p : mod.named_parameters(/*recurse=*/true)) {
      const std::string& name = p.name;
      const auto pos = name.rfind("weight_hh_l");
      if (pos == std::string::npos) continue;

      // Parse digits after "weight_hh_l"
      const size_t i0 = pos + std::string("weight_hh_l").size();
      size_t i1 = i0;
      while (i1 < name.size() && std::isdigit(static_cast<unsigned char>(name[i1]))) ++i1;
      if (i1 == i0) continue;

      const int layer = std::stoi(name.substr(i0, i1 - i0));
      max_layer = std::max(max_layer, layer);

      if (name.find("_reverse", i1) != std::string::npos) has_reverse = true;

      const torch::Tensor& W = p.value; // GRU weight_hh_l{k}: [3H, H]
      if (W.defined() && W.dim() == 2 && hidden_size < 0) hidden_size = W.size(1);
    }

    if (max_layer < 0 || hidden_size <= 0) return torch::Tensor();

    const int num_layers = max_layer + 1;
    const int num_dirs   = has_reverse ? 2 : 1;
    const int64_t L = static_cast<int64_t>(num_layers * num_dirs);

    return torch::zeros({L, batch, hidden_size},
                        torch::TensorOptions().dtype(torch::kFloat).device(device_));
  }

  bool init(const std::string& model_dir,
            const std::string& norm_json,
            bool use_cuda)
  {
#ifdef TORCH_CUDA_AVAILABLE
    use_cuda_ = use_cuda && torch::cuda::is_available();
#else
    use_cuda_ = false;
#endif
    device_ = torch::Device(use_cuda_ ? torch::kCUDA : torch::kCPU);

    // ---- load norms (YAML parser can read JSON) ----
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
        std::cerr << "[NN_v12] Failed to open norm_stats: " << norm_json << "\n";
        return false;
      }
    }

    auto x_mean = ns["x_mean"]; auto x_std = ns["x_std"];
    auto y_mean = ns["y_mean"]; auto y_std = ns["y_std"];
    if (!x_mean || !x_std || !y_mean || !y_std) {
      std::cerr << "[NN_v12] norm_json missing keys x_mean/x_std/y_mean/y_std\n";
      return false;
    }
    if (!x_mean.IsSequence() || !x_std.IsSequence() || !y_mean.IsSequence() || !y_std.IsSequence()) {
      std::cerr << "[NN_v12] norm_json: x_mean/x_std/y_mean/y_std must be sequences\n";
      return false;
    }

    const int D = static_cast<int>(x_mean.size());
    if (D != 9) {
      std::cerr << "[NN_v12] norm_json: unsupported input dim D=" << D
                << " (expected 9=euler-only v12)\n";
      return false;
    }
    if ((int)x_std.size() != D) {
      std::cerr << "[NN_v12] norm_json: x_std size != x_mean size\n";
      return false;
    }
    if ((int)y_mean.size() != V_DIM || (int)y_std.size() != V_DIM) {
      std::cerr << "[NN_v12] norm_json: y_mean/y_std must have size 3\n";
      return false;
    }

    in_dim_ = D;
    x_mean_.assign(D, 0.0);
    x_std_.assign(D, 1.0);
    for (int i=0;i<D;++i){
      x_mean_[i] = x_mean[i].as<double>();
      x_std_[i]  = x_std[i].as<double>();
      if (!(std::isfinite(x_std_[i])) || std::fabs(x_std_[i]) < 1e-12) x_std_[i] = 1.0;
    }
    for (int i=0;i<V_DIM;++i){
      y_mean_[i] = y_mean[i].as<double>();
      y_std_[i]  = y_std[i].as<double>();
      if (!(std::isfinite(y_std_[i])) || std::fabs(y_std_[i]) < 1e-12) y_std_[i] = 1.0;
    }

    x_mean_t_ = torch::from_blob(x_mean_.data(), {in_dim_}, torch::kDouble).clone().to(device_);
    x_std_t_  = torch::from_blob(x_std_.data(),  {in_dim_}, torch::kDouble).clone().to(device_);
    y_mean_t_ = torch::from_blob(y_mean_.data(), {V_DIM},   torch::kDouble).clone().to(device_);
    y_std_t_  = torch::from_blob(y_std_.data(),  {V_DIM},   torch::kDouble).clone().to(device_);

    std::cerr << "[NN_v12] Norms loaded: D=" << in_dim_ << " (euler-only)\n";

    // ---- collect ONLY stateful TorchScript members ----
    namespace fs = std::filesystem;
    fs::path md(model_dir);
#ifdef MVS_PROJECT_ROOT
    if (!md.is_absolute()) {
      fs::path alt = fs::path(MVS_PROJECT_ROOT) / md;
      if (fs::exists(alt)) md = alt;
    }
#endif
    if (!fs::exists(md)) {
      std::cerr << "[NN_v12] Model path does not exist: " << md << "\n";
      return false;
    }

    auto is_pt = [](const fs::path& p){
      std::string ext = p.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      return (ext == ".pt" || ext == ".ts");
    };

    std::vector<std::string> files_stateful;

    if (fs::is_directory(md)) {
      for (auto& e : fs::directory_iterator(md)) {
        if (!e.is_regular_file()) continue;
        const auto& p = e.path();
        if (!is_pt(p)) continue;
        const std::string name = p.filename().string();
        if (name.rfind("member_", 0) == 0 &&
            name.find("_onestep_stateful") != std::string::npos) {
          files_stateful.push_back(p.string());
        }
      }
      std::sort(files_stateful.begin(), files_stateful.end());
    } else if (fs::is_regular_file(md) && is_pt(md)) {
      const std::string name = md.filename().string();
      if (name.find("_onestep_stateful") != std::string::npos) {
        files_stateful.push_back(md.string());
      }
    } else {
      std::cerr << "[NN_v12] Invalid model_dir: " << md << "\n";
      return false;
    }

    if (files_stateful.empty()) {
      std::cerr << "[NN_v12] No stateful members found (member_*_onestep_stateful.pt) in: "
                << md << "\n";
      return false;
    }

    try {
      members_.clear();
      members_.reserve(files_stateful.size());
      for (const auto& f : files_stateful) {
        auto m = torch::jit::load(f, device_);
        m.eval();
        members_.push_back(std::move(m));
        std::cerr << "[NN_v12] Loaded " << f << (use_cuda_ ? " (CUDA)\n" : " (CPU)\n");
      }
    } catch (const c10::Error& e) {
      std::cerr << "[NN_v12] Failed to load model: " << e.what() << "\n";
      return false;
    }

    if (members_.size() == 1) {
      std::cerr << "[NN_v12] WARNING: only 1 model loaded; covariance will be near floor.\n";
    } else {
      std::cerr << "[NN_v12] Ensemble size: " << members_.size() << "\n";
    }

    // Hidden states initialized lazily in infer()
    h_.clear();
    return true;
  }

  bool infer(const std::vector<Eigen::Matrix<double,IN_ROW_DIM,1>>& window,
             Vec3& v_nav_mean,
             Mat3& Rv_nav)
  {
    if (members_.empty() || window.empty()) return false;
    if (in_dim_ != 9) return false;

    torch::NoGradGuard ng;

    // -----------------------------
    // Lazy init per-member hidden states [L,1,H]
    // If we initialize here, we warm-start through the full window once.
    // -----------------------------
    bool need_warm_start = false;

    if (h_.size() != members_.size()) {
      h_.resize(members_.size());
      for (size_t m = 0; m < members_.size(); ++m) {
        torch::Tensor h0;

        // v11-like preference: use init_state() when available.
        try {
          h0 = members_[m].run_method("init_state", 1).toTensor().to(device_).to(torch::kFloat);
        } catch (const c10::Error&) {
          // v12 robustness: fall back to inferred zeros (works with exports lacking init_state()).
          h0 = make_h0_from_gru_weights(members_[m], /*batch=*/1);
        }

        if (!h0.defined()) {
          std::cerr << "[NN_v12] Failed to initialize hidden state for member " << m << ".\n"
                    << "         Export must be stateful (forward(x,h)->(y,h_next)) and contain GRU params.\n";
          return false;
        }
        h_[m] = h0;
      }
      need_warm_start = true;
    }

    // Warm-start: feed 0..T-1. Steady-state: feed only last sample.
    const int T = static_cast<int>(window.size());
    const int start_idx = (need_warm_start ? 0 : (T - 1));

    // Pre-allocate tensor [1,1,D] (double for normalization, then cast to float)
    torch::Tensor x = torch::empty({1, 1, in_dim_},
        torch::TensorOptions().dtype(torch::kDouble).device(device_));

    const int M = static_cast<int>(members_.size());
    std::vector<Vec3> preds;
    preds.reserve(M);

    for (int m = 0; m < M; ++m) {

      torch::Tensor y_last;  // [1,1,3]

      for (int i = start_idx; i < T; ++i) {
        const auto& r = window[i];

        const double ax = r(0), ay = r(1), az = r(2);

        // Wrap Euler to match dataset/training ([-pi, +pi)) using ssa()
        const double phi   = wrap_pm_pi_ssa(r(3));
        const double theta = wrap_pm_pi_ssa(r(4));
        const double psi   = wrap_pm_pi_ssa(r(5));

        const double tau_x = r(6), tau_y = r(7), tau_n = r(8);

        // Fill x (D=9 Euler-only): [ax, ay, az, phi, theta, psi, tau_x, tau_y, tau_n]
        {
          auto xa = x.accessor<double,3>();
          xa[0][0][0] = ax;
          xa[0][0][1] = ay;
          xa[0][0][2] = az;
          xa[0][0][3] = phi;
          xa[0][0][4] = theta;
          xa[0][0][5] = psi;
          xa[0][0][6] = tau_x;
          xa[0][0][7] = tau_y;
          xa[0][0][8] = tau_n;
        }

        // Normalize: (x - mean) / std
        torch::Tensor x_norm =
          (x - x_mean_t_.view({1,1,in_dim_})) / x_std_t_.view({1,1,in_dim_});
        torch::Tensor xf = x_norm.to(torch::kFloat);

        // forward(x:[1,1,D], h:[L,1,H]) -> (y:[1,1,3], h_next)
        c10::IValue out_iv;
        try {
          out_iv = members_[m].forward({ xf, h_[m] });
        } catch (const c10::Error& e) {
          std::cerr << "[NN_v12] forward() failed on member " << (m+1)
                    << " : " << e.what() << "\n";
          return false;
        }

        auto tup = out_iv.toTuple();
        const auto& elems = tup->elements();
        if (elems.size() != 2 || !elems[0].isTensor() || !elems[1].isTensor()) {
          std::cerr << "[NN_v12] forward() did not return (Tensor, Tensor) on member " << m << "\n";
          return false;
        }

        y_last = elems[0].toTensor();                 // [1,1,3]
        torch::Tensor hnext = elems[1].toTensor();    // [L,1,H]
        h_[m] = hnext.to(device_).to(torch::kFloat);  // persist hidden state
      }

      // Validate output shape
      if (y_last.dim() != 3 || y_last.size(0) != 1 || y_last.size(1) != 1 || y_last.size(2) != 3) {
        std::cerr << "[NN_v12] Bad output shape from member " << m << ": " << y_last.sizes() << "\n";
        return false;
      }

      // De-normalize on CPU double
      torch::Tensor y3 = y_last.index({0,0}).to(torch::kDouble).to(torch::kCPU); // [3]
      double vv[3];
      std::memcpy(vv, y3.data_ptr<double>(), 3*sizeof(double));

      Vec3 v;
      for (int k = 0; k < 3; ++k) v(k) = vv[k]*y_std_[k] + y_mean_[k]; // END: [vE,vN,vD]
      preds.push_back(v);
    }

    // Ensemble mean and covariance in END
    v_nav_mean.setZero();
    for (const auto& v : preds) v_nav_mean += v;
    v_nav_mean /= std::max(1, M);

    Eigen::Matrix<double,3,Eigen::Dynamic> Dm(3, M);
    for (int i = 0; i < M; ++i) Dm.col(i) = preds[i] - v_nav_mean;

    if (M >= 2) {
      Rv_nav = (Dm * Dm.transpose()) / static_cast<double>(M - 1);
    } else {
      Rv_nav = Mat3::Zero();
    }
    Rv_nav += 1e-6 * Mat3::Identity();

    return true;
  }

}; // close Impl

// --------------------- NN_v12 public wrapper ---------------------
NN_v12::NN_v12()  : impl_(std::make_unique<Impl>()) {}
NN_v12::~NN_v12() = default;

bool NN_v12::init(const std::string& model_dir,
                  const std::string& norm_json,
                  bool use_cuda) {
  return impl_->init(model_dir, norm_json, use_cuda);
}

bool NN_v12::infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
                   Vec3& v_nav_mean,
                   Mat3& Rv_nav) {
  return impl_->infer(window, v_nav_mean, Rv_nav);
}

void NN_v12::reset_states() {
  if (impl_) impl_->reset_hidden();
}

// ============================================================
//                       EKF v12
// ============================================================

NN_qObs_Aided_EKF_v12::NN_qObs_Aided_EKF_v12(const Config_v12& cfg) : cfg_(cfg) {
  P_.setIdentity();
  P_ *= 1e-2;
  R_nb_.setIdentity();
  g_n_ = Vec3(0,0,cfg_.g);

  // -------- Heave spring–damper constants from RAN (option A) --------
  const double rho        = 1025.0;
  const double L          = 5.46;
  const double Beam_pont  = 0.50;
  const double Cw_pont    = 0.80;
  const double m          = 850.0;
  const double zeta       = 0.30;

  const double Aw_pont = Cw_pont * L * Beam_pont;
  const double K_h     = rho * cfg_.g * (2.0 * Aw_pont);
  const double C_h     = 2.0 * zeta * std::sqrt(m * K_h);

  k_z_  = K_h / m;
  c_z_  = C_h / m;
  z0_   = 0.0;

#ifdef EKF_DEBUG
  dbgHeader("init");
  std::cerr << "Heave k_z="<<k_z_<<" c_z="<<c_z_<<" g="<<cfg_.g<<"\n";
#endif
}

void NN_qObs_Aided_EKF_v12::setHeaveEquilibrium(double z0){ z0_ = z0; }

void NN_qObs_Aided_EKF_v12::setRotationNavFromBody(const Mat3& R_nb) {
  R_nb_ = R_nb;
#ifdef EKF_DEBUG
  const double yaw = yaw_from_Rnb_END(R_nb_);
  dbgHeader("setR"); std::cerr << "direct R_nb set, yaw="<<rad2deg(yaw)<<" deg\n";
  check_R_sanity(R_nb_, "setR");
#endif
}

void NN_qObs_Aided_EKF_v12::setRotationFromQuat(const Quat& q_nb) {
  const Quat q_can = q_nb; //canonicalizeQuat(q_nb);
  R_nb_ = RnbFromQuatCustom(q_can);
}

void NN_qObs_Aided_EKF_v12::setNN(NN_v12* nn, int seq_len, int stride)
{
  nn_ = nn;
  nn_seq_len_ = std::max(1, seq_len);
  nn_stride_  = std::max(1, stride);
  nn_count_   = 0;
  nn_buf_.clear();

  // Match v11 warm-start semantics
  nn_warmed_ = false;

  // Ensure the NN hidden state is cleared so first infer(window) truly warm-starts
  if (nn_) nn_->reset_states();
}

void NN_qObs_Aided_EKF_v12::nn_prune_() {
  const int MAX_KEEP = std::max(nn_seq_len_, 4);
  while ((int)nn_buf_.size() > MAX_KEEP) nn_buf_.pop_front();
}

// Feed one sample to NN buffer (v11-like cadence/warm-start), but pack Euler angles.
void NN_qObs_Aided_EKF_v12::feedNN(const Vec3& accel_b,
                                  const Quat& q_nb,
                                  double tau_x,
                                  double tau_y,
                                  double tau_n)
{
  if (!nn_ || nn_seq_len_ <= 0) return;

  // Use provided attitude (v11-like). (R_nb_ may also be set elsewhere.)
  const Quat q_can = canonicalizeQuat(q_nb);
  const Mat3 R = RnbFromQuatCustom(q_can);   // BODY -> END

  auto clamp1 = [](double s){ return std::max(-1.0, std::min(1.0, s)); };

  // Extract Euler consistent with your END ZYX convention:
  // theta = asin(-R(2,0)), phi = atan2(R(2,1), R(2,2)), psi = atan2(R(0,0), R(1,0)).
  double theta = std::asin(clamp1(-R(2,0)));
  double phi   = std::atan2(R(2,1), R(2,2));
  double psi   = yaw_from_Rnb_END(R);

  // Wrap exactly like training-time: [-pi, +pi)
  phi   = wrap_pm_pi_ssa(phi);
  theta = wrap_pm_pi_ssa(theta);
  psi   = wrap_pm_pi_ssa(psi);

  // Canonical buffered row (10):
  // [ax ay az phi theta psi tau_x tau_y tau_n 0]
  Eigen::Matrix<double,10,1> row;
  row << accel_b.x(), accel_b.y(), accel_b.z(),
         phi, theta, psi,
         tau_x, tau_y, tau_n,
         0.0;

  nn_buf_.push_back(row);
  nn_prune_();

  nn_count_++;

  // v11 semantics: do NOT call infer() before we have nn_seq_len_ samples
  if ((int)nn_buf_.size() < nn_seq_len_) {
    nn_warmed_ = false;
    return;
  }

  Vec3 v_nav_mean;
  Mat3 Rv_nav;

  // 1) Warm-start once: feed the last nn_seq_len_ samples as a window
  if (!nn_warmed_) {
    std::vector<Eigen::Matrix<double,10,1>> window;
    window.reserve(nn_seq_len_);

    auto it = nn_buf_.end();
    for (int i = 0; i < nn_seq_len_; ++i) --it;
    for (int i = 0; i < nn_seq_len_; ++i, ++it) window.push_back(*it);

    if (!nn_->infer(window, v_nav_mean, Rv_nav)) return;

    nn_warmed_ = true;

    // Apply EKF update only on stride (warm-start still updates NN hidden state)
    if ((nn_count_ % nn_stride_) == 0) {
      (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1);
    }
    return;
  }

  // 2) Steady-state: step the GRU every tick with the newest sample only
  {
    std::vector<Eigen::Matrix<double,10,1>> one;
    one.reserve(1);
    one.push_back(row);

    if (!nn_->infer(one, v_nav_mean, Rv_nav)) return;
  }

  // Apply EKF update only on stride
  if ((nn_count_ % nn_stride_) != 0) return;

  (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1);
}

bool NN_qObs_Aided_EKF_v12::updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w)
{
  // residual: z - h(x) with h(x)=v^n (END)
  const Vec3 res = z_v_nav - x_.v;

  // H = [0_{3x3}  I_{3x3}  0_{3x3}]
  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3).setIdentity();

  const double wt = (std::isfinite(w) && w > 0.0) ? w : 1.0;
  const Mat3 R = Rv / wt;

  // Innovation covariance: S = HPH^T + R
  const Mat3 S = (H * P_ * H.transpose()) + R;

  // Prefer solve over explicit inverse
  Eigen::LDLT<Mat3> ldltS(S);
  if (ldltS.info() != Eigen::Success) return false;

  // Optional NIS gate: nis = res^T S^{-1} res
  if (cfg_.chi2_gate_vec3 > 0.0) {
    const Vec3 Sinv_res = ldltS.solve(res);
    const double nis = res.dot(Sinv_res);
    if (nis >= cfg_.chi2_gate_vec3) return false;
  }

  // Kalman gain: K = P H^T S^{-1}
  const Eigen::Matrix<double,9,3> PHt = P_ * H.transpose();
  const Eigen::Matrix<double,9,3> K   = PHt * ldltS.solve(Mat3::Identity());

  // State correction: dx = K * res
  const Eigen::Matrix<double,9,1> dx = K * res;

#ifdef EKF_DEBUG
  if ((++g_dbg_counter % DBG_PRINT_EVERY) == 0) {
    const double yaw = yaw_from_Rnb_END(R_nb_);
    dbgHeader("updNNv_full");
    dbgPrintVec("res_vn", res); std::cerr << "  ";
    std::cerr << "d|p|="   << dx.segment<3>(0).norm()
              << "  d|v|=" << dx.segment<3>(3).norm()
              << "  d|b_a|="<< dx.segment<3>(6).norm()
              << "  yaw="  << rad2deg(yaw) << "deg\n";
  }
#endif

  // Full update (position, velocity, accel bias)
  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  // Joseph-form covariance update: P = (I-KH) P (I-KH)^T + K R K^T
  const Eigen::Matrix<double,9,9> I = Eigen::Matrix<double,9,9>::Identity();
  const Eigen::Matrix<double,9,9> IKH = I - (K * H);
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  // Optional: enforce symmetry
  P_ = 0.5 * (P_ + P_.transpose());

  return true;
}


const State9_v12& NN_qObs_Aided_EKF_v12::state() const { return x_; }
const Mat99&     NN_qObs_Aided_EKF_v12::cov()   const { return P_; }

void NN_qObs_Aided_EKF_v12::setState(const State9_v12& x, const Mat99& P){ x_ = x; P_ = P; }

void NN_qObs_Aided_EKF_v12::propagate(const Vec3& omega_b_meas,
                                      const Vec3& accel_b_meas,
                                      double dt)
{
  if (!(dt > 0.0)) return;
  last_omega_b_meas_ = omega_b_meas;

#ifdef ENABLE_IMU_LEVER_ARM
  static Vec3 omega_prev = omega_b_meas;
  const Vec3 alpha_b = (omega_b_meas - omega_prev) / std::max(1e-6, dt);
  omega_prev = omega_b_meas;

  const Vec3 r_ib_b(0.0, 0.0, 0.0);
  const Vec3 a_corr = accel_b_meas
                    - (alpha_b.cross(r_ib_b) + omega_b_meas.cross(omega_b_meas.cross(r_ib_b)));
  const Vec3 f_b = a_corr - x_.b_a;
#else
  const Vec3 f_b = accel_b_meas - x_.b_a;
#endif

  Vec3 a_n = R_nb_ * f_b + g_n_;

  // Optional heave spring–damper (Down axis)
  // a_n.z() += -k_z_ * (x_.p.z() - z0_) - c_z_ * x_.v.z();

  const Vec3 dp = x_.v * dt + 0.5 * a_n * dt * dt;
  const Vec3 dv = a_n * dt;

  x_.p += dp;
  x_.v += dv;

  Mat99 A = Mat99::Zero();
  A.block<3,3>(0,3).setIdentity();
  A.block<3,3>(3,6) = -R_nb_;
  // A(5,2) += -k_z_;
  // A(5,5) += -c_z_;

  if (cfg_.tau_ba > 0.0) {
    x_.b_a += (-(1.0 / cfg_.tau_ba) * x_.b_a) * dt;
    A.block<3,3>(6,6) = -(1.0 / cfg_.tau_ba) * Mat3::Identity();
  }

  const Mat99 I9 = Mat99::Identity();
  const Mat99 Ad = I9 + A * dt + 0.5 * (A * A) * (dt * dt);

  Mat3 Qa_body = (cfg_.sigma_a * cfg_.sigma_a) * Mat3::Identity();
  Mat3 S = R_nb_ * Qa_body * R_nb_.transpose();
  const double dt2 = dt * dt, dt3 = dt2 * dt;
  Mat99 Qd = Mat99::Zero();
  Mat3 Qpv = (dt2 / 2.0) * S;
  Qd.block<3,3>(0,0) = (dt3 / 3.0) * S;
  Qd.block<3,3>(0,3) = Qpv;
  Qd.block<3,3>(3,0) = Qpv.transpose();
  Qd.block<3,3>(3,3) = dt * S;
  Qd.block<3,3>(6,6) += (cfg_.sigma_ba_rw * cfg_.sigma_ba_rw) * dt * Mat3::Identity();

  P_ = Ad * P_ * Ad.transpose() + Qd;

#ifdef EKF_DEBUG
  const double yaw = yaw_from_Rnb_END(R_nb_);
  if ((g_dbg_counter % DBG_PRINT_EVERY) == 0 || near180(yaw)) {
    dbgHeader("prop");
    std::cerr << "dt="<<dt<<" yaw="<<rad2deg(yaw)<<"deg";
    if (near180(yaw)) std::cerr << " **NEAR 180°**";
    std::cerr << "  ";
    dbgPrintVec("f_b", f_b); std::cerr << "  ";
    dbgPrintVec("a_n", a_n); std::cerr << "  ";
    dbgPrintVec("dp", dp); std::cerr << "  ";
    dbgPrintVec("dv", dv); std::cerr << "  ";
    dbgPrintVec("v", x_.v); std::cerr << "\n";

    Mat3 A36 = A.block<3,3>(3,6);
    dbgHeader("prop");
    std::cerr << "A(3,6) vs -R_nb_ max|Δ|="
              << (A36 + R_nb_).cwiseAbs().maxCoeff()
              << "  (expect ~0)\n";
  }
#endif
}

bool NN_qObs_Aided_EKF_v12::updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w){
#if defined(EKF_DEBUG) && defined(DBG_DISABLE_LEVER_ARM)
  const Vec3 h  = x_.p;
#else
  const Vec3 h  = x_.p + R_nb_ * r_body;
#endif
  const Vec3 res = z_nav_ant - h;

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,0) = Mat3::Identity();

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;

  const Mat3 R = Rpos / wt;
  Eigen::Matrix3d S = (H * P_ * H.transpose()) + R;
  Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.ldlt().solve(Eigen::Matrix3d::Identity());
  const Eigen::Matrix<double,9,1> dx = K * res;

  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

bool NN_qObs_Aided_EKF_v12::updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                                          const Vec3& omega_b_meas, double w){
#if defined(EKF_DEBUG) && defined(DBG_DISABLE_LEVER_ARM)
  const Vec3 la = Vec3::Zero();
#else
  const Vec3 la = R_nb_ * (omega_b_meas.cross(r_body));
#endif
  const Vec3 v_ant_pred = x_.v + la;
  const Vec3 res = z_v_nav_ant - v_ant_pred;

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = Mat3::Identity();

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rvel / wt;
  const Mat3 S = (H * P_ * H.transpose()) + R;
  const Eigen::Matrix<double,9,3> K  = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

  x_.v   += dx.segment<3>(3);
  x_.p   += dx.segment<3>(0);
  x_.b_a += dx.segment<3>(6);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

bool NN_qObs_Aided_EKF_v12::updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w){
  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b_pred = R_bn * x_.v;
  const Vec3 res = z_v_b - v_b_pred;

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = R_bn;

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rvb / wt;
  const Mat3 S = (H * P_ * H.transpose()) + R;
  const Eigen::Matrix<double,9,3> K  = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

  Eigen::Matrix<double,9,9> J = Eigen::Matrix<double,9,9>::Zero();
  J.block<3,3>(3,3).setIdentity();
  const Eigen::Matrix<double,9,1> Jdx = J * dx;
  x_.v   += Jdx.segment<3>(3);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - (J * K * H);
  P_ = IKH * P_ * IKH.transpose() + (J * K) * R * (J * K).transpose();

  return true;
}

Eigen::Matrix<double,9,1> NN_qObs_Aided_EKF_v12::getState9() const {
  Eigen::Matrix<double,9,1> x9;
  x9.segment<3>(0) = x_.p;
  x9.segment<3>(3) = x_.v;
  x9.segment<3>(6) = x_.b_a;
  return x9;
}

// 12-state legacy vector [u v w p q r x y z phi theta psi]^T
Eigen::Matrix<double,12,1> NN_qObs_Aided_EKF_v12::getState12(const Vec3& b_gyro_hat) const
{
  Eigen::Matrix<double,12,1> x12;

  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b  = R_bn * x_.v;

  const Vec3 omega_b = last_omega_b_meas_ - b_gyro_hat;

  auto clamp1 = [](double s){ return std::max(-1.0, std::min(1.0, s)); };
  const double r20 = R_nb_(2,0);
  const double r21 = R_nb_(2,1);
  const double r22 = R_nb_(2,2);
  const double r10 = R_nb_(1,0);
  const double r00 = R_nb_(0,0);

  double theta = -std::asin(clamp1(r20));
  double phi   = std::atan2(r21, r22);
  double psi   = std::atan2(r00, r10);

  phi   = wrap_pm_pi_ssa(phi);
  theta = wrap_pm_pi_ssa(theta);
  psi   = wrap_pm_pi_ssa(psi);

  x12 << v_b.x(), v_b.y(), v_b.z(),
          omega_b.x(), omega_b.y(), omega_b.z(),
          x_.p.x(), x_.p.y(), x_.p.z(),
          phi, theta, psi;

  return x12;
}

bool NN_qObs_Aided_EKF_v12::kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                                         const Mat3& R, double chi2_gate){
  const Mat3 S = (H * P_ * H.transpose()) + R;

  if (chi2_gate > 0.0) {
    const double nis = r.transpose() * S.inverse() * r;
    if (nis >= chi2_gate) return false;
  }

  const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * r;

  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

} // namespace nnqekf_v12

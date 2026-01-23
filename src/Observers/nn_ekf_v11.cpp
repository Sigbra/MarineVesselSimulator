// nn_ekf_v11.cpp — 9-state EKF (“NN_qObs_Aided_EKF_v11”) with external attitude (q-Obs)
// + Velocity-only NN_v11 TorchScript wrapper (STATEFUL, PIMPL) for END-frame velocity aiding.

#include "Observers/nn_ekf_v11.hpp"
#include "Utilities/calculations.hpp"   // Quat, RnbFromQuatCustom(...)

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

namespace nnqekf_v11 {

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

// Deterministic, stateless quaternion canonicalization to remove ±q ambiguity.
// Matches the rule used in calculations.cpp (normalize, then tie-break on w/z).
Quat NN_qObs_Aided_EKF_v11::canonicalizeQuat(const Quat& q_in) {
  Quat q = q_in;
  const double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
  if (n > 0.0) { q.w/=n; q.x/=n; q.y/=n; q.z/=n; }
  // same hemisphere rule as before
  if (q.w < 0.0 || (std::abs(q.w) <= 1e-12 && q.z < 0.0)) {
    q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z;
  }
  return q;
}


// ============================================================
//                       NN_v11 (PIMPL)
// ============================================================
struct NN_v11::Impl {
  // device
  bool use_cuda_ = true;
  torch::Device device_{torch::kCPU};

  // ensemble modules & hidden states
  std::vector<torch::jit::script::Module> members_;
  std::vector<torch::Tensor> h_;  // per-member [L,1,H], persisted across calls

  // norms
  static constexpr int IN_DIM = 10; // [ax,ay,az,qw,qx,qy,qz, tau_x, tau_y, tau_n]
  static constexpr int V_DIM  = 3;  // [vE,vN,vD]

  std::array<double,IN_DIM> x_mean_{}, x_std_{};
  std::array<double,V_DIM>  y_mean_{}, y_std_{};
  torch::Tensor x_mean_t_, x_std_t_, y_mean_t_, y_std_t_;
  NN_v11::QuatInputPolicy quat_policy_ = NN_v11::QuatInputPolicy::Hemisphere;


//   bool init(const std::string& model_dir,
//             const std::string& norm_json,
//             bool use_cuda)
//   {
// #ifdef TORCH_CUDA_AVAILABLE
//     use_cuda_ = use_cuda && torch::cuda::is_available();
// #else
//     use_cuda_ = false;
// #endif
//     device_ = torch::Device(use_cuda_ ? torch::kCUDA : torch::kCPU);

//     // ---- load norms (YAML parser can read JSON) ----
//     YAML::Node ns;
//     {
//       namespace fs = std::filesystem;
//       fs::path p(norm_json);
// #ifdef MVS_PROJECT_ROOT
//       if (!p.is_absolute()) {
//         fs::path alt = fs::path(MVS_PROJECT_ROOT) / p;
//         if (fs::exists(alt)) p = alt;
//       }
// #endif
//       try {
//         ns = YAML::LoadFile(p.string());
//       } catch (const YAML::BadFile&) {
//         std::cerr << "[NN_v11] Failed to open norm_stats: " << norm_json << "\n";
//         return false;
//       }
//     }
//     auto x_mean = ns["x_mean"]; auto x_std = ns["x_std"];
//     auto y_mean = ns["y_mean"]; auto y_std = ns["y_std"];
//     if (!x_mean || !x_std || !y_mean || !y_std) {
//       std::cerr << "[NN_v11] norm_json missing keys x_mean/x_std/y_mean/y_std\n";
//       return false;
//     }
//     for (int i=0;i<IN_DIM;++i){ x_mean_[i]=x_mean[i].as<double>(); x_std_[i]=x_std[i].as<double>(); }
//     for (int i=0;i<V_DIM;++i){  y_mean_[i]=y_mean[i].as<double>(); y_std_[i]=y_std[i].as<double>(); }

//     x_mean_t_ = torch::from_blob(x_mean_.data(), {IN_DIM}, torch::kDouble).clone().to(device_);
//     x_std_t_  = torch::from_blob(x_std_.data(),  {IN_DIM}, torch::kDouble).clone().to(device_);
//     y_mean_t_ = torch::from_blob(y_mean_.data(), {V_DIM},  torch::kDouble).clone().to(device_);
//     y_std_t_  = torch::from_blob(y_std_.data(),  {V_DIM},  torch::kDouble).clone().to(device_);

//     // ---- collect ONLY stateful TorchScript members ----
//     namespace fs = std::filesystem;
//     fs::path md(model_dir);
// #ifdef MVS_PROJECT_ROOT
//     if (!md.is_absolute()) {
//       fs::path alt = fs::path(MVS_PROJECT_ROOT) / md;
//       if (fs::exists(alt)) md = alt;
//     }
// #endif
//     if (!fs::exists(md)) {
//       std::cerr << "[NN_v11] Model path does not exist: " << md << "\n";
//       return false;
//     }

//     auto is_pt = [](const fs::path& p){
//       std::string ext = p.extension().string();
//       std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
//       return (ext == ".pt" || ext == ".ts");
//     };

//     std::vector<std::string> files_stateful;

//     if (fs::is_directory(md)) {
//       for (auto& e : fs::directory_iterator(md)) {
//         if (!e.is_regular_file()) continue;
//         const auto& p = e.path();
//         if (!is_pt(p)) continue;
//         const std::string name = p.filename().string();
//         // Accept only member_*_onestep_stateful.pt
//         if (name.rfind("member_", 0) == 0 &&
//             name.find("_onestep_stateful") != std::string::npos) {
//           files_stateful.push_back(p.string());
//         }
//       }
//       std::sort(files_stateful.begin(), files_stateful.end());
//     } else if (fs::is_regular_file(md) && is_pt(md)) {
//       const std::string name = md.filename().string();
//       if (name.find("_onestep_stateful") != std::string::npos) {
//         files_stateful.push_back(md.string());
//       }
//     } else {
//       std::cerr << "[NN_v11] Invalid model_dir: " << md << "\n";
//       return false;
//     }

//     if (files_stateful.empty()) {
//       std::cerr << "[NN_v11] No stateful members found (files named member_*_onestep_stateful.pt) in: "
//                 << md << "\n";
//       return false;
//     }

//     try {
//       members_.clear();
//       members_.reserve(files_stateful.size());
//       for (const auto& f : files_stateful) {
//         auto m = torch::jit::load(f, device_);
//         m.eval();
//         members_.push_back(std::move(m));
//         std::cerr << "[NN_v11] Loaded " << f << (use_cuda_ ? " (CUDA)\n" : " (CPU)\n");
//       }
//     } catch (const c10::Error& e) {
//       std::cerr << "[NN_v11] Failed to load model: " << e.what() << "\n";
//       return false;
//     }

//     if (members_.size() == 1) {
//       std::cerr << "[NN_v11] WARNING: only 1 model loaded; covariance will be near floor.\n";
//     } else {
//       std::cerr << "[NN_v11] Ensemble size: " << members_.size() << "\n";
//     }

//     // Hidden states initialized lazily in infer()
//     h_.clear();
//     return true;
//   }
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
        std::cerr << "[NN_v11] Failed to open norm_stats: " << norm_json << "\n";
        return false;
      }
    }
    auto x_mean = ns["x_mean"]; auto x_std = ns["x_std"];
    auto y_mean = ns["y_mean"]; auto y_std = ns["y_std"];
    if (!x_mean || !x_std || !y_mean || !y_std) {
      std::cerr << "[NN_v11] norm_json missing keys x_mean/x_std/y_mean/y_std\n";
      return false;
    }
    for (int i=0;i<IN_DIM;++i){ x_mean_[i]=x_mean[i].as<double>(); x_std_[i]=x_std[i].as<double>(); }
    for (int i=0;i<V_DIM;++i){  y_mean_[i]=y_mean[i].as<double>(); y_std_[i]=y_std[i].as<double>(); }

    x_mean_t_ = torch::from_blob(x_mean_.data(), {IN_DIM}, torch::kDouble).clone().to(device_);
    x_std_t_  = torch::from_blob(x_std_.data(),  {IN_DIM}, torch::kDouble).clone().to(device_);
    y_mean_t_ = torch::from_blob(y_mean_.data(), {V_DIM},  torch::kDouble).clone().to(device_);
    y_std_t_  = torch::from_blob(y_std_.data(),  {V_DIM},  torch::kDouble).clone().to(device_);

    // ---- resolve model path ----
    namespace fs = std::filesystem;
    fs::path md(model_dir);
  #ifdef MVS_PROJECT_ROOT
    if (!md.is_absolute()) {
      fs::path alt = fs::path(MVS_PROJECT_ROOT) / md;
      if (fs::exists(alt)) md = alt;
    }
  #endif
    if (!fs::exists(md)) {
      std::cerr << "[NN_v11] Model path does not exist: " << md << "\n";
      return false;
    }

    // ---- infer quaternion input preprocessing policy from folder name ----
    // Example paths:
    //   nn_model_..._hem/ts/...
    //   nn_model_..._sign/ts/...
    quat_policy_ = inferQuatPolicyFromModelPath(md);
    std::cerr << "[NN_v11] Quaternion input policy: "
              << (quat_policy_ == NN_v11::QuatInputPolicy::Hemisphere ? "Hemisphere" : "SignContinuity")
              << " (inferred from model path)\n";

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
        // Accept only member_*_onestep_stateful.pt
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
      std::cerr << "[NN_v11] Invalid model_dir: " << md << "\n";
      return false;
    }

    if (files_stateful.empty()) {
      std::cerr << "[NN_v11] No stateful members found (files named member_*_onestep_stateful.pt) in: "
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
        std::cerr << "[NN_v11] Loaded " << f << (use_cuda_ ? " (CUDA)\n" : " (CPU)\n");
      }
    } catch (const c10::Error& e) {
      std::cerr << "[NN_v11] Failed to load model: " << e.what() << "\n";
      return false;
    }

    if (members_.size() == 1) {
      std::cerr << "[NN_v11] WARNING: only 1 model loaded; covariance will be near floor.\n";
    } else {
      std::cerr << "[NN_v11] Ensemble size: " << members_.size() << "\n";
    }

    // Hidden states initialized lazily in infer()
    h_.clear();
    return true;
  }


  static NN_v11::QuatInputPolicy inferQuatPolicyFromModelPath(const std::filesystem::path& p_in)
  {
    auto toLower = [](std::string s){
      std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
      return s;
    };

    // Search in full resolved path string; folder naming convention is “…_hem/…” or “…_sign/…”
    const std::string s = toLower(p_in.string());

    const bool has_hem  = (s.find("_hem")  != std::string::npos);
    const bool has_sign = (s.find("_sign") != std::string::npos);

    if (has_hem && !has_sign)  return NN_v11::QuatInputPolicy::Hemisphere;
    if (has_sign && !has_hem)  return NN_v11::QuatInputPolicy::SignContinuity;

    // Ambiguous or missing: default to hemisphere and warn.
    std::cerr << "[NN_v11] WARNING: could not uniquely infer quaternion input policy from model path:\n"
              << "        " << p_in << "\n"
              << "        Expected folder name containing '_hem' or '_sign'. Defaulting to Hemisphere.\n";
    return NN_v11::QuatInputPolicy::Hemisphere;
  }


  // bool infer(const std::vector<Eigen::Matrix<double,IN_DIM,1>>& window,
  //            Vec3& v_nav_mean,
  //            Mat3& Rv_nav)
  // {
  //   if (members_.empty() || window.empty()) return false;

  //   torch::NoGradGuard ng;

  //   // -----------------------------
  //   // Lazy init per-member hidden states [L,1,H]
  //   // -----------------------------
  //   if (h_.size() != members_.size()) {
  //     h_.resize(members_.size());
  //     for (size_t m = 0; m < members_.size(); ++m) {
  //       try {
  //         // stateful wrapper must implement init_state(batch:int)
  //         h_[m] = members_[m].run_method("init_state", 1).toTensor().to(device_);
  //       } catch (const c10::Error& e) {
  //         std::cerr << "[NN_v11] init_state() missing on member " << m
  //                   << " (" << e.what() << ")\n"
  //                   << "          Did you load the *stateful* export (…_onestep_stateful.pt)?\n";
  //         return false;
  //       }
  //     }
  //   }

  //   // -----------------------------
  //   // Use ONLY the last input (one-step stateful), build x:[1,1,IN_DIM]
  //   // -----------------------------
  //   const auto& r = window.back();

  //   torch::Tensor x = torch::empty({1, 1, IN_DIM},
  //       torch::TensorOptions().dtype(torch::kDouble).device(device_));
  //   {
  //     auto xa = x.accessor<double,3>();
  //     for (int j = 0; j < IN_DIM; ++j) xa[0][0][j] = r(j);
  //   }

  //   // Normalize: (x - mean) / std
  //   x = (x - x_mean_t_.view({1,1,IN_DIM})) / x_std_t_.view({1,1,IN_DIM});
  //   torch::Tensor xf = x.to(torch::kFloat);

  //   // -----------------------------
  //   // Forward each ensemble member with its persistent hidden state
  //   // -----------------------------
  //   const int M = static_cast<int>(members_.size());
  //   std::vector<Vec3> preds; preds.reserve(M);

  //   for (int m = 0; m < M; ++m) {
  //     c10::IValue out_iv;
  //     try {
  //       // forward(x:[1,1,10], h:[L,1,H]) -> (y:[1,1,3], h_next)
  //       out_iv = members_[m].forward({ xf, h_[m] });
  //     } catch (const c10::Error& e) {
  //       std::cerr << "[NN_v11] forward() failed on member " << (m+1)
  //                 << " : " << e.what() << "\n";
  //       return false;
  //     }

  //     c10::intrusive_ptr<c10::ivalue::Tuple> tup = out_iv.toTuple();
  //     const auto& elems = tup->elements();
  //     if (elems.size() != 2 || !elems[0].isTensor() || !elems[1].isTensor()) {
  //       std::cerr << "[NN_v11] forward() did not return (Tensor, Tensor) on member " << m << "\n";
  //       return false;
  //     }

  //     torch::Tensor y     = elems[0].toTensor();  // [1,1,3]
  //     torch::Tensor hnext = elems[1].toTensor();  // [L,1,H]
  //     h_[m] = hnext;                              // persist hidden state

  //     if (y.dim() != 3 || y.size(0) != 1 || y.size(1) != 1 || y.size(2) != 3) {
  //       std::cerr << "[NN_v11] Bad output shape from member " << m << ": " << y.sizes() << "\n";
  //       return false;
  //     }

  //     // De-normalize on CPU double
  //     torch::Tensor y3 = y.index({0,0}).to(torch::kDouble).to(torch::kCPU); // [3]
  //     double vv[3]; std::memcpy(vv, y3.data_ptr<double>(), 3*sizeof(double));

  //     Vec3 v;
  //     for (int k = 0; k < 3; ++k) v(k) = vv[k]*y_std_[k] + y_mean_[k]; // END: [vE,vN,vD]
  //     preds.push_back(v);
  //   }

  //   // -----------------------------
  //   // Ensemble mean and covariance in END
  //   // -----------------------------
  //   v_nav_mean.setZero();
  //   for (const auto& v : preds) v_nav_mean += v;
  //   v_nav_mean /= std::max(1, M);

  //   Eigen::Matrix<double,3,Eigen::Dynamic> D(3, M);
  //   for (int i = 0; i < M; ++i) D.col(i) = preds[i] - v_nav_mean;

  //   if (M >= 2) {
  //     Rv_nav = (D * D.transpose()) / static_cast<double>(M - 1);
  //   } else {
  //     Rv_nav = Mat3::Zero();
  //   }
  //   Rv_nav += 1e-6 * Mat3::Identity(); // small numerical floor

  //   return true;
  // }

  bool infer(const std::vector<Eigen::Matrix<double,IN_DIM,1>>& window,
           Vec3& v_nav_mean,
           Mat3& Rv_nav)
{
  if (members_.empty() || window.empty()) return false;

  torch::NoGradGuard ng;

  // -----------------------------
  // Lazy init per-member hidden states [L,1,H]
  // If we initialize here, we warm-start through the full window once.
  // -----------------------------
  bool need_warm_start = false;

  if (h_.size() != members_.size()) {
    h_.resize(members_.size());
    for (size_t m = 0; m < members_.size(); ++m) {
      try {
        // stateful wrapper must implement init_state(batch:int)
        h_[m] = members_[m].run_method("init_state", 1).toTensor().to(device_);
      } catch (const c10::Error& e) {
        std::cerr << "[NN_v11] init_state() missing on member " << m
                  << " (" << e.what() << ")\n"
                  << "          Did you load the *stateful* export (…_onestep_stateful.pt)?\n";
        return false;
      }
    }
    need_warm_start = true;
  }

  // Warm-start: feed 0..T-1. Steady-state: feed only last sample.
  const int T = static_cast<int>(window.size());
  const int start_idx = (need_warm_start ? 0 : (T - 1));

  // Pre-allocate tensor [1,1,IN_DIM] (double for normalization, then cast to float)
  torch::Tensor x = torch::empty({1, 1, IN_DIM},
      torch::TensorOptions().dtype(torch::kDouble).device(device_));

  // -----------------------------
  // Forward each ensemble member with its persistent hidden state
  // -----------------------------
  const int M = static_cast<int>(members_.size());
  std::vector<Vec3> preds; preds.reserve(M);

  for (int m = 0; m < M; ++m) {

    torch::Tensor y_last;  // [1,1,3]

    // Step through required part of the window to update h_[m]
    for (int i = start_idx; i < T; ++i) {
      const auto& r = window[i];

      // Fill x from canonical row r (IN_DIM = 10 for v11)
      {
        auto xa = x.accessor<double,3>();
        for (int j = 0; j < IN_DIM; ++j) xa[0][0][j] = r(j);
      }

      // Normalize: (x - mean) / std
      torch::Tensor x_norm =
        (x - x_mean_t_.view({1,1,IN_DIM})) / x_std_t_.view({1,1,IN_DIM});
      torch::Tensor xf = x_norm.to(torch::kFloat);

      c10::IValue out_iv;
      try {
        // forward(x:[1,1,IN_DIM], h:[L,1,H]) -> (y:[1,1,3], h_next)
        out_iv = members_[m].forward({ xf, h_[m] });
      } catch (const c10::Error& e) {
        std::cerr << "[NN_v11] forward() failed on member " << (m+1)
                  << " : " << e.what() << "\n";
        return false;
      }

      c10::intrusive_ptr<c10::ivalue::Tuple> tup = out_iv.toTuple();
      const auto& elems = tup->elements();
      if (elems.size() != 2 || !elems[0].isTensor() || !elems[1].isTensor()) {
        std::cerr << "[NN_v11] forward() did not return (Tensor, Tensor) on member " << m << "\n";
        return false;
      }

      y_last = elems[0].toTensor();      // [1,1,3]
      torch::Tensor hnext = elems[1].toTensor(); // [L,1,H]
      h_[m] = hnext;                     // persist hidden state (updated each step)
    }

    // Validate output shape (after final step)
    if (y_last.dim() != 3 || y_last.size(0) != 1 || y_last.size(1) != 1 || y_last.size(2) != 3) {
      std::cerr << "[NN_v11] Bad output shape from member " << m << ": " << y_last.sizes() << "\n";
      return false;
    }

    // De-normalize on CPU double
    torch::Tensor y3 = y_last.index({0,0}).to(torch::kDouble).to(torch::kCPU); // [3]
    double vv[3]; std::memcpy(vv, y3.data_ptr<double>(), 3*sizeof(double));

    Vec3 v;
    for (int k = 0; k < 3; ++k) v(k) = vv[k]*y_std_[k] + y_mean_[k]; // END: [vE,vN,vD]
    preds.push_back(v);
  }

  // -----------------------------
  // Ensemble mean and covariance in END
  // -----------------------------
  v_nav_mean.setZero();
  for (const auto& v : preds) v_nav_mean += v;
  v_nav_mean /= std::max(1, M);

  Eigen::Matrix<double,3,Eigen::Dynamic> D(3, M);
  for (int i = 0; i < M; ++i) D.col(i) = preds[i] - v_nav_mean;

  if (M >= 2) {
    Rv_nav = (D * D.transpose()) / static_cast<double>(M - 1);
  } else {
    Rv_nav = Mat3::Zero();
  }
  Rv_nav += 1e-6 * Mat3::Identity(); // small numerical floor

  return true;
}

}; // <-- close Impl

// --------------------- NN_v11 public wrapper ---------------------
NN_v11::NN_v11()  : impl_(std::make_unique<Impl>()) {}
NN_v11::~NN_v11() = default;

bool NN_v11::init(const std::string& model_dir,
                  const std::string& norm_json,
                  bool use_cuda) {
  return impl_->init(model_dir, norm_json, use_cuda);
}

bool NN_v11::infer(const std::vector<Eigen::Matrix<double,10,1>>& window,
                   Vec3& v_nav_mean,
                   Mat3& Rv_nav) {
  return impl_->infer(window, v_nav_mean, Rv_nav);
}

NN_v11::QuatInputPolicy NN_v11::quatInputPolicy() const
{
  return impl_->quat_policy_;
}


// ============================================================
//                       EKF v11
// ============================================================

NN_qObs_Aided_EKF_v11::NN_qObs_Aided_EKF_v11(const Config_v11& cfg) : cfg_(cfg) {
  P_.setIdentity();
  P_ *= 1e-3;                 // mild initial uncertainty
  R_nb_.setIdentity();        // BODY->END
  g_n_ = Vec3(0,0,cfg_.g);    // Down positive

  // -------- Heave spring–damper constants from RAN (option A) --------
  const double rho        = 1025.0;   // seawater density
  const double L          = 5.46;     // hull length
  const double Beam_pont  = 0.50;     // single pontoon beam
  const double Cw_pont    = 0.80;     // waterplane coefficient
  const double m          = 850.0;    // mass (kg) — RAN nominal
  const double zeta       = 0.50;     // damping ratio (for heave)

  const double Aw_pont = Cw_pont * L * Beam_pont;
  const double K_h     = rho * cfg_.g * (2.0 * Aw_pont);          // N/m
  const double C_h     = 2.0 * zeta * std::sqrt(m * K_h);         // N·s/m

  // Per-mass coefficients used in accel (a_D += -k_z*(z-z0) - c_z*v_D)
  k_z_  = K_h / m;            // [1/s^2]
  c_z_  = C_h / m;            // [1/s]
  z0_   = 0.0;                // equilibrium Down (can be overridden)

#ifdef EKF_DEBUG
  dbgHeader("init");
  std::cerr << "Heave k_z="<<k_z_<<" c_z="<<c_z_<<" g="<<cfg_.g<<"\n";
#endif
}

void NN_qObs_Aided_EKF_v11::setHeaveEquilibrium(double z0){ z0_ = z0; }

void NN_qObs_Aided_EKF_v11::setRotationNavFromBody(const Mat3& R_nb) { 
  R_nb_ = R_nb; 
#ifdef EKF_DEBUG
  const double yaw = yaw_from_Rnb_END(R_nb_);
  dbgHeader("setR"); std::cerr << "direct R_nb set, yaw="<<rad2deg(yaw)<<" deg\n";
  check_R_sanity(R_nb_, "setR");
#endif
}

void NN_qObs_Aided_EKF_v11::setRotationFromQuat(const Quat& q_nb) {
  const Quat q_can = canonicalizeQuat(q_nb);
  R_nb_ = RnbFromQuatCustom(q_can);   // now matches Mat3 RnbFromQuatCustom(const ::Quat&)
}


// void NN_qObs_Aided_EKF_v11::setNN(NN_v11* nn, int seq_len, int stride) {
//   nn_ = nn;
//   nn_seq_len_ = std::max(1, seq_len);
//   nn_stride_  = std::max(1, stride);
//   nn_count_   = 0;
//   nn_buf_.clear();
// }
void NN_qObs_Aided_EKF_v11::setNN(NN_v11* nn, int seq_len, int stride)
{
  nn_ = nn;
  nn_seq_len_ = std::max(1, seq_len);
  nn_stride_  = std::max(1, stride);
  nn_count_   = 0;
  nn_buf_.clear();

  nn_warmed_ = false;

  // Policy derived from NN model folder name (…_hem vs …_sign)
  nn_use_sign_continuity_ =
      (nn_ && nn_->quatInputPolicy() == NN_v11::QuatInputPolicy::SignContinuity);

  // Reset continuity state (only used when nn_use_sign_continuity_ == true)
  nn_has_prev_q_nn_ = false;
  nn_prev_q_nn_.w = 1.0; nn_prev_q_nn_.x = 0.0; nn_prev_q_nn_.y = 0.0; nn_prev_q_nn_.z = 0.0;

#ifdef EKF_DEBUG
  dbgHeader("setNN");
  std::cerr << "seq_len="<<nn_seq_len_<<" stride="<<nn_stride_
            << " quat_policy="<<(nn_use_sign_continuity_ ? "SignContinuity" : "Hemisphere")<<"\n";
#endif
}


void NN_qObs_Aided_EKF_v11::nn_prune_() {
  const int MAX_KEEP = std::max(nn_seq_len_, 4);
  while ((int)nn_buf_.size() > MAX_KEEP) nn_buf_.pop_front();
}

// void NN_qObs_Aided_EKF_v11::feedNN(const Vec3& accel_b,
//                                   const Quat& q_nb,
//                                   double tau_x,
//                                   double tau_y,
//                                   double tau_n)
// {
//   if (!nn_ || nn_seq_len_ <= 0) return;

//   // 1) Normalize input quaternion (defensive)
//   Quat q = q_nb;
//   {
//     const double n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
//     if (n2 > 0.0) {
//       const double invn = 1.0 / std::sqrt(n2);
//       q.w *= invn; q.x *= invn; q.y *= invn; q.z *= invn;
//     } else {
//       // fallback to identity if someone passes a zero quaternion
//       q.w = 1.0; q.x = q.y = q.z = 0.0;
//     }
//   }

//   // 2) Keep NN input quaternion continuous across calls (avoid ±q flip)
//   static bool has_prev_nn = false;
//   static Quat q_prev_nn; // default constructed (all zeros)
//   if (!has_prev_nn) {
//     q_prev_nn.w = 1.0; q_prev_nn.x = 0.0; q_prev_nn.y = 0.0; q_prev_nn.z = 0.0;
//     has_prev_nn = true;
//   }
//   const double dot = q.w*q_prev_nn.w + q.x*q_prev_nn.x + q.y*q_prev_nn.y + q.z*q_prev_nn.z;
//   if (dot < 0.0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
//   q_prev_nn = q;

//   // 3) Buffer one-step NN input row: [ax ay az qw qx qy qz tau_x tau_y tau_n]
//   Eigen::Matrix<double,10,1> row;
//   row << accel_b.x(), accel_b.y(), accel_b.z(),
//          q.w, q.x, q.y, q.z,
//          tau_x, tau_y, tau_n;

//   nn_buf_.push_back(row);
//   nn_prune_();

//   nn_count_++;

//   // If we do not yet have enough samples for a proper warm-start, do nothing.
//   // IMPORTANT: do NOT call infer() before we have nn_seq_len_ samples, otherwise
//   // the GRU hidden state gets initialized on too-short history.
//   if ((int)nn_buf_.size() < nn_seq_len_) {
//     nn_warmed_ = false;
//     return;
//   }

//   Vec3 v_nav_mean;   // [vE, vN, vD]
//   Mat3 Rv_nav;

//   // ------------------------------------------------------------------
//   // 1) Warm-start once: feed the last nn_seq_len_ samples sequentially
//   // ------------------------------------------------------------------
//   if (!nn_warmed_) {
//     std::vector<Eigen::Matrix<double,10,1>> window;
//     window.reserve(nn_seq_len_);

//     auto it = nn_buf_.end();
//     for (int i = 0; i < nn_seq_len_; ++i) --it;
//     for (int i = 0; i < nn_seq_len_; ++i, ++it) window.push_back(*it);

//     if (!nn_->infer(window, v_nav_mean, Rv_nav)) return;

//     nn_warmed_ = true;

//     // Only apply EKF update on stride, but we still did the warm-start state fill.
//     if ((nn_count_ % nn_stride_) == 0) {
//       (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1);
//     }
//     return; // avoid double-feeding the last sample this tick
//   }

//   // ------------------------------------------------------------------
//   // 2) Steady-state: step the GRU every tick with the newest sample only
//   // ------------------------------------------------------------------
//   {
//     std::vector<Eigen::Matrix<double,10,1>> one;
//     one.reserve(1);
//     one.push_back(row);

//     if (!nn_->infer(one, v_nav_mean, Rv_nav)) return;
//   }

//   // Apply EKF update only on stride
//   if ((nn_count_ % nn_stride_) != 0) return;

//   (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1);
// }

void NN_qObs_Aided_EKF_v11::feedNN(const Vec3& accel_b,
                                  const Quat& q_nb,
                                  double tau_x,
                                  double tau_y,
                                  double tau_n)
{
  if (!nn_ || nn_seq_len_ <= 0) return;

  // 1) Normalize input quaternion (defensive)
  Quat q = q_nb;
  {
    const double n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    if (n2 > 0.0) {
      const double invn = 1.0 / std::sqrt(n2);
      q.w *= invn; q.x *= invn; q.y *= invn; q.z *= invn;
    } else {
      q.w = 1.0; q.x = q.y = q.z = 0.0;
    }
  }

  // 2) Match the NN training-time quaternion representation:
  //    - Hemisphere: stateless canonicalization (qw >= 0 with tie-break)
  //    - SignContinuity: flip if dot(prev, curr) < 0 to avoid ±q jumps
  if (nn_use_sign_continuity_) {
    if (!nn_has_prev_q_nn_) {
      nn_prev_q_nn_ = q;
      nn_has_prev_q_nn_ = true;
    } else {
      const double dot = q.w*nn_prev_q_nn_.w + q.x*nn_prev_q_nn_.x + q.y*nn_prev_q_nn_.y + q.z*nn_prev_q_nn_.z;
      if (dot < 0.0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
      nn_prev_q_nn_ = q;
    }
  } else {
    // Hemisphere canonicalization (same rule as canonicalizeQuat)
    q = canonicalizeQuat(q);
  }

  // 3) Buffer one-step NN input row: [ax ay az qw qx qy qz tau_x tau_y tau_n]
  Eigen::Matrix<double,10,1> row;
  row << accel_b.x(), accel_b.y(), accel_b.z(),
         q.w, q.x, q.y, q.z,
         tau_x, tau_y, tau_n;

  nn_buf_.push_back(row);
  nn_prune_();

  nn_count_++;

  // Do not call infer() before we have nn_seq_len_ samples (proper warm-start).
  if ((int)nn_buf_.size() < nn_seq_len_) {
    nn_warmed_ = false;
    return;
  }

  Vec3 v_nav_mean;   // [vE, vN, vD]
  Mat3 Rv_nav;

  // 1) Warm-start once: feed the last nn_seq_len_ samples sequentially
  if (!nn_warmed_) {
    std::vector<Eigen::Matrix<double,10,1>> window;
    window.reserve(nn_seq_len_);

    auto it = nn_buf_.end();
    for (int i = 0; i < nn_seq_len_; ++i) --it;
    for (int i = 0; i < nn_seq_len_; ++i, ++it) window.push_back(*it);

    if (!nn_->infer(window, v_nav_mean, Rv_nav)) return;

    nn_warmed_ = true;

    // Apply EKF update only on stride
    if ((nn_count_ % nn_stride_) == 0) {
      (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1e-4);
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

  if ((nn_count_ % nn_stride_) != 0) return;

  (void)updateNNVelNav(v_nav_mean, Rv_nav, /*w=*/1e-4);
}

bool NN_qObs_Aided_EKF_v11::updateNNVelNav(const Vec3& z_v_nav, const Mat3& Rv, double w)
{
  // residual: z - h(x) with h(x)=v^n
  const Vec3 res = z_v_nav - x_.v;

  // H = [0_{3x3}  I_{3x3}  0_{3x3}]
  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3).setIdentity();

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rv / wt;

  // Innovation covariance
  const Mat3 S = (H * P_ * H.transpose()) + R;

  // Optional NIS gate
  if (cfg_.chi2_gate_vec3 > 0.0) {
    const double nis = res.transpose() * S.inverse() * res;
    if (nis >= cfg_.chi2_gate_vec3) return false;
  }

  const Eigen::Matrix<double,9,3> K  = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

#ifdef EKF_DEBUG
  if ((++g_dbg_counter % DBG_PRINT_EVERY) == 0) {
    const double yaw = yaw_from_Rnb_END(R_nb_);
    dbgHeader("updNNv");
    dbgPrintVec("res_vn", res); std::cerr << "  ";
    std::cerr << "NIS=" << (res.transpose()*S.inverse()*res) << "  yaw="<<rad2deg(yaw)<<"deg";
    if (near180(yaw)) std::cerr << " **NEAR 180°**";
    std::cerr << "  d|v|="<< dx.segment<3>(3).norm() << "\n";
  }
#endif

  // Masked update: only velocity
  Eigen::Matrix<double,9,9> J = Eigen::Matrix<double,9,9>::Zero();
  J.block<3,3>(3,3).setIdentity();
  const Eigen::Matrix<double,9,1> Jdx = J * dx;
  x_.v   += Jdx.segment<3>(3);

  // Joseph form (masked)
  const Eigen::Matrix<double,9,9> I = Eigen::Matrix<double,9,9>::Identity();
  const Eigen::Matrix<double,9,9> IKH = I - (J * K * H);
  P_ = IKH * P_ * IKH.transpose() + (J * K) * R * (J * K).transpose();

  return true;
}

const State9_v11& NN_qObs_Aided_EKF_v11::state() const { return x_; }
const Mat99&     NN_qObs_Aided_EKF_v11::cov()   const { return P_; }

void NN_qObs_Aided_EKF_v11::setState(const State9_v11& x, const Mat99& P){ x_ = x; P_ = P; }

void NN_qObs_Aided_EKF_v11::propagate(const Vec3& omega_b_meas,
                                      const Vec3& accel_b_meas,
                                      double dt)
{
  if (!(dt > 0.0)) return;
  last_omega_b_meas_ = omega_b_meas;

  // ---------- Specific force in BODY ----------
#ifdef ENABLE_IMU_LEVER_ARM
  static Vec3 omega_prev = omega_b_meas;
  const Vec3 alpha_b = (omega_b_meas - omega_prev) / std::max(1e-6, dt);
  omega_prev = omega_b_meas;

  const Vec3 r_ib_b(0.0, 0.0, 0.0); // update with real lever arm if used
  const Vec3 a_corr = accel_b_meas
                    - (alpha_b.cross(r_ib_b) + omega_b_meas.cross(omega_b_meas.cross(r_ib_b)));
  const Vec3 f_b = a_corr - x_.b_a;            // specific force minus bias
#else
  const Vec3 f_b = accel_b_meas - x_.b_a;      // assume "specific force" from IMU
#endif

  // ---------- Acceleration in END (Down-positive gravity) ----------
  Vec3 a_n = R_nb_ * f_b + g_n_;               // g_n_ ~ [0,0,+9.81]

  // Optional heave spring–damper (Down axis)
  //a_n.z() += -k_z_ * (x_.p.z() - z0_) - c_z_ * x_.v.z();

  // ---------- Nominal propagation ----------
  const Vec3 dp = x_.v * dt + 0.5 * a_n * dt * dt;
  const Vec3 dv = a_n * dt;

  x_.p += dp;
  x_.v += dv;

  // ---------- Linearized error model: δx=[δp, δv, δb_a] ----------
  Mat99 A = Mat99::Zero();
  A.block<3,3>(0,3).setIdentity(); // δṗ = δv
  A.block<3,3>(3,6) = -R_nb_;      // ∂(v̇)/∂b_a = -R_nb_ (since use accel - b_a)
  //A(5,2) += -k_z_;
  //A(5,5) += -c_z_;
  if (cfg_.tau_ba > 0.0) {
    x_.b_a += (-(1.0 / cfg_.tau_ba) * x_.b_a) * dt;
    A.block<3,3>(6,6) = -(1.0 / cfg_.tau_ba) * Mat3::Identity();
  }

  // 2nd-order Ad
  const Mat99 I9 = Mat99::Identity();
  const Mat99 Ad = I9 + A * dt + 0.5 * (A * A) * (dt * dt);

  // Process noise (discrete integrated accel white-noise)
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

    // Check A(3,6) sign consistency
    Mat3 A36 = A.block<3,3>(3,6);
    dbgHeader("prop");
    std::cerr << "A(3,6) vs -R_nb_ max|Δ|=" 
              << (A36 + R_nb_).cwiseAbs().maxCoeff()
              << "  (expect ~0)\n";
  }
#endif
}

bool NN_qObs_Aided_EKF_v11::updateGnssPos(const Vec3& z_nav_ant, const Mat3& Rpos, const Vec3& r_body, double w){
  // predicted antenna position
#if defined(EKF_DEBUG) && defined(DBG_DISABLE_LEVER_ARM)
  const Vec3 h  = x_.p;              // lever arm disabled for comparison
#else
  const Vec3 h  = x_.p + R_nb_ * r_body;
#endif
  const Vec3 res = z_nav_ant - h;    // residual

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,0) = Mat3::Identity();  // d/dp

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;

  const Mat3 R = Rpos / wt;
  Eigen::Matrix3d S = (H * P_ * H.transpose()) + R;
  Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.ldlt().solve(Eigen::Matrix3d::Identity());
  const Eigen::Matrix<double,9,1> dx = K * res;

#ifdef EKF_DEBUG
  const double nis = (res.transpose() * S.inverse() * res);
  const Vec3 dv = dx.segment<3>(3);
  const Vec3 dp = dx.segment<3>(0);
  const double yaw = yaw_from_Rnb_END(R_nb_);
  dbgHeader("updPOS");
  dbgPrintVec("res_pn", res); std::cerr << "  NIS="<<nis
    << "  d|v|="<< dv.norm() << "  d|p|="<< dp.norm()
    << "  yaw="<<rad2deg(yaw)<<"deg";
  if (near180(yaw)) std::cerr << " **NEAR 180°**";
  std::cerr << "\n";
#endif

  // Inject
  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  // Joseph
  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

bool NN_qObs_Aided_EKF_v11::updateGnssVel(const Vec3& z_v_nav_ant, const Mat3& Rvel, const Vec3& r_body,
                                          const Vec3& omega_b_meas, double w){
#if defined(EKF_DEBUG) && defined(DBG_DISABLE_LEVER_ARM)
  const Vec3 la = Vec3::Zero();
#else
  const Vec3 la = R_nb_ * (omega_b_meas.cross(r_body));
#endif
  const Vec3 v_ant_pred = x_.v + la;
  const Vec3 res = z_v_nav_ant - v_ant_pred;

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = Mat3::Identity(); // d/dv

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rvel / wt;
  const Mat3 S = (H * P_ * H.transpose()) + R;
  const Eigen::Matrix<double,9,3> K  = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

#ifdef EKF_DEBUG
  const double nis = (res.transpose() * S.inverse() * res);
  const double yaw = yaw_from_Rnb_END(R_nb_);
  dbgHeader("updVEL");
  dbgPrintVec("res_vn", res); std::cerr << "  NIS="<<nis
    << "  |la|="<< la.norm()
    << "  d|v|="<< dx.segment<3>(3).norm()
    << "  yaw="<<rad2deg(yaw)<<"deg";
  if (near180(yaw)) std::cerr << " **NEAR 180°**";
  std::cerr << "\n";
#endif

  x_.v   += dx.segment<3>(3);
  x_.p   += dx.segment<3>(0);
  x_.b_a += dx.segment<3>(6);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

bool NN_qObs_Aided_EKF_v11::updateNNBodyVel(const Vec3& z_v_b, const Mat3& Rvb, double w){
  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b_pred = R_bn * x_.v;   // predicted body velocity at CG
  const Vec3 res = z_v_b - v_b_pred;   // residual in BODY

  Eigen::Matrix<double,3,9> H = Eigen::Matrix<double,3,9>::Zero();
  H.block<3,3>(0,3) = R_bn;            // d(R_bn v_n)/d v_n = R_bn

  const double wt = (std::isfinite(w) && w>0.0) ? w : 1.0;
  const Mat3 R = Rvb / wt;
  const Mat3 S = (H * P_ * H.transpose()) + R;
  const Eigen::Matrix<double,9,3> K  = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * res;

#ifdef EKF_DEBUG
  const double nis = (res.transpose() * S.inverse() * res);
  const double yaw = yaw_from_Rnb_END(R_nb_);
  dbgHeader("updNNb");
  dbgPrintVec("res_vb", res); std::cerr << "  NIS="<<nis
    << "  d|v|="<< dx.segment<3>(3).norm()
    << "  yaw="<<rad2deg(yaw)<<"deg";
  if (near180(yaw)) std::cerr << " **NEAR 180°**";
  std::cerr << "\n";
#endif

  // Inject masked (velocity only) for NN body vel
  Eigen::Matrix<double,9,9> J = Eigen::Matrix<double,9,9>::Zero();
  J.block<3,3>(3,3).setIdentity();
  const Eigen::Matrix<double,9,1> Jdx = J * dx;
  x_.v   += Jdx.segment<3>(3);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - (J * K * H);
  P_ = IKH * P_ * IKH.transpose() + (J * K) * R * (J * K).transpose();

  return true;
}

Eigen::Matrix<double,9,1> NN_qObs_Aided_EKF_v11::getState9() const {
  Eigen::Matrix<double,9,1> x9;
  x9.segment<3>(0) = x_.p;
  x9.segment<3>(3) = x_.v;
  x9.segment<3>(6) = x_.b_a;
  return x9;
}

// Optional convenience: 12-state legacy vector [u v w p q r x y z phi theta psi]^T
Eigen::Matrix<double,12,1> NN_qObs_Aided_EKF_v11::getState12(const Vec3& b_gyro_hat) const {
  Eigen::Matrix<double,12,1> x12;

  const Mat3 R_bn = R_nb_.transpose();
  const Vec3 v_b  = R_bn * x_.v; // [u v w]

  const Vec3 omega_b = last_omega_b_meas_ - b_gyro_hat;

  auto clamp1 = [](double s){ return std::max(-1.0, std::min(1.0, s)); };
  const double r20 = R_nb_(2,0);
  const double r21 = R_nb_(2,1);
  const double r22 = R_nb_(2,2);
  const double r10 = R_nb_(1,0);
  const double r00 = R_nb_(0,0);

  const double theta = -std::asin(clamp1(r20));  // pitch
  const double phi   = std::atan2(r21, r22);     // roll
  const double psi   = std::atan2(r00, r10);     // heading from North→East

  x12 << v_b.x(), v_b.y(), v_b.z(),
          omega_b.x(), omega_b.y(), omega_b.z(),
          x_.p.x(), x_.p.y(), x_.p.z(),
          phi, theta, psi;
  return x12;
}

bool NN_qObs_Aided_EKF_v11::kalmanUpdate(const Vec3& r, const Eigen::Matrix<double,3,9>& H,
                                         const Mat3& R, double chi2_gate){
  const Mat3 S = (H * P_ * H.transpose()) + R;

  if (chi2_gate > 0.0) {
    const double nis = r.transpose() * S.inverse() * r;
#ifdef EKF_DEBUG
    dbgHeader("kalUpd"); std::cerr << "NIS="<<nis<<"\n";
#endif
    if (nis >= chi2_gate) return false;
  }

  const Eigen::Matrix<double,9,3> K = P_ * H.transpose() * S.inverse();
  const Eigen::Matrix<double,9,1> dx = K * r;

#ifdef EKF_DEBUG
  const double yaw = yaw_from_Rnb_END(R_nb_);
  dbgHeader("kalUpd");
  dbgPrintVec("r", r); std::cerr << "  d|v|="<< dx.segment<3>(3).norm()
                                 << "  yaw="<<rad2deg(yaw)<<"deg";
  if (near180(yaw)) std::cerr << " **NEAR 180°**";
  std::cerr << "\n";
#endif

  x_.p   += dx.segment<3>(0);
  x_.v   += dx.segment<3>(3);
  x_.b_a += dx.segment<3>(6);

  const Mat99 I = Mat99::Identity();
  const Mat99 IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();

  return true;
}

} // namespace nnqekf_v11

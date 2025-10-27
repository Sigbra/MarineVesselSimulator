#include "Observers/nn_EKF_v2.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

// Forward declarations (avoid bringing CasADi into this TU)
Eigen::Matrix3d Rzyx(double phi, double theta, double psi);
Eigen::Matrix3d Tzyx(double phi, double theta);
double ssa(double a);

// ===================================================================================
//                               NN Interface (impl)
// ===================================================================================

struct NN_INTERFACE_V2::Impl {
  // Torch
  torch::jit::Module module;
  bool use_cuda = false;
  bool is_stateless = false;     // expects (x, h_prev) and returns (..., h_next)
  bool is_ensemble = false;      // stateless with Hnext [M,L,B,H]
  bool second_is_logvar = true;  // single model: true (PHYSICAL log-variance); ensemble: false (PHYSICAL variance diag)

  mutable torch::Tensor h_;      // [L,B,H] or [M,L,B,H]

  // Normalization stats (kept for backward-compat; UNUSED by v2 runtime)
  Eigen::Array<double,6,1> in_mu, in_std;
  Eigen::Array<double,6,1> out_mu, out_std;

  Impl() {
    in_mu.setZero();  in_std.setOnes();
    out_mu.setZero(); out_std.setOnes();
  }
};

static bool decide_use_cuda(bool requested) {
#ifdef TORCH_CUDA_AVAILABLE
  if (!requested) return false;
  try {
    auto t = torch::rand({1}).to(torch::kCUDA);
    (void)t;
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)requested;
  return false;
#endif
}

static bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

static std::vector<std::string> build_candidate_paths(const std::string& requested) {
  std::vector<std::string> cands;
  if (!requested.empty()) cands.emplace_back(requested);
#ifdef MVS_PROJECT_ROOT
  if (!requested.empty() && requested.front() != '/') {
    cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/" + requested);
  }
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v2/ensemble_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v2/model_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v2/model.pt");
#endif
  // dedupe
  std::vector<std::string> uniq;
  for (const auto& s : cands) {
    bool seen=false; for (const auto& u : uniq) if (u==s){seen=true;break;}
    if (!seen) uniq.push_back(s);
  }
  return uniq;
}

NN_INTERFACE_V2::NN_INTERFACE_V2() : p_(new Impl) {}
NN_INTERFACE_V2::~NN_INTERFACE_V2() { delete p_; }

void NN_INTERFACE_V2::load(const std::string& model_path, bool use_cuda_request) {
  p_->use_cuda = decide_use_cuda(use_cuda_request);
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  auto cands = build_candidate_paths(model_path);
  std::vector<std::string> tried;
  std::exception_ptr last_exc;

  for (const auto& cand : cands) {
    if (!file_exists(cand)) { tried.push_back(cand + " (missing)"); continue; }
    try {
      auto mod = torch::jit::load(cand, dev);
      mod.eval();

      bool is_stateless=false, is_ensemble=false, second_is_logvar=true;
      try {
        auto x = torch::zeros({1,6}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        torch::IValue none; // None
        auto out = mod.forward({x, none});
        if (out.isTuple() && out.toTuple()->elements().size()==3) {
          is_stateless = true;
          const auto hnext = out.toTuple()->elements()[2].toTensor();
          is_ensemble = (hnext.dim()==4); // [M,L,B,H]
          second_is_logvar = !is_ensemble; // ensemble assumed PHYSICAL variance diag
        }
      } catch (...) {
        // try stateful
      }
      if (!is_stateless) {
        auto x = torch::zeros({1,6}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        auto any = mod.forward({x});
        if (!any.isTuple() || any.toTuple()->elements().size()<1) {
          tried.push_back(cand + " (bad signature)"); continue;
        }
        second_is_logvar = (any.toTuple()->elements().size()>=2);
      }

      // success
      p_->module = std::move(mod);
      p_->is_stateless = is_stateless;
      p_->is_ensemble  = is_ensemble;
      p_->second_is_logvar = second_is_logvar;
      p_->h_ = torch::Tensor();

      std::cout << "[NN_INTERFACE_V2] Loaded '" << cand << "' on "
                << (p_->use_cuda ? "CUDA" : "CPU")
                << " | sig=" << (p_->is_stateless ? (p_->is_ensemble ? "ensemble(x,H)->(mu_phys,cov_phys,H)"
                                                                     : "stateless(x,h)->(mu_phys,logvar_phys,h)")
                                                  : (p_->second_is_logvar ? "stateful(x)->(mu_phys,logvar_phys)"
                                                                          : "stateful(x)->(mu_phys)"))
                << std::endl;
      return;
    } catch (...) {
      last_exc = std::current_exception();
      tried.push_back(cand + " (load failed)");
    }
  }

  std::cerr << "[NN_INTERFACE_V2] ERROR: could not load a TorchScript model.\nTried:\n";
  for (const auto& s : tried) std::cerr << "  - " << s << "\n";
  if (last_exc) {
    try { std::rethrow_exception(last_exc); }
    catch (const std::exception& e) { std::cerr << "Last error: " << e.what() << "\n"; }
  }
  throw std::runtime_error("NN_INTERFACE_V2: no valid model found.");
}

void NN_INTERFACE_V2::set_normalization(const std::array<double,6>& in_mu,
                                        const std::array<double,6>& in_std,
                                        const std::array<double,6>& out_mu,
                                        const std::array<double,6>& out_std) {
  // Kept for backward-compat; v2 exports already handle normalization internally.
  for (int i=0;i<6;++i) {
    p_->in_mu(i)=in_mu[i];
    p_->in_std(i)=std::max(1e-12, in_std[i]);
    p_->out_mu(i)=out_mu[i];
    p_->out_std(i)=std::max(1e-12, out_std[i]);
  }
}

static bool parse_stats_json(const std::string& text,
                             std::array<double,6>& in_mu,
                             std::array<double,6>& in_std,
                             std::array<double,6>& out_mu,
                             std::array<double,6>& out_std)
{
  auto grab = [&](const std::string& key)->std::optional<double>{
    std::regex rgx("\"" + key + "\"\\s*:\\s*([-+eE0-9\\.]+)");
    std::smatch m; if (std::regex_search(text, m, rgx)) {
      try { return std::stod(m[1]); } catch (...) { return std::nullopt; }
    }
    return std::nullopt;
  };

  const char* in_keys [6] = {"ax","ay","az","wx","wy","wz"};
  const char* out_keys[6] = {"u","v","w","p","q","r"};

  for (int i=0;i<6;++i) {
    auto mi = grab(std::string("\"inputs\"") + ".*\"mean\".*\"" + in_keys[i] + "\"");
    auto si = grab(std::string("\"inputs\"") + ".*\"std\".*\""  + in_keys[i] + "\"");
    auto mo = grab(std::string("\"targets\"")+ ".*\"mean\".*\"" + out_keys[i] + "\"");
    auto so = grab(std::string("\"targets\"")+ ".*\"std\".*\""  + out_keys[i] + "\"");
    if (!mi || !si || !mo || !so) return false;
    in_mu [i] = *mi;
    in_std[i] = *si;
    out_mu[i] = *mo;
    out_std[i] = *so;
  }
  return true;
}

bool NN_INTERFACE_V2::load_normalization_json(const std::string& json_path) {
  std::ifstream f(json_path);
  if (!f.good()) return false;
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  std::array<double,6> in_mu, in_std, out_mu, out_std;
  if (!parse_stats_json(s, in_mu, in_std, out_mu, out_std)) return false;
  set_normalization(in_mu, in_std, out_mu, out_std);
  std::cout << "[NN_INTERFACE_V2] Loaded normalization from " << json_path
            << " (note: v2 TS handles normalization internally)" << std::endl;
  return true;
}

void NN_INTERFACE_V2::reset() { p_->h_ = torch::Tensor(); }

std::pair<std::array<double,6>, Eigen::Matrix<double,6,6>>
NN_INTERFACE_V2::predict_uvwpqr(const std::array<double,6>& imu) const {
  torch::NoGradGuard ng;
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  // Pass RAW IMU to TS (v2 TS normalizes internally)
  std::array<float,6> xraw{};
  for (int i=0;i<6;++i) xraw[i] = static_cast<float>(imu[i]);

  torch::Tensor x = torch::from_blob((void*)xraw.data(), {1,6}, torch::TensorOptions().dtype(torch::kFloat32));
  if (p_->use_cuda) x = x.to(torch::kCUDA);

  torch::Tensor mu_ts, second, hnext;
  if (p_->is_stateless) {
    torch::IValue h_in = p_->h_.defined() ? torch::IValue(p_->h_) : torch::IValue();
    auto any = p_->module.forward({x, h_in});
    auto tup = any.toTuple();
    mu_ts = tup->elements()[0].toTensor(); // [1,6] PHYSICAL μ
    second = tup->elements()[1].toTensor(); // [1,6] PHYSICAL logvar (single) OR PHYSICAL var (ensemble)
    p_->h_ = tup->elements()[2].toTensor(); // [L,1,H] or [M,L,1,H]
  } else {
    auto any = p_->module.forward({x});
    auto tup = any.toTuple();
    mu_ts = tup->elements()[0].toTensor(); // [1,6] PHYSICAL μ
    if (tup->elements().size() >= 2) {
      second = tup->elements()[1].toTensor(); // PHYSICAL logvar
    } else {
      second = torch::full_like(mu_ts, std::log(1e-6f)); // fallback: PHYSICAL logvar
    }
  }

  // Outputs are already PHYSICAL; build covariance
  std::array<double,6> mu{};
  Eigen::Matrix<double,6,6> R; R.setZero();

  auto mu_d = mu_ts.to(torch::kFloat64).squeeze(0).contiguous();
  auto sec_d= second.to(torch::kFloat64).squeeze(0).contiguous();

  for (int i=0;i<6;++i) {
    mu[i] = mu_d[i].item<double>();

    double var_phys;
    if (p_->second_is_logvar) {
      // single-model path: PHYSICAL log-variance
      var_phys = std::exp(sec_d[i].item<double>());
    } else {
      // ensemble path: PHYSICAL variance (predictive covariance diag)
      var_phys = sec_d[i].item<double>();
    }
    if (var_phys < 1e-12) var_phys = 1e-12;
    R(i,i) = var_phys;
  }

  return {mu, R};
}

// ===================================================================================
//                                       EKF v2
// ===================================================================================

NN_EKF_V2::NN_EKF_V2()
: h_(0.01), g_(9.80665),
  bias_adapt_window_s_(2.0),
  steps_since_abs_aid_(1e9) // large → initially "no aiding recently"
{
  x_.setZero();
  P_.setIdentity();
  Qd_.setZero();

  // Priors (tune): diag blocks
  P_.block<3,3>(0,0).setIdentity();   P_.block<3,3>(0,0) *= 0.5*0.5;                     // u,v,w
  P_.block<3,3>(3,3).setIdentity();   P_.block<3,3>(3,3) *= std::pow(5.0*M_PI/180.0,2);   // p,q,r
  P_.block<3,3>(6,6).setIdentity();   P_.block<3,3>(6,6) *= 5.0*5.0;                      // pos
  P_.block<3,3>(9,9).setIdentity();   P_.block<3,3>(9,9) *= std::pow(5.0*M_PI/180.0,2);   // angles
  P_.block<3,3>(12,12).setIdentity(); P_.block<3,3>(12,12) *= 0.2*0.2;                    // b_u,v,w
  P_.block<3,3>(15,15).setIdentity(); P_.block<3,3>(15,15) *= std::pow(0.5*M_PI/180.0,2); // b_p,q,r

  // Process noise (tune). Keep biases almost constant (tiny walk).
  Qd_.block<3,3>(0,0).setIdentity();   Qd_.block<3,3>(0,0)   *= 1e-3;
  Qd_.block<3,3>(3,3).setIdentity();   Qd_.block<3,3>(3,3)   *= 1e-4;
  Qd_.block<3,3>(6,6).setIdentity();   Qd_.block<3,3>(6,6)   *= 1e-6;
  Qd_.block<3,3>(9,9).setIdentity();   Qd_.block<3,3>(9,9)   *= 1e-5;
  Qd_.block<3,3>(12,12).setIdentity(); Qd_.block<3,3>(12,12) *= 1e-9; // bias lin
  Qd_.block<3,3>(15,15).setIdentity(); Qd_.block<3,3>(15,15) *= 1e-8; // bias ang
}

void NN_EKF_V2::setDt(double h){ h_ = h; }
void NN_EKF_V2::setGravity(double g){ g_ = g; (void)g_; }
void NN_EKF_V2::setState(const VecN& x0){ x_ = x0; }
void NN_EKF_V2::setCovariance(const MatN& P0){ P_ = P0; }
void NN_EKF_V2::setProcessNoise(const MatN& Qd){ Qd_ = Qd; }
const NN_EKF_V2::VecN&  NN_EKF_V2::state() const { return x_; }
const NN_EKF_V2::MatN&  NN_EKF_V2::covariance() const { return P_; }

void NN_EKF_V2::setBiasAdaptationWindow(double seconds){ bias_adapt_window_s_ = std::max(0.0, seconds); }
double NN_EKF_V2::biasAdaptationWindow() const { return bias_adapt_window_s_; }

void NN_EKF_V2::initNN(const std::string& model_path,
                       bool use_cuda_request,
                       const std::string& norm_json)
{
  if (!nn_) nn_ = std::make_unique<NN_INTERFACE_V2>();
  nn_->load(model_path, use_cuda_request);

  // v2 TS already embeds/uses normalization internally; JSON load is optional/backward-compat.
  if (!norm_json.empty()) {
    if (!nn_->load_normalization_json(norm_json)) {
      std::cerr << "[NN_EKF_V2] WARNING: failed to load normalization JSON '"
                << norm_json << "'. Using identity normalization.\n";
    }
  }
  nn_->reset();
}

void NN_EKF_V2::setNNNormalization(const std::array<double,6>& in_mu,
                                   const std::array<double,6>& in_std,
                                   const std::array<double,6>& out_mu,
                                   const std::array<double,6>& out_std)
{
  if (!nn_) nn_ = std::make_unique<NN_INTERFACE_V2>();
  nn_->set_normalization(in_mu, in_std, out_mu, out_std);
}

void NN_EKF_V2::resetNN()   { if (nn_) nn_->reset(); }
bool NN_EKF_V2::hasNN() const { return static_cast<bool>(nn_); }

// Dynamics (unbiased velocities drive kinematics; biases are measurement-only)
NN_EKF_V2::VecN NN_EKF_V2::f(const VecN& xs) const {
  VecN xd; xd.setZero();

  const double u=xs(0), v=xs(1), w=xs(2);
  const double p=xs(3), q=xs(4), r=xs(5);
  const double phi=xs(9), th=xs(10), psi=xs(11);

  // u,v,w and p,q,r random walks (derivative ~ 0). Bias states too (near-zero walk via Qd).
  xd.segment<3>(0).setZero();
  xd.segment<3>(3).setZero();
  xd.segment<3>(12).setZero();
  xd.segment<3>(15).setZero();

  // Position & attitude kinematics (use UNBIASED velocities/rates)
  const Eigen::Matrix3d Rbn = Rzyx(phi, th, psi);
  xd.segment<3>(6) = Rbn * Eigen::Vector3d(u, v, w);

  const Eigen::Matrix3d T = Tzyx(phi, th);
  xd.segment<3>(9) = T * Eigen::Vector3d(p, q, r);

  return xd;
}

NN_EKF_V2::MatN NN_EKF_V2::A_numeric(const VecN& xs, double eps) const {
  MatN A = MatN::Zero();
  const VecN f0 = f(xs);
  for (int i=0;i<NX;++i){
    VecN xh = xs; xh(i) += eps;
    A.col(i) = (f(xh) - f0) / eps;
  }
  return A;
}

bool NN_EKF_V2::allowBiasAdaptation() const {
  const double t_since_abs = steps_since_abs_aid_ * h_;
  return (t_since_abs <= bias_adapt_window_s_);
}

void NN_EKF_V2::predict() {
  const VecN xd = f(x_);
  x_ += h_ * xd;

  const MatN A  = A_numeric(x_);
  const MatN Fd = MatN::Identity() + h_ * A;
  P_ = Fd * P_ * Fd.transpose() + Qd_;

  // advance bias gating timer
  if (steps_since_abs_aid_ < (1L<<60)) ++steps_since_abs_aid_;
}

// One-call IMU -> NN -> EKF (NN measurement)
void NN_EKF_V2::updateFromIMU(const std::array<double,6>& imu) {
  if (!nn_) throw std::runtime_error("NN_EKF_V2::updateFromIMU(): NN not initialized.");
  const auto [mu6, Rnn] = nn_->predict_uvwpqr(imu);

  Eigen::Matrix<double,6,1> mu;
  for (int k=0;k<6;++k) mu(k) = mu6[k];

  updateNN(mu, Rnn);
}

// NN update (μ,R) with bias gating
void NN_EKF_V2::updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                         const Eigen::Matrix<double,6,6>& R_uvwpr)
{
  Eigen::Matrix<double,6,NX> H; H.setZero();
  // Sensitivity to unbiased states
  H(0,0)=1; H(1,1)=1; H(2,2)=1;
  H(3,3)=1; H(4,4)=1; H(5,5)=1;
  // Sensitivity to bias states — gated
  if (allowBiasAdaptation()) {
    //H(0,12)=1; H(1,13)=1; H(2,14)=1;
    H(3,15)=1; H(4,16)=1; H(5,17)=1;
  } // else: columns for biases remain zero → biases frozen

  // Predicted measurement (uses unbiased + bias)
  Eigen::Matrix<double,6,1> yhat;
  yhat << x_(0)+x_(12), x_(1)+x_(13), x_(2)+x_(14),
          x_(3)+x_(15), x_(4)+x_(16), x_(5)+x_(17);

  const Eigen::Matrix<double,6,1> innov = mu_uvwpr - yhat;

  Eigen::Matrix<double,6,6> R = R_uvwpr;
  for (int i=0;i<6;++i) if (R(i,i) < 1e-12) R(i,i) = 1e-12;

  const Eigen::Matrix<double,6,6> S = H*P_*H.transpose() + R;
  const Eigen::Matrix<double,NX,6> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// Convenience diag-variance overload
void NN_EKF_V2::updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                         const Eigen::Matrix<double,6,1>& var_uvwpr)
{
  Eigen::Matrix<double,6,6> R = Eigen::Matrix<double,6,6>::Zero();
  for (int i=0;i<6;++i) R(i,i) = std::max(1e-12, var_uvwpr(i));
  updateNN(mu_uvwpr, R);
}

// GNSS & Heading (absolute aiding) — reset bias gating timer
void NN_EKF_V2::updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN)
{
  Eigen::Matrix<double,2,NX> H; H.setZero();
  H(0,6)=1.0; H(1,7)=1.0;

  Eigen::Vector2d yhat(x_(6), x_(7));
  const Eigen::Vector2d innov = pEN - yhat;

  const Eigen::Matrix2d S = H*P_*H.transpose() + R_EN;
  const Eigen::Matrix<double,NX,2> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;

  steps_since_abs_aid_ = 0; // allow bias adaptation for a while
}

void NN_EKF_V2::updatePosEND(const Eigen::Vector3d& pEND, const Eigen::Matrix3d& R_END)
{
  Eigen::Matrix<double,3,NX> H; H.setZero();
  H(0,6)=1.0; H(1,7)=1.0; H(2,8)=1.0;

  Eigen::Vector3d yhat = x_.segment<3>(6);
  const Eigen::Vector3d innov = pEND - yhat;

  const Eigen::Matrix3d S = H*P_*H.transpose() + R_END;
  const Eigen::Matrix<double,NX,3> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;

  steps_since_abs_aid_ = 0;
}

void NN_EKF_V2::updateHeading(double psi_meas, double R_psi)
{
  Eigen::Matrix<double,1,NX> H; H.setZero();
  H(0,11)=1.0;

  const double innov = ssa(psi_meas - x_(11));
  const double S     = (H*P_*H.transpose())(0,0) + R_psi;
  const Eigen::Matrix<double,NX,1> K = P_*H.transpose() / S;

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;

  steps_since_abs_aid_ = 0;
}

Eigen::VectorXd NN_EKF_V2::getState12() const
{
  Eigen::VectorXd z(12);
  z(0)=x_(0); z(1)=x_(1); z(2)=x_(2);
  z(3)=x_(3); z(4)=x_(4); z(5)=x_(5);
  z(6)=x_(6); z(7)=x_(7); z(8)=x_(8);
  z(9)=x_(9); z(10)=x_(10); z(11)=x_(11);
  return z;
}

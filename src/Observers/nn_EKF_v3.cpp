#include "Observers/nn_EKF_v3.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

// Forward declarations to avoid including CasADi in this TU
Eigen::Matrix3d Rzyx(double phi, double theta, double psi);
Eigen::Matrix3d Tzyx(double phi, double theta);
double ssa(double a);

// ===================================================================================
//                               NN Interface (impl)
// ===================================================================================

struct NN_INTERFACE_V3::Impl {
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

NN_INTERFACE_V3::NN_INTERFACE_V3() : p_(new Impl) {}
NN_INTERFACE_V3::~NN_INTERFACE_V3() { delete p_; }

void NN_INTERFACE_V3::load(const std::string& model_path, bool use_cuda_request) {
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

      std::cout << "[NN_INTERFACE_V3] Loaded '" << cand << "' on "
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

  std::cerr << "[NN_INTERFACE_V3] ERROR: could not load a TorchScript model.\nTried:\n";
  for (const auto& s : tried) std::cerr << "  - " << s << "\n";
  if (last_exc) {
    try { std::rethrow_exception(last_exc); }
    catch (const std::exception& e) { std::cerr << "Last error: " << e.what() << "\n"; }
  }
  throw std::runtime_error("NN_INTERFACE_V3: no valid model found.");
}

void NN_INTERFACE_V3::set_normalization(const std::array<double,6>& in_mu,
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

bool NN_INTERFACE_V3::load_normalization_json(const std::string& json_path) {
  std::ifstream f(json_path);
  if (!f.good()) return false;
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  std::array<double,6> in_mu, in_std, out_mu, out_std;
  if (!parse_stats_json(s, in_mu, in_std, out_mu, out_std)) return false;
  set_normalization(in_mu, in_std, out_mu, out_std);
  std::cout << "[NN_INTERFACE_V3] Loaded normalization from " << json_path
            << " (note: v2 TS handles normalization internally)" << std::endl;
  return true;
}

void NN_INTERFACE_V3::reset() { p_->h_ = torch::Tensor(); }

std::pair<std::array<double,6>, Eigen::Matrix<double,6,6>>
NN_INTERFACE_V3::predict_uvwpqr(const std::array<double,6>& imu) const {
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
    if (p_->second_is_logvar) var_phys = std::exp(sec_d[i].item<double>());
    else                      var_phys = sec_d[i].item<double>();
    if (var_phys < 1e-12) var_phys = 1e-12;
    R(i,i) = var_phys;
  }

  return {mu, R};
}

// ===================================================================================
//                                       EKF v3
// ===================================================================================

NN_EKF_V3::NN_EKF_V3()
: h_(0.05)
{
  x_.setZero();
  P_.setIdentity();
  Qd_.setZero();

  // Simple default tuning; adjust to your liking
  // States: [u v p q r pE pN phi theta psi]
  P_.diagonal() <<
    0.5*0.5, 0.5*0.5,                             // u, v
    std::pow(5.0*M_PI/180.0,2),                   // p
    std::pow(5.0*M_PI/180.0,2),                   // q
    std::pow(5.0*M_PI/180.0,2),                   // r
    5.0*5.0, 5.0*5.0,                              // pE, pN
    std::pow(5.0*M_PI/180.0,2),                   // phi
    std::pow(5.0*M_PI/180.0,2),                   // theta
    std::pow(5.0*M_PI/180.0,2);                   // psi

  Qd_.setZero();
  Qd_.diagonal() <<
    1e-3, 1e-3,   // u, v RW
    1e-4, 1e-4, 1e-4, // p,q,r RW
    1e-6, 1e-6,  // pE, pN drift
    1e-5, 1e-5, 1e-5; // phi, theta, psi RW
}

void NN_EKF_V3::setDt(double h){ h_ = h; }
void NN_EKF_V3::setState(const VecN& x0){ x_ = x0; }
void NN_EKF_V3::setCovariance(const MatN& P0){ P_ = P0; }
void NN_EKF_V3::setProcessNoise(const MatN& Qd){ Qd_ = Qd; }
const NN_EKF_V3::VecN&  NN_EKF_V3::state() const { return x_; }
const NN_EKF_V3::MatN&  NN_EKF_V3::covariance() const { return P_; }

void NN_EKF_V3::initNN(const std::string& model_path,
                       bool use_cuda_request,
                       const std::string& norm_json)
{
  if (!nn_) nn_ = std::make_unique<NN_INTERFACE_V3>();
  nn_->load(model_path, use_cuda_request);

  if (!norm_json.empty()) {
    if (!nn_->load_normalization_json(norm_json)) {
      std::cerr << "[NN_EKF_V3] WARNING: failed to load normalization JSON '"
                << norm_json << "'. Using identity normalization.\n";
    }
  }
  nn_->reset();
}

void NN_EKF_V3::setNNNormalization(const std::array<double,6>& in_mu,
                                   const std::array<double,6>& in_std,
                                   const std::array<double,6>& out_mu,
                                   const std::array<double,6>& out_std)
{
  if (!nn_) nn_ = std::make_unique<NN_INTERFACE_V3>();
  nn_->set_normalization(in_mu, in_std, out_mu, out_std);
}

void NN_EKF_V3::resetNN()   { if (nn_) nn_->reset(); }
bool NN_EKF_V3::hasNN() const { return static_cast<bool>(nn_); }

// Dynamics (no w, no z). u,v and p,q,r are random walks; they drive pE,pN and attitude.
NN_EKF_V3::VecN NN_EKF_V3::f(const VecN& xs) const {
  VecN xd; xd.setZero();

  const double u=xs(0), v=xs(1);
  const double p=xs(2), q=xs(3), r=xs(4);
  const double phi=xs(7), th=xs(8), psi=xs(9);

  // u,v,p,q,r as random walks (derivatives ≈ 0)
  xd(0)=0.0; xd(1)=0.0; xd(2)=0.0; xd(3)=0.0; xd(4)=0.0;

  // Position kinematics: set w=0 → body vel [u v 0] to EN
  const Eigen::Matrix3d Rbn = Rzyx(phi, th, psi);
  const Eigen::Vector3d v_body(u, v, 0.0);
  const Eigen::Vector3d v_nav = Rbn * v_body;
  xd(5) = v_nav.x(); // pE_dot
  xd(6) = v_nav.y(); // pN_dot

  // Euler-angle rates
  const Eigen::Matrix3d T = Tzyx(phi, th);
  const Eigen::Vector3d euler_dot = T * Eigen::Vector3d(p,q,r);
  xd(7) = euler_dot.x();
  xd(8) = euler_dot.y();
  xd(9) = euler_dot.z();

  return xd;
}

NN_EKF_V3::MatN NN_EKF_V3::A_numeric(const VecN& xs, double eps) const {
  MatN A = MatN::Zero();
  const VecN f0 = f(xs);
  for (int i=0;i<NX;++i){
    VecN xh = xs; xh(i) += eps;
    A.col(i) = (f(xh) - f0) / eps;
  }
  return A;
}

void NN_EKF_V3::predict() {
  const VecN xd = f(x_);
  x_ += h_ * xd;

  const MatN A  = A_numeric(x_);
  const MatN Fd = MatN::Identity() + h_ * A;
  P_ = Fd * P_ * Fd.transpose() + Qd_;
}

// One-call IMU -> NN -> EKF (NN measurement)
void NN_EKF_V3::updateFromIMU(const std::array<double,6>& imu) {
  if (!nn_) throw std::runtime_error("NN_EKF_V3::updateFromIMU(): NN not initialized.");
  const auto [mu6, R6] = nn_->predict_uvwpqr(imu);

  Eigen::Matrix<double,6,1> mu;
  for (int k=0;k<6;++k) mu(k) = mu6[k];

  updateNN(mu, R6);  // updateNN will sub-select [u,v,p,q,r]
}

// ------------------------
// FIX: sub-select [u,v,p,q,r] from NN (drop w)
// ------------------------
void NN_EKF_V3::updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                         const Eigen::Matrix<double,6,6>& R_uvwpr)
{
  // indices in NN output to KEEP (u=0, v=1, p=3, q=4, r=5)
  constexpr int keep[5] = {0,1,3,4,5};

  // Build 5×1 measurement and 5×5 covariance
  Eigen::Matrix<double,5,1> mu5;
  for (int i=0;i<5;++i) mu5(i) = mu_uvwpr(keep[i]);

  Eigen::Matrix<double,5,5> R5; R5.setZero();
  for (int i=0;i<5;++i) {
    for (int j=0;j<5;++j) {
      R5(i,j) = R_uvwpr(keep[i], keep[j]);
    }
  }
  for (int i=0;i<5;++i) if (R5(i,i) < 1e-12) R5(i,i) = 1e-12;

  // H maps to state: [u v p q r pE pN phi theta psi]
  Eigen::Matrix<double,5,NX> H; H.setZero();
  H(0,0)=1.0; // u
  H(1,1)=1.0; // v
  H(2,2)=1.0; // p
  H(3,3)=1.0; // q
  H(4,4)=1.0; // r

  // yhat = [u v p q r]^T from state
  Eigen::Matrix<double,5,1> yhat;
  yhat << x_(0), x_(1), x_(2), x_(3), x_(4);

  const Eigen::Matrix<double,5,1> innov = mu5 - yhat;
  const Eigen::Matrix<double,5,5> S = H*P_*H.transpose() + R5;
  const Eigen::Matrix<double,NX,5> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// Diagonal-variance overload (also sub-select)
void NN_EKF_V3::updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                         const Eigen::Matrix<double,6,1>& var_uvwpr)
{
  Eigen::Matrix<double,6,6> R6 = Eigen::Matrix<double,6,6>::Zero();
  for (int i=0;i<6;++i) R6(i,i) = std::max(1e-12, var_uvwpr(i));
  updateNN(mu_uvwpr, R6);
}

// GNSS EN update
void NN_EKF_V3::updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN)
{
  Eigen::Matrix<double,2,NX> H; H.setZero();
  H(0,5)=1.0; H(1,6)=1.0; // pE, pN

  Eigen::Vector2d yhat(x_(5), x_(6));
  const Eigen::Vector2d innov = pEN - yhat;

  const Eigen::Matrix2d S = H*P_*H.transpose() + R_EN;
  const Eigen::Matrix<double,NX,2> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// Heading update
void NN_EKF_V3::updateHeading(double psi_meas, double R_psi)
{
  Eigen::Matrix<double,1,NX> H; H.setZero();
  H(0,9)=1.0; // psi

  const double innov = ssa(psi_meas - x_(9));
  const double S     = (H*P_*H.transpose())(0,0) + R_psi;
  const Eigen::Matrix<double,NX,1> K = P_*H.transpose() / S;

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

Eigen::VectorXd NN_EKF_V3::getState12() const
{
  // 12-element legacy layout: [u v w p q r pE pN pD phi theta psi]
  // w (index 2) and pD/z (index 8) were removed in v3 → return 0 for them.
  Eigen::VectorXd z(12);
  z.setZero();
  z(0) = x_(0);  // u
  z(1) = x_(1);  // v
  // z(2) = 0.0; // w (removed)
  z(3) = x_(2);  // p
  z(4) = x_(3);  // q
  z(5) = x_(4);  // r
  z(6) = x_(5);  // pE
  z(7) = x_(6);  // pN
  // z(8) = 0.0; // pD/z (removed)
  z(9)  = x_(7); // phi
  z(10) = x_(8); // theta
  z(11) = x_(9); // psi
  return z;
}


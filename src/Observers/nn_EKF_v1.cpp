#include "Observers/nn_EKF_v1.hpp"
#include <cmath>
#include <torch/script.h>
#include <torch/torch.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// extern helpers you already have:
Eigen::Matrix3d Rzyx(double phi, double theta, double psi);
Eigen::Matrix3d Tzyx(double phi, double theta);
double ssa(double a);

// --------------------------- ctor / config ---------------------------
NN_EKF_V1::NN_EKF_V1()
: h_(0.01),
  g_(9.80665)
{
  x_.setZero();
  P_.setIdentity();
  Qd_.setZero();

  // Reasonable default priors (tune!!)
  // velocities/rates
  P_.block<3,3>(0,0).setIdentity();   P_.block<3,3>(0,0) *= 0.5*0.5;       // u,v,w
  P_.block<3,3>(3,3).setIdentity();   P_.block<3,3>(3,3) *= std::pow(5.0*M_PI/180.0,2); // p,q,r
  // positions
  P_.block<3,3>(6,6).setIdentity();   P_.block<3,3>(6,6) *= 5.0*5.0;       // pE,pN,pD
  // attitude
  P_.block<3,3>(9,9).setIdentity();   P_.block<3,3>(9,9) *= std::pow(5.0*M_PI/180.0,2);
  // NN biases
  P_.block<3,3>(12,12).setIdentity(); P_.block<3,3>(12,12) *= 0.2*0.2;     // b_u,b_v,b_w (m/s)^2
  P_.block<3,3>(15,15).setIdentity(); P_.block<3,3>(15,15) *= std::pow(0.5*M_PI/180.0,2);

  // Process noise (per step). Random walks; tune to match maneuvering.
  Qd_.block<3,3>(0,0).setIdentity();   Qd_.block<3,3>(0,0)   *= 1e-3;  // u,v,w
  Qd_.block<3,3>(3,3).setIdentity();   Qd_.block<3,3>(3,3)   *= 1e-4;  // p,q,r
  Qd_.block<3,3>(6,6).setIdentity();   Qd_.block<3,3>(6,6)   *= 1e-6;  // pE,pN,pD
  Qd_.block<3,3>(9,9).setIdentity();   Qd_.block<3,3>(9,9)   *= 1e-5;  // phi,theta,psi
  Qd_.block<3,3>(12,12).setIdentity(); Qd_.block<3,3>(12,12) *= 1e-6;  // b_u,b_v,b_w
  Qd_.block<3,3>(15,15).setIdentity(); Qd_.block<3,3>(15,15) *= 1e-6;  // b_p,b_q,b_r
}

void NN_EKF_V1::setDt(double h){ h_ = h; }
void NN_EKF_V1::setGravity(double g){ g_ = g; }
void NN_EKF_V1::setState(const VecN& x0){ x_ = x0; }
void NN_EKF_V1::setCovariance(const MatN& P0){ P_ = P0; }
void NN_EKF_V1::setProcessNoise(const MatN& Qd){ Qd_ = Qd; }
const NN_EKF_V1::VecN& NN_EKF_V1::state() const { return x_; }
const NN_EKF_V1::MatN& NN_EKF_V1::covariance() const { return P_; }

// ----------------------------- dynamics ------------------------------
NN_EKF_V1::VecN NN_EKF_V1::f(const VecN& xs) const {
  VecN xd; xd.setZero();

  const double u=xs(0), v=xs(1), w=xs(2);
  const double p=xs(3), q=xs(4), r=xs(5);
  const double phi=xs(9), th=xs(10), psi=xs(11);

  // Velocity & rate dynamics: random walks (0 drift, driven by Q)
  xd.segment<3>(0).setZero(); // u,v,w
  xd.segment<3>(3).setZero(); // p,q,r

  // Position kinematics in END: p_dot = R_b2n * v_b
  const Eigen::Matrix3d Rbn = Rzyx(phi, th, psi); // body->END
  const Eigen::Vector3d vb(u,v,w);
  xd.segment<3>(6) = Rbn * vb;

  // Attitude kinematics
  const Eigen::Matrix3d T = Tzyx(phi, th);
  const Eigen::Vector3d wb(p,q,r);
  xd.segment<3>(9) = T * wb; // [phi_dot, theta_dot, psi_dot]

  // Bias random walks (derivative 0; Qd drives diffusion)
  xd.segment<3>(12).setZero();
  xd.segment<3>(15).setZero();

  return xd;
}

NN_EKF_V1::MatN NN_EKF_V1::A_numeric(const VecN& xs, double eps) const {
  MatN A = MatN::Zero();
  VecN f0 = f(xs);
  for (int i=0;i<NX;++i){
    VecN xh = xs; xh(i) += eps;
    A.col(i) = (f(xh) - f0) / eps;
  }
  return A;
}

// ------------------------------- predict ----------------------------
void NN_EKF_V1::predict() {
  // State propagate (Euler)
  const VecN xd = f(x_);
  x_ += h_ * xd;

  // Covariance propagate
  const MatN A  = A_numeric(x_);
  const MatN Fd = MatN::Identity() + h_ * A;
  P_ = Fd * P_ * Fd.transpose() + Qd_;
}

// --------------------------- NN update (6D) --------------------------
void NN_EKF_V1::updateNN(const Eigen::Matrix<double,6,1>& mu_uvwpr,
                        const Eigen::Matrix<double,6,1>& var_uvwpr)
{
  // y = [u+b_u, v+b_v, w+b_w, p+b_p, q+b_q, r+b_r]^T + n
  Eigen::Matrix<double,6,NX> H; H.setZero();
  // map u..r
  H(0,0)=1; H(1,1)=1; H(2,2)=1;    // u,v,w
  H(3,3)=1; H(4,4)=1; H(5,5)=1;    // p,q,r
  // add bias columns
  H(0,12)=1; H(1,13)=1; H(2,14)=1; // b_u,b_v,b_w
  H(3,15)=1; H(4,16)=1; H(5,17)=1; // b_p,b_q,b_r

  Eigen::Matrix<double,6,1> yhat;
  yhat << x_(0)+x_(12), x_(1)+x_(13), x_(2)+x_(14),
          x_(3)+x_(15), x_(4)+x_(16), x_(5)+x_(17);

  const Eigen::Matrix<double,6,1> innov = mu_uvwpr - yhat;

  Eigen::Matrix<double,6,6> R = Eigen::Matrix<double,6,6>::Zero();
  for (int i=0;i<6;++i) R(i,i) = std::max(1e-9, var_uvwpr(i));

  const Eigen::Matrix<double,6,6> S = H*P_*H.transpose() + R;
  const Eigen::Matrix<double,NX,6> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// ------------------------ GNSS pos (E,N) update ----------------------
void NN_EKF_V1::updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN)
{
  Eigen::Matrix<double,2,NX> H; H.setZero();
  H(0,6)=1.0; // pE
  H(1,7)=1.0; // pN

  Eigen::Vector2d yhat; yhat << x_(6), x_(7);
  const Eigen::Vector2d innov = pEN - yhat;

  const Eigen::Matrix2d S = H*P_*H.transpose() + R_EN;
  const Eigen::Matrix<double,NX,2> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// ------------------------- GNSS pos (E,N,D) update -------------------
void NN_EKF_V1::updatePosEND(const Eigen::Vector3d& pEND, const Eigen::Matrix3d& R_END)
{
  Eigen::Matrix<double,3,NX> H; H.setZero();
  H(0,6)=1.0; H(1,7)=1.0; H(2,8)=1.0;

  Eigen::Vector3d yhat = x_.segment<3>(6);
  const Eigen::Vector3d innov = pEND - yhat;

  const Eigen::Matrix3d S = H*P_*H.transpose() + R_END;
  const Eigen::Matrix<double,NX,3> K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// ------------------------------ heading ------------------------------
void NN_EKF_V1::updateHeading(double psi_meas, double R_psi)
{
  Eigen::Matrix<double,1,NX> H; H.setZero();
  H(0,11)=1.0; // psi

  const double innov = ssa(psi_meas - x_(11));
  const double S     = (H*P_*H.transpose())(0,0) + R_psi;
  const Eigen::Matrix<double,NX,1> K = P_*H.transpose() / S;

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

// ------------------------------ export -------------------------------
Eigen::VectorXd NN_EKF_V1::getState12() const
{
  Eigen::VectorXd z(12);
  // BODY velocities/rates
  z(0)=x_(0); z(1)=x_(1); z(2)=x_(2);
  z(3)=x_(3); z(4)=x_(4); z(5)=x_(5);
  // END positions
  z(6)=x_(6); z(7)=x_(7); z(8)=x_(8);
  // attitude
  z(9)=x_(9); z(10)=x_(10); z(11)=x_(11);
  return z;
}


// ----------------------------------------NN Interface-------------------------------------------------


struct NNObserver::Impl {
  torch::jit::Module module;
  bool use_cuda = false;
  bool is_stateless = false;     // true if model expects (x, h_prev) and returns (..., h_next)
  mutable torch::Tensor h_;      // cached hidden state (stateless model only)
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

  // 0) Env override (highest priority)
  if (const char* env = std::getenv("MVS_NN_MODEL")) {
    if (*env) cands.emplace_back(env);
  }

  // 1) As given (could be absolute or relative to CWD)
  if (!requested.empty()) cands.emplace_back(requested);

  // 2) If requested is relative, also try relative to project root
#ifdef MVS_PROJECT_ROOT
  if (!requested.empty() && requested.front() != '/') {
    cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/" + requested);
  }
  // 3) Preferred defaults under project root
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v1/model_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v2/model.pt");
  // 4) Legacy path (your old code)
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/external/nn_observer/model.pt");
#endif

  // Remove duplicates while preserving order
  std::vector<std::string> uniq;
  uniq.reserve(cands.size());
  for (const auto& s : cands) {
    bool seen = false;
    for (const auto& u : uniq) if (u == s) { seen = true; break; }
    if (!seen) uniq.push_back(s);
  }
  return uniq;
}

NNObserver::NNObserver() : p_(new Impl) {}
NNObserver::~NNObserver() { delete p_; }

void NNObserver::load(const std::string& path) {
  load(path, /*use_cuda_request=*/true);
}

void NNObserver::load(const std::string& path, bool use_cuda_request) {
  p_->use_cuda = decide_use_cuda(use_cuda_request);
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  auto cands = build_candidate_paths(path);
  std::vector<std::string> tried;
  std::exception_ptr last_exc;

  for (const auto& cand : cands) {
    if (!file_exists(cand)) {
      tried.push_back(cand + " (missing)");
      continue;
    }
    try {
      auto mod = torch::jit::load(cand, dev);
      mod.eval();

      // Signature detection:
      bool is_stateless = false;
      try {
        auto x = torch::zeros({1,6}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        torch::IValue none; // Optional[Tensor] None
        auto any = mod.forward({x, none});
        if (any.isTuple() && any.toTuple()->elements().size() == 3) {
          is_stateless = true;
        }
      } catch (...) {
        // Not stateless; fall through to stateful check below.
      }

      if (!is_stateless) {
        auto x = torch::zeros({1,6}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        auto any = mod.forward({x});
        if (!any.isTuple() || any.toTuple()->elements().size() < 1) {
          tried.push_back(cand + " (bad signature)");
          continue; // try next
        }
      }
      // Success: assign to impl and report
      p_->module      = std::move(mod);
      p_->is_stateless = is_stateless;
      p_->h_ = torch::Tensor(); // clear hidden on load

      std::cout << "[NNObserver] Loaded '" << cand << "' on "
                << (p_->use_cuda ? "CUDA" : "CPU")
                << " | signature=" << (p_->is_stateless ? "stateless(x,h)->(mu,logvar,h)"
                                                         : "stateful(x)->(mu,logvar)")
                << std::endl;
      return;
    } catch (...) {
      last_exc = std::current_exception();
      tried.push_back(cand + " (load failed)");
    }
  }

  std::cerr << "[NNObserver] ERROR: could not load a TorchScript model.\nTried:\n";
  for (const auto& s : tried) std::cerr << "  - " << s << "\n";
  if (last_exc) {
    try { std::rethrow_exception(last_exc); }
    catch (const std::exception& e) {
      std::cerr << "Last error: " << e.what() << "\n";
    }
  }
  throw std::runtime_error("NNObserver: no valid model found.");
}

void NNObserver::reset() {
  p_->h_ = torch::Tensor();  // stateless: clears cached hidden; stateful: harmless no-op
}

std::pair<std::array<double,6>, std::array<double,6>>
NNObserver::predict_uvwpqr(const std::array<double,6>& imu) const {
  torch::NoGradGuard ng;
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  // Create input [1,6] float32 on correct device
  torch::Tensor x = torch::from_blob(
      const_cast<double*>(imu.data()),
      {1,6},
      torch::TensorOptions().dtype(torch::kFloat64));
  if (p_->use_cuda) x = x.to(torch::kCUDA);
  x = x.to(torch::kFloat32);

  torch::Tensor mu, logvar;

  if (p_->is_stateless) {
    // forward(x, h_prev) -> (mu, logvar, h_next)
    torch::IValue h_in;
    if (p_->h_.defined()) {
      h_in = p_->h_;
    } else {
      h_in = torch::IValue(); // None
    }
    auto any = p_->module.forward({x, h_in});
    auto tup = any.toTuple();
    mu     = tup->elements()[0].toTensor(); // [1,6]
    logvar = tup->elements()[1].toTensor(); // [1,6]
    p_->h_ = tup->elements()[2].toTensor(); // [L,1,H]
  } else {
    // stateful: forward(x) -> (mu, logvar)
    auto any = p_->module.forward({x});
    auto tup = any.toTuple();
    mu     = tup->elements()[0].toTensor();
    if (tup->elements().size() >= 2) {
      logvar = tup->elements()[1].toTensor();
    } else {
      // Deterministic head fallback: synthesize small variance
      logvar = torch::full_like(mu, std::log(1e-6f));
    }
  }

  // Convert to double and exponentiate variance
  torch::Tensor var = torch::exp(logvar).to(torch::kFloat64);
  mu  = mu.to(torch::kFloat64);

  std::array<double,6> mu6{}, var6{};
  auto mu_s  = mu.squeeze(0).contiguous();
  auto var_s = var.squeeze(0).contiguous();
  for (int i = 0; i < 6; ++i) {
    mu6[i] = mu_s[i].item<double>();
    double v = var_s[i].item<double>();
    var6[i] = (v < 1e-9 ? 1e-9 : v);  // small floor for numerical stability
  }
  return {mu6, var6};
}
// ============================================================================
// nn_ekf_v4.cpp
// ============================================================================
#include "Observers/nn_ekf_v4.hpp"

#include <fstream>
#include <iostream>
#include <vector>

// Torch ONLY in this TU
#include <torch/script.h>
#include <torch/torch.h>

// ================================ NN Interface ================================
struct NN_INTERFACE_V4::Impl {
  torch::jit::Module module;
  bool use_cuda = false;
  bool is_stateless = false;      // expects (x, h_prev) and returns (..., h_next)
  bool is_ensemble = false;       // ensemble export returns cov [B,7,7]
  bool second_is_logvar = true;   // single-model export: second is logvar[7]
  bool loaded = false;
  mutable torch::Tensor h_;       // [L,B,H] or [M,L,B,H]
};

static bool decide_use_cuda(bool requested) {
#ifdef TORCH_CUDA_AVAILABLE
  if (!requested) return false;
  try { (void)torch::rand({1}).to(torch::kCUDA); return true; } catch (...) { return false; }
#else
  (void)requested; return false;
#endif
}

static bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary); return f.good();
}

static std::vector<std::string> build_candidate_paths_v4(const std::string& requested) {
  std::vector<std::string> cands;
  if (!requested.empty()) cands.emplace_back(requested);
#ifdef MVS_PROJECT_ROOT
  if (!requested.empty() && requested.front() != '/') {
    cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/" + requested);
  }
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v4/ensemble_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v4/model_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v4/model.pt");
#endif
  // dedupe
  std::vector<std::string> uniq;
  for (const auto& s : cands) { bool seen=false; for (const auto& u : uniq) if (u==s){seen=true;break;} if (!seen) uniq.push_back(s); }
  return uniq;
}

NN_INTERFACE_V4::NN_INTERFACE_V4() : p_(new Impl) {}
NN_INTERFACE_V4::~NN_INTERFACE_V4() { delete p_; }

void NN_INTERFACE_V4::load(const std::string& model_path, bool use_cuda_request) {
  p_->use_cuda = decide_use_cuda(use_cuda_request);
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  auto cands = build_candidate_paths_v4(model_path);
  std::vector<std::string> tried; std::exception_ptr last_exc;

  for (const auto& cand : cands) {
    if (!file_exists(cand)) { tried.push_back(cand + " (missing)"); continue; }
    try {
      auto mod = torch::jit::load(cand, dev); mod.eval();

      bool is_stateless=false, is_ensemble=false, second_is_logvar=true;
      try {
        auto x = torch::zeros({1,8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        torch::IValue none; // None
        auto any = mod.forward({x, none});
        if (any.isTuple() && any.toTuple()->elements().size()==3) {
          is_stateless = true;
          auto tup = any.toTuple();
          // Determine if ensemble (cov [B,7,7]) or single (logvar [B,7])
          const auto s2 = tup->elements()[1];
          if (s2.isTensor()) {
            const auto t = s2.toTensor();
            is_ensemble = (t.dim()==3 && t.size(1)==7 && t.size(2)==7);
            second_is_logvar = !is_ensemble;
          }
          // Grab and discard h_next to exercise path
          (void)tup->elements()[2];
        }
      } catch (...) {
        // v4 only supports stateless exports; keep probing
      }

      if (!is_stateless) {
        tried.push_back(cand + " (unexpected signature; v4 expects stateless TS)");
        continue;
      }

      p_->module = std::move(mod);
      p_->is_stateless = is_stateless;
      p_->is_ensemble  = is_ensemble;
      p_->second_is_logvar = second_is_logvar;
      p_->h_ = torch::Tensor();
      p_->loaded = true;

      std::cout << "[NN_INTERFACE_V4] Loaded '" << cand << "' on "
                << (p_->use_cuda ? "CUDA" : "CPU")
                << " | sig=" << (p_->is_ensemble ? "ensemble(x,H)->(mu7,cov7,H)" : "single(x,h)->(mu7,logvar7,h)")
                << std::endl;
      return;
    } catch (...) {
      last_exc = std::current_exception();
      tried.push_back(cand + " (load failed)");
    }
  }

  std::cerr << "[NN_INTERFACE_V4] ERROR: could not load a TorchScript model. Tried:\n";
  for (const auto& s : tried) std::cerr << "  - " << s << "\n";
  if (last_exc) { try { std::rethrow_exception(last_exc); } catch (const std::exception& e) { std::cerr << "Last error: " << e.what() << "\n"; } }
  throw std::runtime_error("NN_INTERFACE_V4: no valid model found.");
}

void NN_INTERFACE_V4::reset() { p_->h_ = torch::Tensor(); }

std::pair<std::array<double,7>, Eigen::Matrix<double,7,7>>
NN_INTERFACE_V4::predict(const std::array<double,8>& x_in) const {
  if (!p_->loaded) throw std::runtime_error("NN_INTERFACE_V4::predict(): model not loaded");

  torch::NoGradGuard ng;
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  // Input: [ax,ay,az, wx,wy,wz, cpsi,spsi]
  std::array<float,8> xraw{};
  for (int i=0;i<8;++i) xraw[i] = static_cast<float>(x_in[i]);

  torch::Tensor x = torch::from_blob((void*)xraw.data(), {1,8}, torch::TensorOptions().dtype(torch::kFloat32));
  if (p_->use_cuda) x = x.to(torch::kCUDA);

  // Forward
  auto any = p_->module.forward({x, p_->h_.defined()? torch::IValue(p_->h_) : torch::IValue()});
  auto tup = any.toTuple();
  torch::Tensor mu_t = tup->elements()[0].toTensor(); // [1,7]
  torch::Tensor s2_t = tup->elements()[1].toTensor(); // [1,7] logvar OR [1,7,7] cov
  p_->h_ = tup->elements()[2].toTensor();             // [L,1,H] or [M,L,1,H]

  std::array<double,7> mu{};
  auto mu_d = mu_t.to(torch::kFloat64).squeeze(0).contiguous();
  for (int i=0;i<7;++i) mu[i] = mu_d[i].item<double>();

  Eigen::Matrix<double,7,7> C; C.setZero();
  if (p_->is_ensemble) {
    // s2_t is predictive covariance [1,7,7] in PHYSICAL units
    auto cov = s2_t.to(torch::kFloat64).squeeze(0).contiguous();
    for (int i=0;i<7;++i) for (int j=0;j<7;++j)
      C(i,j) = cov.index({i,j}).item<double>();
  } else {
    // s2_t is PHYSICAL log-variance [1,7]
    auto lv = s2_t.to(torch::kFloat64).squeeze(0).contiguous();
    for (int i=0;i<7;++i) {
      double var = std::exp(lv[i].item<double>());
      if (var < 1e-12) var = 1e-12;
      C(i,i) = var;
    }
  }

  return {mu, C};
}

// ================================== EKF v4 ==================================
NN_EKF_V4::NN_EKF_V4()
: h_(0.05)
{
  x_.setZero();
  P_.setIdentity();
  Qd_.setZero();

  // States: [Vn, Ve, r, pE, pN, psi, bVn, bVe]
  P_.diagonal() <<
    0.5*0.5, 0.5*0.5,                 // Vn, Ve (m/s)
    std::pow(5.0*M_PI/180.0,2),       // r (rad/s)
    5.0*5.0, 5.0*5.0,                 // pE, pN (m)
    std::pow(10.0*M_PI/180.0,2),      // psi (rad)
    0.2*0.2, 0.2*0.2;                 // bVn, bVe (m/s biases)

  // Process noise (discrete RW); tune as needed
  Qd_.setZero();
  Qd_.diagonal() <<
    1e-4, 1e-4,        // Vn, Ve
    5e-5,              // r
    1e-8, 1e-8,        // pE, pN (drift only via velocities)
    5e-5,              // psi
    1e-6, 1e-6;        // bVn, bVe (very slow RW — mostly corrected by GNSS)
}

void NN_EKF_V4::setDt(double h){ h_ = h; }
void NN_EKF_V4::setState(const VecN& x0){ x_ = x0; }
void NN_EKF_V4::setCovariance(const MatN& P0){ P_ = P0; }
void NN_EKF_V4::setProcessNoise(const MatN& Qd){ Qd_ = Qd; }

void NN_EKF_V4::initNN(const std::string& model_path, bool use_cuda_request) {
  if (!nn_) nn_ = std::make_unique<NN_INTERFACE_V4>();
  nn_->load(model_path, use_cuda_request);
  nn_->reset();
}

void NN_EKF_V4::resetNN(){ if (nn_) nn_->reset(); }

// xdot = f(x)
NN_EKF_V4::VecN NN_EKF_V4::f(const VecN& xs) const {
  VecN xd; xd.setZero();
  const double Vn = xs(0), Ve = xs(1), r = xs(2);
  const double bVn = xs(6), bVe = xs(7);

  // Random walks for Vn, Ve, r
  xd(0) = 0.0;  // Vn
  xd(1) = 0.0;  // Ve
  xd(2) = 0.0;  // r

  // Position kinematics (END): integrate nav velocities minus bias
  xd(3) = Ve - bVe; // pE_dot
  xd(4) = Vn - bVn; // pN_dot

  // Yaw kinematics
  xd(5) = r;       // psi_dot

  // Bias RW
  xd(6) = 0.0;     // bVn
  xd(7) = 0.0;     // bVe
  return xd;
}

NN_EKF_V4::MatN NN_EKF_V4::A_numeric(const VecN& xs, double eps) const {
  MatN A = MatN::Zero();
  const VecN f0 = f(xs);
  for (int i=0;i<NX;++i){ VecN xh = xs; xh(i) += eps; A.col(i) = (f(xh) - f0) / eps; }
  return A;
}

void NN_EKF_V4::predict() {
  const VecN xd = f(x_);
  x_ += h_ * xd;
  x_(5) = ssa(x_(5)); // keep yaw bounded

  const MatN A  = A_numeric(x_);
  const MatN Fd = MatN::Identity() + h_ * A;
  P_ = Fd * P_ * Fd.transpose() + Qd_;
}

void NN_EKF_V4::updateFromIMU(const std::array<double,8>& imu8) {
  if (!nn_) throw std::runtime_error("NN_EKF_V4::updateFromIMU(): NN not initialized.");
  const auto [mu7, Cov7] = nn_->predict(imu8);

  // Cache extras for getState12()
  nn_p_     = mu7[2];
  nn_q_     = mu7[3];
  nn_phi_   = mu7[5];
  nn_theta_ = mu7[6];

  updateNN(mu7, Cov7);
}

void NN_EKF_V4::updateNN(const std::array<double,7>& mu7, const Eigen::Matrix<double,7,7>& Cov7) {
  // Indices in NN output
  constexpr int ID_VN = 0; constexpr int ID_VE = 1; constexpr int ID_R = 4;

  // z = [Vn, Ve, r]^T
  Eigen::Matrix<double,3,1> z;
  z(0) = mu7[ID_VN];
  z(1) = mu7[ID_VE];
  z(2) = mu7[ID_R];

  // Extract 3x3 covariance
  Eigen::Matrix<double,3,3> R3; R3.setZero();
  R3(0,0) = Cov7(ID_VN, ID_VN);
  R3(0,1) = Cov7(ID_VN, ID_VE); R3(1,0) = R3(0,1);
  R3(0,2) = Cov7(ID_VN, ID_R);  R3(2,0) = R3(0,2);
  R3(1,1) = Cov7(ID_VE, ID_VE);
  R3(1,2) = Cov7(ID_VE, ID_R);  R3(2,1) = R3(1,2);
  R3(2,2) = Cov7(ID_R,  ID_R);
  for (int i=0;i<3;++i) if (R3(i,i) < 1e-12) R3(i,i) = 1e-12;

  // h(x) = [Vn, Ve, r]^T
  Eigen::Matrix<double,3,NX> H; H.setZero();
  H(0,0) = 1.0; // Vn
  H(1,1) = 1.0; // Ve
  H(2,2) = 1.0; // r

  Eigen::Matrix<double,3,1> yhat; yhat << x_(0), x_(1), x_(2);
  const auto innov = z - yhat;
  const auto S = H*P_*H.transpose() + R3;
  const auto K = P_*H.transpose()*S.inverse();

  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

void NN_EKF_V4::updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN) {
  Eigen::Matrix<double,2,NX> H; H.setZero(); H(0,3)=1.0; H(1,4)=1.0; // pE, pN
  Eigen::Vector2d yhat(x_(3), x_(4));
  const Eigen::Vector2d innov = pEN - yhat;
  const auto S = H*P_*H.transpose() + R_EN;
  const auto K = P_*H.transpose()*S.inverse();
  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

void NN_EKF_V4::updateHeading(double psi_meas, double R_psi) {
  Eigen::Matrix<double,1,NX> H; H.setZero(); H(0,5)=1.0; // psi
  const double innov = ssa(psi_meas - x_(5));
  const double S = (H*P_*H.transpose())(0,0) + R_psi;
  const auto K = P_*H.transpose() / S;
  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_;
}

Eigen::VectorXd NN_EKF_V4::getState12() const {
  // 12-element legacy layout: [u v w p q r pE pN pD phi theta psi]
  Eigen::VectorXd z(12); z.setZero();

  // Back-compute body velocities from nav-frame (Ve, Vn, Vd=0)
  const double psi = x_(5);
  const Eigen::Matrix3d Rbn = Rzyx(nn_phi_, nn_theta_, psi); // body->nav
  const Eigen::Vector3d v_nav(x_(1), x_(0), 0.0);            // [Ve, Vn, Vd]
  const Eigen::Vector3d v_body = Rbn.transpose() * v_nav;    // [u, v, w]

  z(0) = v_body.x(); // u
  z(1) = v_body.y(); // v
  z(2) = v_body.z(); // w (≈0 if Vd assumed 0)

  z(3) = nn_p_;      // p  (from NN cache)
  z(4) = nn_q_;      // q
  z(5) = x_(2);      // r  (EKF state)

  z(6) = x_(3);      // pE (East)
  z(7) = x_(4);      // pN (North)
  // z(8) = 0.0;     // pD (Down)

  z(9)  = nn_phi_;   // phi   (from NN cache)
  z(10) = nn_theta_; // theta (from NN cache)
  z(11) = psi;       // psi   (EKF state)
  return z;
}

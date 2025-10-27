// ============================================================================
// nn_ekf_v5.cpp
// ============================================================================

#include "Observers/nn_ekf_v5.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <array>

// Torch ONLY in this TU
#include <torch/script.h>
#include <torch/torch.h>

// ================================ NN Interface ================================
struct NN_INTERFACE_V5::Impl {
  torch::jit::Module module;
  bool use_cuda = false;
  bool is_stateless = false;      // expects (x, h_prev) and returns (..., h_next)
  bool is_ensemble = false;       // ensemble export returns cov [B,9,9]
  bool second_is_logvar = true;   // single-model export: second is logvar[9]
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

static std::vector<std::string> build_candidate_paths_v5(const std::string& requested) {
  std::vector<std::string> cands;
  if (!requested.empty()) cands.emplace_back(requested);
#ifdef MVS_PROJECT_ROOT
  if (!requested.empty() && requested.front() != '/') {
    cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/" + requested);
  }
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v5/ensemble_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v5/model_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model_v5/model.pt");
#endif
  // dedupe
  std::vector<std::string> uniq;
  for (const auto& s : cands) { bool seen=false; for (const auto& u : uniq) if (u==s){seen=true;break;} if (!seen) uniq.push_back(s); }
  return uniq;
}

NN_INTERFACE_V5::NN_INTERFACE_V5() : p_(new Impl) {}
NN_INTERFACE_V5::~NN_INTERFACE_V5() { delete p_; }

void NN_INTERFACE_V5::load(const std::string& model_path, bool use_cuda_request) {
  p_->use_cuda = decide_use_cuda(use_cuda_request);
  const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);

  auto cands = build_candidate_paths_v5(model_path);
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
          // Determine if ensemble (cov [B,9,9]) or single (logvar [B,9])
          const auto s2 = tup->elements()[1];
          if (s2.isTensor()) {
            const auto t = s2.toTensor();
            is_ensemble = (t.dim()==3 && t.size(1)==9 && t.size(2)==9);
            second_is_logvar = !is_ensemble;
          }
          (void)tup->elements()[2]; // h_next
        }
      } catch (...) {
        // v5 expects stateless; keep probing
      }

      if (!is_stateless) {
        tried.push_back(cand + " (unexpected signature; v5 expects stateless TS)");
        continue;
      }

      p_->module = std::move(mod);
      p_->is_stateless = is_stateless;
      p_->is_ensemble  = is_ensemble;
      p_->second_is_logvar = second_is_logvar;
      p_->h_ = torch::Tensor();
      p_->loaded = true;

      std::cout << "[NN_INTERFACE_V5] Loaded '" << cand << "' on "
                << (p_->use_cuda ? "CUDA" : "CPU")
                << " | sig=" << (p_->is_ensemble ? "ensemble(x,H)->(mu9,cov9,H)" : "single(x,h)->(mu9,logvar9,h)")
                << std::endl;
      return;
    } catch (...) {
      last_exc = std::current_exception();
      tried.push_back(cand + " (load failed)");
    }
  }

  std::cerr << "[NN_INTERFACE_V5] ERROR: could not load a TorchScript model. Tried:\n";
  for (const auto& s : tried) std::cerr << "  - " << s << "\n";
  if (last_exc) { try { std::rethrow_exception(last_exc); } catch (const std::exception& e) { std::cerr << "Last error: " << e.what() << "\n"; } }
  throw std::runtime_error("NN_INTERFACE_V5: no valid model found.");
}

void NN_INTERFACE_V5::reset() { p_->h_ = torch::Tensor(); }

std::pair<std::array<double,9>, Eigen::Matrix<double,9,9>>
NN_INTERFACE_V5::predict(const std::array<double,8>& x_in) const {
  if (!p_->loaded) throw std::runtime_error("NN_INTERFACE_V5::predict(): model not loaded");

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
  torch::Tensor mu_t = tup->elements()[0].toTensor(); // [1,9]
  torch::Tensor s2_t = tup->elements()[1].toTensor(); // [1,9] logvar OR [1,9,9] cov
  p_->h_ = tup->elements()[2].toTensor();             // [L,1,H] or [M,L,1,H]

  // Means
  std::array<double,9> mu{};
  auto mu_d = mu_t.to(torch::kFloat64).squeeze(0).contiguous();
  for (int i=0;i<9;++i) mu[i] = mu_d[i].item<double>();

  // Covariance
  Eigen::Matrix<double,9,9> C; C.setZero();
  if (p_->is_ensemble) {
    // predictive covariance [1,9,9] in PHYSICAL units
    auto cov = s2_t.to(torch::kFloat64).squeeze(0).contiguous();
    for (int i=0;i<9;++i) for (int j=0;j<9;++j)
      C(i,j) = cov.index({i,j}).item<double>();
  } else {
    // PHYSICAL log-variance [1,9] → diagonal covariance
    auto lv = s2_t.to(torch::kFloat64).squeeze(0).contiguous();
    for (int i=0;i<9;++i) {
      double var = std::exp(lv[i].item<double>());
      if (var < 1e-12) var = 1e-12;
      C(i,i) = var;
    }
  }

  return {mu, C};
}

// ================================== EKF v5 ==================================
NN_EKF_V5::NN_EKF_V5()
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
    1e-5, 1e-5;        // bVn, bVe (give GNSS a chance to pull the biases)
}

void NN_EKF_V5::setDt(double h){ h_ = h; }
void NN_EKF_V5::setState(const VecN& x0){ x_ = x0; }
void NN_EKF_V5::setCovariance(const MatN& P0){ P_ = P0; }
void NN_EKF_V5::setProcessNoise(const MatN& Qd){ Qd_ = Qd; }

void NN_EKF_V5::initNN(const std::string& model_path, bool use_cuda_request) {
  if (!nn_) nn_ = std::make_unique<NN_INTERFACE_V5>();
  nn_->load(model_path, use_cuda_request);
  nn_->reset();
}

void NN_EKF_V5::resetNN(){ if (nn_) nn_->reset(); }

const NN_EKF_V5::VecN& NN_EKF_V5::state() const { return x_; }

// xdot = f(x)
NN_EKF_V5::VecN NN_EKF_V5::f(const VecN& xs) const {
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

NN_EKF_V5::MatN NN_EKF_V5::A_numeric(const VecN& xs, double eps) const {
  MatN A = MatN::Zero();
  const VecN f0 = f(xs);
  for (int i=0;i<NX;++i){ VecN xh = xs; xh(i) += eps; A.col(i) = (f(xh) - f0) / eps; }
  return A;
}

void NN_EKF_V5::predict() {
  const VecN xd = f(x_);
  x_ += h_ * xd;
  x_(5) = ssa(x_(5)); // keep yaw bounded

  const MatN A  = A_numeric(x_);
  const MatN Fd = MatN::Identity() + h_ * A;
  P_ = Fd * P_ * Fd.transpose() + Qd_;
}

void NN_EKF_V5::updateFromIMU(const std::array<double,8>& imu8) {
  if (!nn_) throw std::runtime_error("NN_EKF_V5::updateFromIMU(): NN not initialized.");
  const auto [mu9, Cov9] = nn_->predict(imu8);

  // Cache extras for getState12()
  nn_p_     = mu9[2];
  nn_q_     = mu9[3];
  nn_phi_   = mu9[5];
  nn_theta_ = mu9[6];

  updateNN(mu9, Cov9);
}

void NN_EKF_V5::updateNN(const std::array<double,9>& mu9,
                         const Eigen::Matrix<double,9,9>& Cov9)
{
  // Indices in NN output
  constexpr int ID_VN   = 0;
  constexpr int ID_VE   = 1;
  constexpr int ID_P    = 2;
  constexpr int ID_Q    = 3;
  constexpr int ID_R    = 4;
  constexpr int ID_PHI  = 5;
  constexpr int ID_TH   = 6;
  constexpr int ID_CPSI = 7;
  constexpr int ID_SPSI = 8;

  // 1) Build z = [Vn, Ve, r, psi_nn]
  const double c = mu9[ID_CPSI];
  const double s = mu9[ID_SPSI];
  const double rnorm = std::hypot(c, s);
  const double c_u = (rnorm > 1e-12) ? (c / rnorm) : 1.0;
  const double s_u = (rnorm > 1e-12) ? (s / rnorm) : 0.0;
  const double psi_nn = std::atan2(s_u, c_u);

  Eigen::Matrix<double,4,1> z;
  z << mu9[ID_VN], mu9[ID_VE], mu9[ID_R], psi_nn;

  // 2) Build the 5x5 sub-covariance for [Vn, Ve, r, cpsi, spsi]
  Eigen::Matrix<double,5,5> S = Eigen::Matrix<double,5,5>::Zero();
  const int map_in[5] = {ID_VN, ID_VE, ID_R, ID_CPSI, ID_SPSI};
  for (int i=0;i<5;++i)
    for (int j=0;j<5;++j)
      S(i,j) = Cov9(map_in[i], map_in[j]);

  // 3) Jacobian A: [Vn,Ve,r,c,s] -> [Vn,Ve,r,psi]
  // dpsi/d(c,s) = [-s_u, c_u]
  Eigen::Matrix<double,4,5> A = Eigen::Matrix<double,4,5>::Zero();
  A(0,0) = 1.0;  // Vn
  A(1,1) = 1.0;  // Ve
  A(2,2) = 1.0;  // r
  A(3,3) = -s_u; // dpsi/dc
  A(3,4) =  c_u; // dpsi/ds

  // 4) Measurement covariance R4 = A S A^T (ensure tiny floors on diag)
  Eigen::Matrix<double,4,4> R4 = A * S * A.transpose();
  for (int i=0;i<4;++i) if (R4(i,i) < 1e-12) R4(i,i) = 1e-12;

  // 5) Measurement model h(x) = [Vn, Ve, r, psi]^T
  Eigen::Matrix<double,4,NX> H; H.setZero();
  H(0,0) = 1.0; // Vn
  H(1,1) = 1.0; // Ve
  H(2,2) = 1.0; // r
  H(3,5) = 1.0; // psi

  Eigen::Matrix<double,4,1> yhat;
  yhat << x_(0), x_(1), x_(2), x_(5);

  // 6) Innovation (wrap yaw!)
  Eigen::Matrix<double,4,1> innov = z - yhat;
  innov(3) = ssa(innov(3));

  // 7) EKF update (Joseph form)
  const auto Syy = H * P_ * H.transpose() + R4;
  const auto K   = P_ * H.transpose() * Syy.inverse();
  const auto I   = MatN::Identity();

  x_ += K * innov;
  // Keep yaw bounded
  x_(5) = ssa(x_(5));

  const MatN IKH = (I - K*H);
  P_ = IKH * P_ * IKH.transpose() + K * R4 * K.transpose();
}

void NN_EKF_V5::updatePosEN(const Eigen::Vector2d& pEN, const Eigen::Matrix2d& R_EN) {
  Eigen::Matrix<double,2,NX> H; H.setZero(); H(0,3)=1.0; H(1,4)=1.0; // pE, pN
  Eigen::Vector2d yhat(x_(3), x_(4));
  const Eigen::Vector2d innov = pEN - yhat;
  const auto S = H*P_*H.transpose() + R_EN;
  const auto K = P_*H.transpose()*S.inverse();
  x_ += K * innov;
  P_  = (MatN::Identity() - K*H) * P_; // Positions are linear; standard form is fine here
}

void NN_EKF_V5::updateHeading(double psi_meas, double R_psi) {
  Eigen::Matrix<double,1,NX> H; H.setZero(); H(0,5)=1.0; // psi
  const double innov = ssa(psi_meas - x_(5));
  const double S = (H*P_*H.transpose())(0,0) + R_psi;
  const auto K = P_*H.transpose() / S;
  x_ += K * innov;
  x_(5) = ssa(x_(5));
  P_  = (MatN::Identity() - K*H) * P_;
}

Eigen::VectorXd NN_EKF_V5::getState12() const {
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

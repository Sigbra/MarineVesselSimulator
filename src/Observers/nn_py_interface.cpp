#include "Observers/nn_py_interface.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model/model_stateless.pt");
  cands.emplace_back(std::string(MVS_PROJECT_ROOT) + "/data/nn_model/model.pt");
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


// #include "Observers/nn_py_interface.hpp"
// #include <torch/script.h>
// #include <torch/torch.h>
// #include <stdexcept>
// #include <vector>
// #include <iostream>

// struct NNObserver::Impl {
//   torch::jit::Module module;
//   bool use_cuda = false;
// };

// static bool decide_use_cuda(bool requested) {
// #ifdef TORCH_CUDA_AVAILABLE
//   if (!requested) return false;
//   try {
//     auto t = torch::rand({1}).to(torch::kCUDA);
//     (void)t;
//     return true;
//   } catch (...) {
//     return false;
//   }
// #else
//   (void)requested;
//   return false;
// #endif
// }

// NNObserver::NNObserver() : p_(new Impl) {}
// NNObserver::~NNObserver(){ delete p_; }

// // Auto-select: prefer CUDA if build supports it and device is usable.
// void NNObserver::load(const std::string& path) {
//   // request CUDA; decide_use_cuda() will fall back to CPU if not usable
//   load(path, /*use_cuda=*/true);
// }

// void NNObserver::load(const std::string& path, bool use_cuda_request) {
//   p_->use_cuda = decide_use_cuda(use_cuda_request);
//   const c10::Device dev = p_->use_cuda ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);
//   p_->module = torch::jit::load(path, dev);
//   p_->module.eval();
//   std::cout << "[NNObserver] Loaded '" << path << "' on "
//             << (p_->use_cuda ? "CUDA" : "CPU") << std::endl;
// }

// std::pair<std::array<double,6>, std::array<double,6>>
// NNObserver::predict_uvwpqr(const std::array<double,6>& imu) const
// {
//   torch::NoGradGuard ng;

//   torch::Tensor x = torch::from_blob(
//       const_cast<double*>(imu.data()),
//       {1,6}, torch::TensorOptions().dtype(torch::kFloat64));

//   if (p_->use_cuda) x = x.to(torch::kCUDA);
//   x = x.to(torch::kFloat32);

//   auto out = p_->module.forward({x}).toTuple();
//   torch::Tensor mu     = out->elements()[0].toTensor(); // [1,6] float32
//   torch::Tensor logvar = out->elements()[1].toTensor(); // [1,6] float32

//   torch::Tensor var = torch::exp(logvar).to(torch::kFloat64);
//   mu  = mu.to(torch::kFloat64);

//   std::array<double,6> mu6{}, var6{};
//   auto mu_s  = mu.squeeze(0).contiguous();
//   auto var_s = var.squeeze(0).contiguous();
//   for (int i=0;i<6;++i) {
//     mu6[i]  = mu_s[i].item<double>();
//     double v = var_s[i].item<double>();
//     var6[i] = (v < 1e-6 ? 1e-6 : v);
//   }
//   return {mu6, var6};
// }


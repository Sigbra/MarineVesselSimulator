#include <tuple>
#include "Guidance/ALOS.hpp"
#include "Utilities/calculations.hpp"

ALOS::ALOS(double Delta_h, double gamma_h, double h)
    : Delta_h_(Delta_h), gamma_h_(gamma_h), h_(h),
      beta_hat(0.0){}

std::tuple<double, double> ALOS::update(double xn, double yn,
                                        double path_x, double path_y,
                                        double path_x_dot, double path_y_dot,
                                        double precomputed_y_e)
{
// Compute the path-tangent angle (pi_h) using the derivatives of the path.
double pi_h = std::atan2(path_y_dot, path_x_dot);

// Use pre-computed cross-track error if provided, otherwise calculate it
double y_e = -(xn - path_x) * std::sin(pi_h) + (yn - path_y) * std::cos(pi_h);

// Compute the desired heading (psi_ref) using the adaptive LOS guidance law.
double psi_ref = pi_h - beta_hat - std::atan(y_e / Delta_h_);

// Update the crab angle estimate.
beta_hat += h_ * gamma_h_ * Delta_h_ * y_e / std::hypot(Delta_h_, y_e);

return std::make_tuple(psi_ref, y_e);
}

void ALOS::reset() {
    beta_hat = 0.0;
}

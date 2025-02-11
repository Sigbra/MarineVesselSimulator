#include "hermite_spline.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include "utilities.hpp"

// Function to compute path length
double computePathLength(const Waypoints& wpt) {
    double pathLength = 0.0;
    for (size_t i = 1; i < wpt.x.size(); ++i) {
        double deltaLength = std::hypot(wpt.x[i] - wpt.x[i - 1], wpt.y[i] - wpt.y[i - 1]);
        pathLength += deltaLength;
    }
    return pathLength;
}

// Simple finite difference method for derivative calculation
std::vector<double> computeDerivative(const std::vector<double>& values, double step) {
    std::vector<double> derivatives(values.size(), 0.0);
    for (size_t i = 1; i < values.size() - 1; ++i) {
        derivatives[i] = (values[i + 1] - values[i - 1]) / (2.0 * step);
    }
    derivatives[0] = derivatives[1];
    derivatives.back() = derivatives[values.size() - 2];
    return derivatives;
}

// Hermite spline interpolation (approximate with linear interpolation for simplicity)
SplineResult hermiteSpline(const Waypoints& wpt, double Umax, double h) {
    double pathLength = computePathLength(wpt);
    double time = pathLength / Umax;
    int N_interval = std::floor(time / h) + 1;
    double deltaPath = pathLength / N_interval;
    int N_horizon = std::round(Umax / deltaPath);
    
    // Parameterize path
    std::vector<double> w_path(N_interval + 1);
    for (int i = 0; i <= N_interval; ++i) {
        w_path[i] = i;
    }

    // Interpolation using linear approximation (or use spline library)
    std::vector<double> x_path = wpt.x;
    std::vector<double> y_path = wpt.y;
    
    // Compute derivatives using finite differences
    std::vector<double> dx_path = computeDerivative(x_path, deltaPath);
    std::vector<double> dy_path = computeDerivative(y_path, deltaPath);
    
    // Compute heading angles relative to North
    std::vector<double> pi_h(N_interval + 1);
    for (size_t i = 0; i < pi_h.size(); ++i) {
        pi_h[i] = std::atan2(dy_path[i], dx_path[i]);
    }

    return {w_path, x_path, y_path, dx_path, dy_path, pi_h, N_horizon};
}
#include "crosstrack_hermite.hpp"

#include <limits>

// Initialize persistent variable
double CrosstrackHermiteLOS::persistentBetaHat = 0.0;

double CrosstrackHermiteLOS::computeLOSAngle(
    const PathData& path,
    double x, double y, double h, double Delta_h,
    const std::vector<double>& pp_x,
    const std::vector<double>& pp_y,
    int& idx_start,
    int N_horizon,
    double gamma_h) {
    
    int idx_end = std::min(idx_start + N_horizon, static_cast<int>(path.w_path.size()) - 1);
    std::vector<int> w_horizon;
    for (int i = idx_start; i <= idx_end; ++i) {
        w_horizon.push_back(i);
    }

    std::vector<double> x_horizon(w_horizon.size()), y_horizon(w_horizon.size());
    for (size_t i = 0; i < w_horizon.size(); ++i) {
        x_horizon[i] = pp_x[w_horizon[i]];
        y_horizon[i] = pp_y[w_horizon[i]];
    }

    double min_distance = std::numeric_limits<double>::max();
    int min_distance_idx = 0;
    for (size_t i = 0; i < x_horizon.size(); ++i) {
        double dist = std::hypot(x_horizon[i] - x, y_horizon[i] - y);
        if (dist < min_distance) {
            min_distance = dist;
            min_distance_idx = i;
        }
    }

    idx_start += min_distance_idx - 1;
    
    double vector_x = x - path.x_path[idx_start];
    double vector_y = y - path.y_path[idx_start];
    double crossProd = path.dx_path[idx_start] * vector_y - path.dy_path[idx_start] * vector_x;
    double y_e = (crossProd >= 0 ? 1 : -1) * min_distance;
    
    double LOSangle;
    if (gamma_h < 0) {
        LOSangle = path.pi_h[idx_start] - std::atan(y_e / Delta_h);
    } else {
        LOSangle = path.pi_h[idx_start] - persistentBetaHat - std::atan(y_e / Delta_h);
        persistentBetaHat += h * gamma_h * Delta_h * y_e / std::sqrt(Delta_h * Delta_h + y_e * y_e);
    }
    
    return LOSangle;
}
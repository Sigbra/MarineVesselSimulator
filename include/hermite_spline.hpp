#ifndef HERMITE_SPLINE_HPP
#define HERMITE_SPLINE_HPP

#include "utilities.hpp"

struct SplineResult {
    std::vector<double> w_path, x_path, y_path, dx_path, dy_path, pi_h;
    int N_horizon;
};

double computePathLength(const Waypoints& wpt);
std::vector<double> computeDerivative(const std::vector<double>& values, double step);
SplineResult hermiteSpline(const Waypoints& wpt, double Umax, double h);

#endif // HERMITE_SPLINE_HPP



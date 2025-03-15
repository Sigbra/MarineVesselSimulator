#include "cubic_spline.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

double computeCurvature(double dx, double dy, double ddx, double ddy) {
    double numerator = std::abs(dx * ddy - dy * ddx);
    double denominator = std::pow(dx * dx + dy * dy, 1.5);
    return (denominator > 1e-6) ? numerator / denominator : 0.0;
}

std::vector<Point> generateCubicSpline(const Waypoints& wpt, double max_curvature, int num_samples) {
    std::vector<Point> path;

    if (wpt.x.size() <= 2 || wpt.y.size() <= 2 || wpt.x.size() != wpt.y.size()) {
        std::cerr << "Error: Invalid waypoint data" << std::endl;
        return path;
    }

    size_t n = wpt.x.size();

    std::vector<double> t(n);
    t[0] = 0.0;
    for (size_t i = 1; i < n; ++i){
        double dx = wpt.x[i] - wpt.x[i-1];
        double dy = wpt.y[i] - wpt.y[i-1];
        t[i] = t[i-1] + std::sqrt(dx*dx + dy*dy);
    }

    gsl_interp_accel* acc_x = gsl_interp_accel_alloc();
    gsl_interp_accel* acc_y = gsl_interp_accel_alloc();
    gsl_spline* spline_x = gsl_spline_alloc(gsl_interp_cspline, n);
    gsl_spline* spline_y = gsl_spline_alloc(gsl_interp_cspline, n);

    gsl_spline_init(spline_x, t.data(), wpt.x.data(), n);
    gsl_spline_init(spline_y, t.data(), wpt.y.data(), n);

    double t_min = t.front();
    double t_max = t.back();
    double dt = (t_max - t_min) / (num_samples - 1);

    bool curvature_ok = true;
    double global_max_curvature = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        double t_i = t_min + i * dt;
        double x_val = gsl_spline_eval(spline_x, t_i, acc_x);
        double y_val = gsl_spline_eval(spline_y, t_i, acc_y);

        double dx = gsl_spline_eval_deriv(spline_x, t_i, acc_x);
        double dy = gsl_spline_eval_deriv(spline_y, t_i, acc_y);

        double ddx = gsl_spline_eval_deriv2(spline_x, t_i, acc_x);
        double ddy = gsl_spline_eval_deriv2(spline_y, t_i, acc_y);

        double curvature = computeCurvature(dx, dy, ddx, ddy);
        global_max_curvature = std::max(global_max_curvature, curvature);

        if (curvature > max_curvature) {
            curvature_ok = false;
        }

        path.push_back({x_val, y_val});
    }

    if (!curvature_ok) {
        std::cerr << "Warning: Maximum curvature (" << global_max_curvature 
        << ") exceeds curve limit (" << max_curvature << ").\n";
    }

    gsl_spline_free(spline_x);
    gsl_spline_free(spline_y);
    gsl_interp_accel_free(acc_x);
    gsl_interp_accel_free(acc_y);

    return path;
}
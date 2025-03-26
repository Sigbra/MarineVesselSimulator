#ifndef LOS_HPP
#define LOS_HPP

#include <tuple>
#include <limits>
#include <cmath>

std::tuple<double, double> LOS(double xn, double yn, double delta,
    double path_x, double path_y,
    double path_x_dot, double path_y_dot,
    double precomputed_y_e = std::numeric_limits<double>::quiet_NaN());

#endif
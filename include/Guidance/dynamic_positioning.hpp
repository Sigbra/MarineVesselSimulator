#ifndef DP_HPP
#define DP_HPP

#include <tuple>

std::tuple<double, double, double> DP(double path_x, double path_y,
    double prev_path_x, double prev_path_y,
  double xn, double yn);

#endif
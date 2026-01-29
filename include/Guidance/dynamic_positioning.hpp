#ifndef DP_HPP
#define DP_HPP

#include <tuple>
#include <optional>

std::tuple<double, double, double> DP(double xn, double yn,
                                      double path_x, double path_y,
                                      double prev_path_x, double prev_path_y, std::optional<double> psi_d_input = std::nullopt);

#endif
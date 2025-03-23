#include "Guidance/dynamic_positioning.hpp"
#include <tuple>
#include <cmath>

std::tuple<double, double, double> DP(double xn, double yn,
    double path_x, double path_y,
    double prev_path_x, double prev_path_y)
{
double xn_d = path_x;
double yn_d = path_y;
double psi_d = std::atan2(path_y - prev_path_y, path_x - prev_path_x);

return std::make_tuple(xn_d, yn_d, psi_d);
}

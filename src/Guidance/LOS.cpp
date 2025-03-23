#include "Guidance/LOS.hpp"
#include <cmath>
#include <tuple>
#include <iostream>

std::tuple<double, double> LOS(double xn, double yn, double delta,
    double path_x, double path_y,
    double path_x_dot, double path_y_dot)
{

const double epsilon = 1e-6;
if (std::abs(path_x_dot) < epsilon && std::abs(path_y_dot) < epsilon) {
    double psi_ref = 0.0;
    double y_e = 0.0;  
    std::cout << "Warning; derivatives too small in LOS()" << std::endl;
    return std::make_tuple(psi_ref, y_e);
}

double psi_p = std::atan2(path_y_dot, path_x_dot);
double y_e = -(xn - path_x)*std::sin(psi_p) 
             +(yn - path_y)*std::cos(psi_p);

double psi_ref = psi_p - std::atan(y_e / delta);

return std::make_tuple(psi_ref, y_e);
}
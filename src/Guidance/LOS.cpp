#include "Guidance/LOS.hpp"
#include "Utilities/calculations.hpp"
#include <cmath>
#include <tuple>
#include <iostream>

// std::tuple<double, double> LOS(double xn, double yn, double delta,
//     double path_x, double path_y,
//     double path_x_dot, double path_y_dot,
//     double precomputed_y_e)
// {
//     const double epsilon = 1e-6;
//     if (std::abs(path_x_dot) < epsilon && std::abs(path_y_dot) < epsilon) {
//         double psi_ref = 0.0;
//         double y_e = 0.0;  
//         std::cout << "Warning; derivatives too small in LOS()" << std::endl;
//         return std::make_tuple(psi_ref, y_e);
//     }

//     double psi_p = std::atan2(path_x_dot, path_y_dot);
    
//     // Use pre-computed cross-track error if provided, otherwise calculate it
//     double y_e;
//     if (std::isnan(precomputed_y_e)) {
//         y_e = - (xn - path_x)*std::cos(psi_p) 
//               + (yn - path_y)*std::sin(psi_p);
//     } else {
//         y_e = precomputed_y_e;
//     }

//     double psi_ref = psi_p + std::atan(y_e / delta);
//     psi_ref = ssa(psi_ref);             // wrap to [-π, π]
//     return std::make_tuple(psi_ref, y_e);

// } 

std::tuple<double, double> LOS(double xn, double yn, double delta,
    double path_x, double path_y,
    double path_x_dot, double path_y_dot,
    double precomputed_y_e)
{
    const double epsilon = 1e-6;
    if (std::abs(path_x_dot) < epsilon && std::abs(path_y_dot) < epsilon) {
        double psi_ref = 0.0;
        double y_e = 0.0;
        std::cout << "Warning; derivatives too small in LOS()" << std::endl;
        return std::make_tuple(psi_ref, y_e);
    }

    // 1) Path tangent in ENU convention (0 = East, CCW+)
    const double alpha = std::atan2(path_y_dot, path_x_dot);  // <-- NOTICE: y first, x second

    // 2) Convert to your yaw convention (0 = North, CW+)
    const double psi_p = ssa(M_PI/2 - alpha);

    // 3) Cross-track error (positive to the left of path direction)
    double y_e;
    if (std::isnan(precomputed_y_e)) {
        // Equivalent to: y_e = -(xn - path_x)*sin(alpha) + (yn - path_y)*cos(alpha);
        // (since cos(psi_p)=sin(alpha), sin(psi_p)=cos(alpha))
        y_e = - (xn - path_x)*std::cos(psi_p)
              + (yn - path_y)*std::sin(psi_p);
    } else {
        y_e = precomputed_y_e;
    }

    // 4) LOS heading with robust atan2, then wrap
    double psi_ref = psi_p + std::atan2(y_e, delta);
    psi_ref = ssa(psi_ref);  // wrap to (-π, π]

    return std::make_tuple(psi_ref, y_e);
}
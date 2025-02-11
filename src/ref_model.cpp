#include "ref_model.hpp"
#include "utilities.hpp"  // Assumes ssa() is declared in utilities.hpp
#include <cmath>        // For std::pow, std::abs, and std::copysign

void refModel(double &x_d, double &v_d, double &a_d,
              double x_ref, double v_max, double zeta_d,
              double w_d, double h, bool eulerAngle)
{
    // Compute the error between the current desired position and the commanded position.
    // If the state is an Euler angle, use the smallest signed angle difference.
    double e_x = eulerAngle ? ssa(x_d - x_ref) : (x_d - x_ref);

    // Compute the desired jerk (time derivative of acceleration) according to Fossen (2021, Eq. 12.10)
    double a_d_dot = -std::pow(w_d, 3) * e_x
                     - (2 * zeta_d + 1) * std::pow(w_d, 2) * v_d
                     - (2 * zeta_d + 1) * w_d * a_d;

    // Propagate the states using Euler integration
    x_d = x_d + h * v_d;
    v_d = v_d + h * a_d;
    a_d = a_d + h * a_d_dot;

    // Apply velocity saturation: if |v_d| exceeds v_max, clamp it to ±v_max.
    if (std::abs(v_d) > v_max) {
        v_d = std::copysign(v_max, v_d);
    }
}

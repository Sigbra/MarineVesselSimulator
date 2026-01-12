#include "Control/PID_heading_motion_control.hpp"
#include "Utilities/calculations.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

HeadingPIDController::HeadingPIDController(double T, double wn, double zeta)
    : T_(T), wn_(wn), zeta_(zeta), z_psi_(0.0)
{
}

double HeadingPIDController::update(double h, const Eigen::MatrixXd& M,
                                    double psi, double psi_d, double r, double r_d, double a_d) {

    // Nomoto model: compute the gain constant.
    double K = T_ / M(5,5);

    // Compute PID gains based on the closed-loop design.
    double Kp = M(5,5) * std::pow(wn_, 2);
    double Kd = M(5,5) * (2 * zeta_ * wn_ - 1/T_);
    double Td = Kd / Kp;
    double Ti = 10 / wn_;

    // Compute the heading error.
    double error_psi = ssa(psi - psi_d);

    // Update the integrated heading error.
    z_psi_ += error_psi * h;

    // Compute the control moment (tau_N) with feed-forward terms.
    double tau_N = (T_/K) * a_d + (1/K) * r_d - 
                   Kp * (error_psi + Td * (r - r_d) + (1/Ti) * z_psi_);

    return tau_N;
}

void HeadingPIDController::reset() {
    z_psi_ = 0.0;
}

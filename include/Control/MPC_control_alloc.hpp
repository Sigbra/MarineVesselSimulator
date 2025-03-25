#ifndef MPC_CONTROL_ALLOC_HPP
#define MPC_CONTROL_ALLOC_HPP

#include <vector>
#include <Eigen/Dense>

std::vector<double> MPC_control_alloc(double tau_X, double tau_Y, double tau_N,
    double U, double T_n, double T_alpha,
    Eigen::Vector2d n_input, Eigen::Vector2d alpha_input);

#endif
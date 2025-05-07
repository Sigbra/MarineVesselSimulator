#ifndef MPC_CONTROL_SYSTEM_HPP
#define MPC_CONTROL_SYSTEM_HPP

#include <vector>
#include <Eigen/Dense>

#pragma once

#include <casadi/casadi.hpp>
#include <vector>

using casadi::MX;
using casadi::DM;
using casadi::Opti;
using casadi::OptiSol;
using casadi::Function;

class MPC_Control_System {
public:
    // Constructor: Sets the prediction horizon, timestep, and vessel parameters.
    MPC_Control_System(int N, double dt);

    // Build and solve the MPC problem given the current state (x0 of length 6) and the desired target point (x_d, y_d, psi_d).
    // The problem is constructed, solved, and the solution values over the horizon stored in member vectors.
    bool solve(const std::vector<double>& x0, double x_s, double y_s, double x_d, double y_d, double psi_d, Eigen::VectorXd n_init, Eigen::VectorXd alpha_init, std::vector<bool> failstate);

    Eigen::VectorXd get_n_opt();
    Eigen::VectorXd get_alpha_opt();

private:
    // Dynamics function: Given state X and control tau, returns the state derivative dX/dt.
    // The state X is [x, y, psi, u, v, r] and controls tau is [tau_x, tau_y, tau_N].
    MX f(const MX& X, const MX& tau);

    // Runge-Kutta 4 integrator for propagating the state.
    MX rk4(const MX& Xk, const MX& tau, const MX& dt);

    Function oneStepDynamicsFunction();

    // Prediction horizon and timestep.
    int N;
    double dt;

    // Vessel parameters.
    double m;         // Mass
    double Iz;        // Yaw inertia
    double Xu;        // Surge damping coefficient
    double Yv;        // Sway damping coefficient
    double Npsi;      // Linear yaw damping coefficient
    double Npsi_nl;   // Nonlinear yaw damping coefficient

    // Storage for the optimal control sequence (over the horizon).
    // Each vector has length N.
    Eigen::VectorXd n_opt;
    Eigen::VectorXd alpha_opt;    

    //Warm start variables
    DM X_prev_;
    DM n_cmd_prev_;
    DM alpha_cmd_prev_;
};

#endif
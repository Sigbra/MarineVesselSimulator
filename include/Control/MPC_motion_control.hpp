#ifndef MPC_MOTION_CONTROL_HPP
#define MPC_MOTION_CONTROL_HPP

#include <vector>
#include <Eigen/Dense>

#pragma once

#include <casadi/casadi.hpp>
#include <vector>

using casadi::MX;
using casadi::Opti;
using casadi::OptiSol;

class MPC_Motion_Control {
public:
    // Constructor: Sets the prediction horizon, timestep, and vessel parameters.
    MPC_Motion_Control(int N, double dt, double m, double Iz, double Xu, double Yv, double Npsi, double Npsi_nl);

    // Build and solve the MPC problem given the current state (x0 of length 6) and the desired target point (x_d, y_d, psi_d).
    // The problem is constructed, solved, and the solution values over the horizon stored in member vectors.
    bool solve(const std::vector<double>& x0, double x_d, double y_d, double psi_d);

    // Returns the first optimal control input as a vector: [tau_x, tau_y, tau_N].
    std::vector<double> get_first_tau();

    // Optionally, get full horizon control trajectories.
    std::vector<double> get_tau_x_horizon();
    std::vector<double> get_tau_y_horizon();
    std::vector<double> get_tau_N_horizon();

private:
    // Dynamics function: Given state X and control tau, returns the state derivative dX/dt.
    // The state X is [x, y, psi, u, v, r] and controls tau is [tau_x, tau_y, tau_N].
    MX f(const MX& X, const MX& tau);

    // Runge-Kutta 4 integrator for propagating the state.
    MX rk4(const MX& Xk, const MX& tau, const MX& dt);

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
    std::vector<double> tau_x_sol;
    std::vector<double> tau_y_sol;
    std::vector<double> tau_N_sol;
};

#endif
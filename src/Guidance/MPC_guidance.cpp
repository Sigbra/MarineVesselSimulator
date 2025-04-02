#include "Guidance/MPC_guidance.hpp"
#include "Utilities/calculations.hpp"
#include <iostream>
#include <cmath>

MPCGuidance::MPCGuidance(double h, double r_max, double U_dot_max, double T_final)
    : h_(h), r_max_(r_max), U_dot_max_(U_dot_max), T_final_(T_final),
      solver_initialized_(false)
{
    // Use a fixed 1-second time step for MPC horizon, independent of simulation step h_
    double mpc_step = 1.0; // 1 second between MPC horizon points
 
    // Calculate number of timesteps based on final time and fixed MPC step (not simulation step h_)
    N_ = static_cast<int>(std::round(T_final_ / mpc_step)) + 1;
    if (N_ < 2) {
        N_ = 2; 
    }

    std::cout << "MPC using " << N_ << " timesteps with " << mpc_step << " second intervals" << std::endl;

    // Initialize solution trajectories
    chi_d_trajectory_.resize(N_, 0.0);
    U_d_trajectory_.resize(N_, 0.0);
    X_d_trajectory_.resize(N_, 0.0);
    Y_d_trajectory_.resize(N_, 0.0);

    // Initialize the MPC solver
    initializeSolver();
}

std::tuple<double, double, bool> MPCGuidance::update(double x, double y, double chi, double U, Vector2D wp1, Vector2D wp2) {

    if (!solver_initialized_) {
        std::cerr << "Error: MPC solver not initialized!" << std::endl;
        return std::make_tuple(0.0, 0.0, false);
    }

    // Extract the coordinates of the two waypoints
    double x1 = wp1.x;
    double y1 = wp1.y;
    double x2 = wp2.x;
    double y2 = wp2.y;

    // Compute distance to the first waypoint
    double dist_to_wp1 = std::hypot(x - x1, y - y1);

    // Set final waypoint for MPC to track
    double x_final = x2;
    double y_final = y2;

    // Initial speed and heading (use previous solution if available)
    double init_speed = U;
    double init_heading = chi;

    // If first iteration, initialize with reasonable values
    if (init_speed < 0.1) {
        init_speed = 5.0; // Default initial speed
    }
    if (std::abs(init_heading) < 0.001) {
        init_heading = std::atan2(y_final - y, x_final - x);
    }

    // Set up parameters for MPC
    std::vector<double> p = {x, y, x_final, y_final, init_heading, init_speed};
 
    // Initial guess (warm start with previous solution if available)
    std::vector<double> x0(4 * N_);
    for (int i = 0; i < N_; ++i) {
 
        double t = static_cast<double>(i) / (N_ - 1); // Normalized time [0,1]
        x0[2*N_ + i] = x + t * (x_final - x);
        x0[3*N_ + i] = y + t * (y_final - y);
        x0[i] = init_heading;
        x0[N_ + i] = init_speed;
 
        // Use previous solution for warm start if available
        if (!chi_d_trajectory_.empty() && chi_d_trajectory_.size() == N_) {
            x0[i] = chi_d_trajectory_[i];
            x0[N_ + i] = U_d_trajectory_[i];
            x0[2*N_ + i] = X_d_trajectory_[i];
            x0[3*N_ + i] = Y_d_trajectory_[i];
        }
    }
 
    // Solve the MPC problem
    try {
        // Create vectors for bounds first
        std::vector<double> lbx(4*N_, -1e6);  // Lower bounds
        std::vector<double> ubx(4*N_, 1e6);   // Upper bounds

        // Constraints on chi_d and U_d
        for (int i = 0; i < N_; ++i) {
            lbx[i] = -M_PI;
            ubx[i] = M_PI;
            lbx[N_ + i] = -10.0;
            ubx[N_ + i] =  10.0;
        }
 
        // Pass the vectors to the solver
        casadi::DMDict arg = {{"x0", x0}, {"p", p}, {"lbx", lbx}, {"ubx", ubx}};

        // Solve NLP
        casadi::DMDict res = solver_(arg);

        // Extract solution
        std::vector<double> solution = res.at("x").get_elements();

        // Store the trajectories
        for (int i = 0; i < N_; ++i) {
            chi_d_trajectory_[i] = solution[i];
            U_d_trajectory_[i] = solution[N_ + i];
            X_d_trajectory_[i] = solution[2*N_ + i];
            Y_d_trajectory_[i] = solution[3*N_ + i];
        }

        // Return the first control action. 
        return std::make_tuple(chi_d_trajectory_[1], U_d_trajectory_[1], false);
 
    } catch (const std::exception& e) {

        std::cerr << "Error solving MPC: " << e.what() << std::endl;
        return std::make_tuple(init_heading, init_speed, false);
    }
}

void MPCGuidance::reset() {
    std::fill(chi_d_trajectory_.begin(), chi_d_trajectory_.end(), 0.0);
    std::fill(U_d_trajectory_.begin(), U_d_trajectory_.end(), 0.0);
    std::fill(X_d_trajectory_.begin(), X_d_trajectory_.end(), 0.0);
    std::fill(Y_d_trajectory_.begin(), Y_d_trajectory_.end(), 0.0);
    std::cout << "MPC: State reset" << std::endl;
}

void MPCGuidance::initializeSolver() {

    using namespace casadi;

    // Fixed MPC time step of 1 second (independent of simulation time step h_)
    double mpc_step = 1.0;
 
    // State variables
    MX chi_d = MX::sym("chi_d", N_);    // Course over ground
    MX U_d = MX::sym("U_d", N_);        // Speed over ground
    MX X_d = MX::sym("X_d", N_);        // X position
    MX Y_d = MX::sym("Y_d", N_);        // Y position
 
    // Parameters
    MX x_init = MX::sym("x_init");      // Initial x position
    MX y_init = MX::sym("y_init");      // Initial y position
    MX x_final = MX::sym("x_final");    // Final x position
    MX y_final = MX::sym("y_final");    // Final y position
    MX init_heading = MX::sym("init_heading"); // Initial heading
    MX init_speed = MX::sym("init_speed");     // Initial speed

    // Cost function: minimize sum of squared velocities
    MX cost = 0;
 
    // Constraints
    std::vector<MX> g;
    std::vector<double> lbg, ubg;
 
    // Initial conditions
    g.push_back(X_d(0) - x_init);
    lbg.push_back(0);
    ubg.push_back(0);
 
    g.push_back(Y_d(0) - y_init);
    lbg.push_back(0);
    ubg.push_back(0);
 
    g.push_back(chi_d(0) - init_heading);
    lbg.push_back(0);
    ubg.push_back(0);

    g.push_back(U_d(0) - init_speed);
    lbg.push_back(0);
    ubg.push_back(0);
 
    // Final conditions
    g.push_back(X_d(N_-1) - x_final);
    lbg.push_back(0);
    ubg.push_back(0);

    g.push_back(Y_d(N_-1) - y_final);
    lbg.push_back(0);
    ubg.push_back(0);
 
    // Dynamics and rate constraints for each time step
    for (int k = 0; k < N_-1; ++k) {
 
        // Dynamics: X_d(k+1) = X_d(k) + mpc_step * U_d(k) * cos(chi_d(k))
        g.push_back(X_d(k+1) - X_d(k) - mpc_step * U_d(k) * cos(chi_d(k)));
        lbg.push_back(0);
        ubg.push_back(0);

        // Dynamics: Y_d(k+1) = Y_d(k) + mpc_step * U_d(k) * sin(chi_d(k))
        g.push_back(Y_d(k+1) - Y_d(k) - mpc_step * U_d(k) * sin(chi_d(k)));
        lbg.push_back(0);
        ubg.push_back(0);

        // Rate constraint on heading change
        g.push_back((chi_d(k+1) - chi_d(k)) / mpc_step);
        lbg.push_back(-r_max_);
        ubg.push_back(r_max_);

        // Rate constraint on speed change
        g.push_back((U_d(k+1) - U_d(k)) / mpc_step);
        lbg.push_back(-U_dot_max_);
        ubg.push_back(U_dot_max_);
 
        // Cost: minimize sum of squared velocities
        cost += pow(U_d(k) * cos(chi_d(k)), 2) + pow(U_d(k) * sin(chi_d(k)), 2);
    }
 
    cost += pow(U_d(N_-1) * cos(chi_d(N_-1)), 2) + pow(U_d(N_-1) * sin(chi_d(N_-1)), 2);
 
    // Bounds for all variables
    std::vector<double> lbx(4 * N_), ubx(4 * N_);
    for (int i = 0; i < N_; ++i) {

        // chi_d bounds: -pi to pi
        lbx[i] = -M_PI;
        ubx[i] = M_PI;

        // U_d bounds: 0 to 10 m/s (adjust as needed)
        lbx[N_ + i] = 0;
        ubx[N_ + i] = 10.0;

        // X_d and Y_d bounds: large values for now
        lbx[2*N_ + i] = -1e6;
        ubx[2*N_ + i] = 1e6;
        lbx[3*N_ + i] = -1e6;
        ubx[3*N_ + i] = 1e6;
    }
 
    // Decision variables (all states stacked)
    MX X = vertcat(chi_d, U_d, X_d, Y_d);
 
    // Parameters (stacked)
    MX P = vertcat(x_init, y_init, x_final, y_final, init_heading, init_speed);

    // NLP formulation
    MXDict nlp = {{"x", X}, {"p", P}, {"f", cost}, {"g", vertcat(g)}};
 
    // Solver options
    Dict opts;
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = 0;
    opts["ipopt.max_iter"] = 1000;

    // Create solver
    solver_ = nlpsol("solver", "ipopt", nlp, opts);

    // Create initial guess
    std::vector<double> x0(4 * N_);

    // Store the solver data
    solver_initialized_ = true;
    std::cout << "MPC solver initialized with " << N_ << " steps over " << T_final_ << " seconds." << std::endl;
} 
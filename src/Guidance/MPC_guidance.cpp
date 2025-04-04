#include "Guidance/MPC_guidance.hpp"
#include "Utilities/calculations.hpp"
#include <iostream>
#include <cmath>

#include "casadi/casadi.hpp"
#include <vector>
#include <cmath>
#include <iostream>

using namespace casadi;

MPCGuidance::MPCGuidance(double r_max, double U_dot_max)
                        : r_max_(r_max), U_dot_max_(U_dot_max) {}

MX MPCGuidance::f(const MX& X, const MX& C) {
    MX dX = MX::zeros(2, 1);
    dX(0) = C(1) * cos(C(0)); 
    dX(1) = C(1) * sin(C(0)); 
    return dX;
};

MX MPCGuidance::rk4(const MX& Xk, const MX& Ck, const MX& dt) {
    MX k1 = f(Xk, Ck);
    MX k2 = f(Xk + (dt / 2) * k1, Ck);
    MX k3 = f(Xk + (dt / 2) * k2, Ck);
    MX k4 = f(Xk + dt * k3, Ck);
    return Xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

void MPCGuidance::update(double h, double x, double y, double chi, double U,
                               Vector2D wpt_prev, Vector2D wpt_goal) {
    // Initial conditions
    double x_init = x;
    double y_init = y;
    double U_init = U;
    double chi_init = chi;

    // Final conditions
    MX X_final   = wpt_goal.x;
    MX Y_final   = wpt_goal.y;
    MX U_final   = 0;
    MX chi_final = std::atan2(wpt_goal.y - wpt_prev.y, wpt_goal.x - wpt_prev.x);
    
    MX r_max = r_max_;
    MX U_dot_max = U_dot_max_;

    Opti opti; 

    int N = 60; 
    int T = 30;
    MX dt = 0.5; //MX(T)/MX(N);  

    int n_states   = 2; // x, y
    int n_controls = 2; // chi, U

    // States init
    MX X = opti.variable(n_states, N+1);
    opti.set_initial(X, repmat(DM::vertcat({x, y}), 1, N+1));
    opti.subject_to(X(Slice(), 0) == MX::vertcat({MX(x), MX(y)}));

    // Controls init
    MX C = opti.variable(n_controls, N);
    opti.set_initial(C, repmat(DM::vertcat({chi, U}), 1, N));

    for (int i = 0; i < N; i++){
        MX X_next = rk4(X(Slice(),i), C(Slice(),i), dt);
        opti.subject_to(X(Slice(),i+1) == X_next);
    }

    // This does not work?
    //opti.subject_to( abs(C(0, Slice(1,N)) - C(0, Slice(0,N-1))) / dt <= r_max);
    //opti.subject_to( abs(C(1, Slice(1,N)) - C(1, Slice(0,N-1))) / dt <= U_dot_max);
    // This works
    for (int i = 0; i < N-1; i++){
        opti.subject_to( abs(C(0, i+1)) - abs(C(0, i)) / dt <= r_max);
        opti.subject_to( abs(C(1, i+1)) - abs(C(1, i)) / dt <= U_dot_max);
    }

    // Final conditions
    opti.subject_to(X(Slice(), N) == DM::vertcat({DM(X_final), DM(Y_final)}));

    MX pos_error = X - repmat(DM::vertcat({DM(X_final), DM(Y_final)}), 1, N+1);
    opti.minimize(sumsqr(pos_error));


    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 5000;  
    // solver_opts["ipopt.tol"] = 0.001;       
    // solver_opts["ipopt.acceptable_tol"] = 0.01;
    // solver_opts["ipopt.acceptable_iter"] = 20;

    try {

        opti.solver("ipopt", solver_opts);
        OptiSol sol = opti.solve();  

        chi_d_.clear();
        U_d_.clear();
        X_d_.clear();
        Y_d_.clear();

        chi_d_.resize(N);
        U_d_.resize(N);
        X_d_.resize(N+1);
        Y_d_.resize(N+1);

        for (int i = 0; i < N; i++) {
            chi_d_[i] = double(sol.value(C(0, i))); 
            U_d_[i]   = double(sol.value(C(1, i))); 
        }
        for (int i = 0; i <= N; i++) {
            X_d_[i] = double(sol.value(X(0, i))); 
            Y_d_[i] = double(sol.value(X(1, i))); 
        }

    } catch (const std::exception &e) {
        std::cerr << "Warning: Guidance MPC - Optimization failed: " << e.what() << std::endl;
    }
}

double MPCGuidance::get_chi_d(int idx) const {
    if (idx >= 0 && idx < chi_d_.size()) {
        return chi_d_[idx];
    }
    std::abort();
    return chi_d_[1]; 
}

double MPCGuidance::get_U_d(int idx) const {
    if (idx >= 0 && idx < U_d_.size()) {
        return U_d_[idx];
    }
    std::abort();
    return U_d_[1];
}

double MPCGuidance::get_X_d(int idx) const {
    if (idx >= 0 && idx < X_d_.size()) {
        return X_d_[idx];
    }
    std::abort();
    return chi_d_[1]; 
}

double MPCGuidance::get_Y_d(int idx) const {
    if (idx >= 0 && idx < Y_d_.size()) {
        return Y_d_[idx];
    }
    std::abort();
    return chi_d_[1]; 
}


// MPCGuidance::MPCGuidance(double h, double r_max, double U_dot_max, double T_final)
//     : h_(h), r_max_(r_max), U_dot_max_(U_dot_max), T_final_(T_final),
//       solver_initialized_(false)
// {
//     // Use a fixed 1-second time step for MPC horizon, independent of simulation step h_
//     double mpc_step = 1.0; // 1 second between MPC horizon points
 
//     // Calculate number of timesteps based on final time and fixed MPC step (not simulation step h_)
//     N_ = static_cast<int>(std::round(T_final_ / mpc_step)) + 1;
//     if (N_ < 2) {
//         N_ = 2; 
//     }

//     std::cout << "MPC using " << N_ << " timesteps with " << mpc_step << " second intervals" << std::endl;

//     // Initialize solution trajectories
//     chi_d_trajectory_.resize(N_, 0.0);
//     U_d_trajectory_.resize(N_, 0.0);
//     X_d_trajectory_.resize(N_, 0.0);
//     Y_d_trajectory_.resize(N_, 0.0);

//     // Initialize the MPC solver
//     initializeSolver();
// }

// std::tuple<double, double, bool> MPCGuidance::update(double x, double y, double chi, double U, Vector2D wpt_prev, Vector2D wpt_goal) {

//     // Initial 
//     double x_init = x;
//     double y_init = y;
//     double U_init = U;
//     double chi_init = chi;

//     // Final 
//     double x_final = wpt_goal.x;
//     double y_final = wpt_goal.y;
//     double U_final = 0;
//     double chi_final = std::atan2(y_final - wpt_prev.y, x_final - wpt_prev.x);

//     // Set up parameters for MPC
//     std::vector<double> p = {chi_init, U_init, x, y, chi_final, U_final, x_final, y_final};
 
//     // Initial guess (warm start with previous solution if available)
//     std::vector<double> x0(4 * N_);
//     for (int i = 0; i < N_; ++i) {
 
//         double t = static_cast<double>(i) / (N_ - 1); // Normalized time [0,1]
//         x0[2*N_ + i] = x + t * (x_final - x);
//         x0[3*N_ + i] = y + t * (y_final - y);
//         x0[i] = chi_init;
//         x0[N_ + i] = U_init;
 
//         // Use previous solution for warm start if available
//         if (!chi_d_trajectory_.empty() && chi_d_trajectory_.size() == N_) {
//             x0[i] = chi_d_trajectory_[i];
//             x0[N_ + i] = U_d_trajectory_[i];
//             x0[2*N_ + i] = X_d_trajectory_[i];
//             x0[3*N_ + i] = Y_d_trajectory_[i];
//         }
//     }
 
//     // Solve the MPC problem
//     try {
//         // Create vectors for bounds first
//         std::vector<double> lbx(4*N_, -1e6);  // Lower bounds
//         std::vector<double> ubx(4*N_, 1e6);   // Upper bounds

//         // Constraints on chi_d and U_d
//         for (int i = 0; i < N_; ++i) {
//             lbx[i] = -M_PI;
//             ubx[i] = M_PI;
//             //Want to go slow in DP mode
//             lbx[N_ + i] = -2.0;
//             ubx[N_ + i] =  2.0;
//         }
 
//         // Pass the vectors to the solver
//         casadi::DMDict arg = {{"x0", x0}, {"p", p}, {"lbx", lbx}, {"ubx", ubx}};

//         // Solve NLP
//         casadi::DMDict res = solver_(arg);

//         // Extract solution
//         std::vector<double> solution = res.at("x").get_elements();

//         // Store the trajectories
//         for (int i = 0; i < N_; ++i) {
//             chi_d_trajectory_[i] = solution[i];
//             U_d_trajectory_[i] = solution[N_ + i];
//             X_d_trajectory_[i] = solution[2*N_ + i];
//             Y_d_trajectory_[i] = solution[3*N_ + i];
//         }

//         // Return the first control action. 
//         return std::make_tuple(chi_d_trajectory_[1], U_d_trajectory_[1], false);
 
//     } catch (const std::exception& e) {

//         std::cerr << "Error solving MPC: " << e.what() << std::endl;
//         return std::make_tuple(init_heading, init_speed, false);
//     }
// }

// void MPCGuidance::reset() {
//     std::fill(chi_d_trajectory_.begin(), chi_d_trajectory_.end(), 0.0);
//     std::fill(U_d_trajectory_.begin(), U_d_trajectory_.end(), 0.0);
//     std::fill(X_d_trajectory_.begin(), X_d_trajectory_.end(), 0.0);
//     std::fill(Y_d_trajectory_.begin(), Y_d_trajectory_.end(), 0.0);
//     std::cout << "MPC: State reset" << std::endl;
// }

// void MPCGuidance::initializeSolver() {

//     using namespace casadi;

//     // Fixed MPC time step of 1 second (independent of simulation time step h_)
//     double mpc_step = 1.0;
 
//     // State variables
//     MX chi_d = MX::sym("chi_d", N_);    // Course over ground
//     MX U_d = MX::sym("U_d", N_);        // Speed over ground
//     MX X_d = MX::sym("X_d", N_);        // X position
//     MX Y_d = MX::sym("Y_d", N_);        // Y position
 
//     // Parameters
//     MX x_init = MX::sym("x_init");      // Initial x position
//     MX y_init = MX::sym("y_init");      // Initial y position
//     MX x_final = MX::sym("x_final");    // Final x position
//     MX y_final = MX::sym("y_final");    // Final y position
//     MX init_heading = MX::sym("init_heading"); // Initial heading
//     MX init_speed = MX::sym("init_speed");     // Initial speed

//     // Cost function: minimize sum of squared velocities
//     MX cost = 0;
 
//     // Constraints
//     std::vector<MX> g;
//     std::vector<double> lbg, ubg;
 
//     // Initial conditions
//     g.push_back(X_d(0) - x_init);
//     lbg.push_back(0);
//     ubg.push_back(0);
 
//     g.push_back(Y_d(0) - y_init);
//     lbg.push_back(0);
//     ubg.push_back(0);
 
//     g.push_back(chi_d(0) - init_heading);
//     lbg.push_back(0);
//     ubg.push_back(0);

//     g.push_back(U_d(0) - init_speed);
//     lbg.push_back(0);
//     ubg.push_back(0);
 
//     // Final conditions
//     g.push_back(X_d(N_-1) - x_final);
//     lbg.push_back(0);
//     ubg.push_back(0);

//     g.push_back(Y_d(N_-1) - y_final);
//     lbg.push_back(0);
//     ubg.push_back(0);
 
//     // Dynamics and rate constraints for each time step
//     for (int k = 0; k < N_-1; ++k) {
 
//         // Dynamics: X_d(k+1) = X_d(k) + mpc_step * U_d(k) * cos(chi_d(k))
//         g.push_back(X_d(k+1) - X_d(k) - mpc_step * U_d(k) * cos(chi_d(k)));
//         lbg.push_back(0);
//         ubg.push_back(0);

//         // Dynamics: Y_d(k+1) = Y_d(k) + mpc_step * U_d(k) * sin(chi_d(k))
//         g.push_back(Y_d(k+1) - Y_d(k) - mpc_step * U_d(k) * sin(chi_d(k)));
//         lbg.push_back(0);
//         ubg.push_back(0);

//         // Rate constraint on heading change
//         g.push_back((chi_d(k+1) - chi_d(k)) / mpc_step);
//         lbg.push_back(-r_max_);
//         ubg.push_back(r_max_);

//         // Rate constraint on speed change
//         g.push_back((U_d(k+1) - U_d(k)) / mpc_step);
//         lbg.push_back(-U_dot_max_);
//         ubg.push_back(U_dot_max_);
 
//         // Cost: minimize sum of squared velocities
//         cost += pow(U_d(k) * cos(chi_d(k)), 2) + pow(U_d(k) * sin(chi_d(k)), 2);
//     }
 
//     cost += pow(U_d(N_-1) * cos(chi_d(N_-1)), 2) + pow(U_d(N_-1) * sin(chi_d(N_-1)), 2);
 
//     // Bounds for all variables
//     std::vector<double> lbx(4 * N_), ubx(4 * N_);
//     for (int i = 0; i < N_; ++i) {

//         // chi_d bounds: -pi to pi
//         lbx[i] = -M_PI;
//         ubx[i] = M_PI;

//         // U_d bounds: 0 to 10 m/s (adjust as needed)
//         lbx[N_ + i] = 0;
//         ubx[N_ + i] = 10.0;

//         // X_d and Y_d bounds: large values for now
//         lbx[2*N_ + i] = -1e6;
//         ubx[2*N_ + i] = 1e6;
//         lbx[3*N_ + i] = -1e6;
//         ubx[3*N_ + i] = 1e6;
//     }
 
//     // Decision variables (all states stacked)
//     MX X = vertcat(chi_d, U_d, X_d, Y_d);
 
//     // Parameters (stacked)
//     MX P = vertcat(x_init, y_init, x_final, y_final, init_heading, init_speed);

//     // NLP formulation
//     MXDict nlp = {{"x", X}, {"p", P}, {"f", cost}, {"g", vertcat(g)}};
 
//     // Solver options
//     Dict opts;
//     opts["ipopt.print_level"] = 0;
//     opts["print_time"] = 0;
//     opts["ipopt.max_iter"] = 1000;

//     // Create solver
//     solver_ = nlpsol("solver", "ipopt", nlp, opts);

//     // Create initial guess
//     std::vector<double> x0(4 * N_);

//     // Store the solver data
//     solver_initialized_ = true;
//     std::cout << "MPC solver initialized with " << N_ << " steps over " << T_final_ << " seconds." << std::endl;
// } 
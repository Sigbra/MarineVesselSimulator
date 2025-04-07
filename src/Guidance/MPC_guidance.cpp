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

std::tuple<double, double, double, double> MPCGuidance::update(double h, double x, double y, double chi, double U,
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

    int N = 40; 
    int T = 20;
    MX dt = static_cast<double>(T) / static_cast<double>(N);  

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

    // Constraints
    // |chi_k+1 - chi_k| / dt <= r_max
    opti.subject_to( (C(0, Slice(1,N)) - C(0, Slice(0,N-1))) <= r_max * dt);
    opti.subject_to( (C(0, Slice(1,N)) - C(0, Slice(0,N-1))) >= - r_max * dt);
    // |U_k+1 - U_k| / dt <= U_dot_max 
    opti.subject_to( (C(1, Slice(1,N)) - C(1, Slice(0,N-1))) <= U_dot_max * dt);
    opti.subject_to( (C(1, Slice(1,N)) - C(1, Slice(0,N-1))) >= - U_dot_max * dt);
    //  -2 km/h <= U <= 2 km/h
    opti.subject_to( C(1, Slice(0,N-1)) <= 2);
    opti.subject_to( C(1, Slice(0,N-1)) >= -2); 

    // Final conditions
    opti.subject_to(X(Slice(), N) == DM::vertcat({DM(X_final), DM(Y_final)}));
    opti.subject_to(C(0, N-1) == chi_final);
    opti.subject_to(C(1, N-1) == U_final);

    // Objective function
    MX pos_error = X - repmat(DM::vertcat({DM(X_final), DM(Y_final)}), 1, N+1);
    opti.minimize(sumsqr(pos_error));

    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 1000;  
    solver_opts["ipopt.tol"] = 0.0001;       
    solver_opts["ipopt.acceptable_tol"] = 0.001;
    solver_opts["ipopt.acceptable_iter"] = 100;

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

        return std::make_tuple(chi_d_[1], U_d_[1], X_d_[1], Y_d_[1]); 

    } catch (const std::exception &e) {
        std::cerr << "Warning: Guidance MPC - Optimization failed: " << e.what() << std::endl;
        // Returns same orientation and position but with 0 speed for safety
        return std::make_tuple(chi_init, 0, x_init, y_init); 
    }
}

#include <casadi/casadi.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include "Control/non_lin_constrained_control_alloc.hpp"
#include "Models/model_utilities.hpp"
#include "Utilities/calculations.hpp"

using namespace casadi;

std::vector<double> NLOptControlAlloc(double tau_X, double tau_Y, double tau_N, double U, Eigen::Vector2d n, Eigen::Vector2d alpha) {

    // Lever arms from ran()
    Eigen::Vector3d CO_offset = CO_Offset(U);
    double ly1 =  1.1 - CO_offset(1);    
    double ly2 = -1.1 + CO_offset(1);    
    double lx  = -1.1 - CO_offset(0);    

    // Constants from ran()
    double g = 9.81;
    double k_pos = 200*g;         
    double k_neg = 200*g;         
    double n_max =  1;           
    double n_min = -1;            
    double alpha_max = M_PI/2; 
    double alpha_min = -M_PI/2;

    Opti opti;

    // n1, alpha1, n2, alpha2
    int n_effectors = 4;
    MX vars = opti.variable(n_effectors);
    // Objective function
    MX J = 0;
    double tau_weight = 10.0; 
    // Small randomness to avoid local minima
    double rand_small = ((double)rand() / RAND_MAX - 0.5) * 0.01;

    opti.set_initial(vars(0), n(0) + rand_small);
    opti.set_initial(vars(1), alpha(0));
    opti.set_initial(vars(2), n(1) + rand_small);
    opti.set_initial(vars(3), alpha(1));

    opti.subject_to(vars(0) >= n_min);
    opti.subject_to(vars(0) <= n_max);
    opti.subject_to(vars(1) >= alpha_min);
    opti.subject_to(vars(1) <= alpha_max);

    opti.subject_to(vars(2) >= n_min);
    opti.subject_to(vars(2) <= n_max);
    opti.subject_to(vars(3) >= alpha_min);
    opti.subject_to(vars(3) <= alpha_max);


    MX Thrust1 = if_else(vars(0) >= 0, k_pos * vars(0) * abs(vars(0)), k_neg * vars(0) * abs(vars(0)));
    MX Thrust2 = if_else(vars(2) >= 0, k_pos * vars(2) * abs(vars(2)), k_neg * vars(2) * abs(vars(2)));

    MX tau_X_model = Thrust1 * cos(vars(1)) + Thrust2 * cos(vars(3));
    MX tau_Y_model = Thrust1 * sin(vars(1)) + Thrust2 * sin(vars(3));
    MX tau_N_model = lx * (Thrust1*sin(vars(1)) + Thrust2*sin(vars(3)))
                    -(ly1*Thrust1*cos(vars(1)) + ly2*Thrust2*cos(vars(3)));
    
    J = 10 * (pow(tau_X - tau_X_model, 2) +
              pow(tau_Y - tau_Y_model, 2) +
              pow(tau_N - tau_N_model, 2));

    // - Penalty for both pods forward, leading to loss of sway control.
    // MX a1 = exp( -pow( abs(vars(1)), 2 ) / 0.1 ); 
    // MX a2 = exp( -pow( abs(vars(2)), 2 ) / 0.1 );
    // J += 5 * a1 * a2; // X * a1_max * a2_max = X
    
    // - Penalties for directing thrust into another pod slip stream 
    //   (effect not captured by the current ran model, but on the real vessel)
    MX b1 = exp( -pow((vars(1) + M_PI/2), 2) / 0.2 ); 
    J += 5 * b1; 

    MX b2 = exp( -pow((vars(3) - M_PI/2), 2) / 0.2 ); 
    J += 5 * b2;

    // Penalties for pods beeing +90 or -90 at the same time,
    // leading to loss of surge control because of slowly time variying dynamics
    // not captured by this optimalization method.
    MX d1 = exp( -pow(abs(vars(1)) - M_PI/2, 2) / 0.1 );
    MX d2 = exp( -pow(abs(vars(3)) - M_PI/2, 2) / 0.1 );
    J += 10 * d1 * d2;

    opti.minimize(J);

    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 200;  
    solver_opts["ipopt.tol"] = 0.0001;       
    solver_opts["ipopt.acceptable_tol"] = 0.001;
    solver_opts["ipopt.acceptable_iter"] = 100;
    opti.solver("ipopt", solver_opts);

    try {
        OptiSol sol = opti.solve();
        DM vars_sol = opti.value(vars);

        double n1_opt     = double(vars_sol(0));
        double alpha1_opt = double(vars_sol(1));
        double n2_opt     = double(vars_sol(2));
        double alpha2_opt = double(vars_sol(3));

        return {n1_opt, alpha1_opt, n2_opt, alpha2_opt};

    } catch (std::exception &e) {
       
        std::cerr << "Optimization failed: " << e.what() << std::endl;
        return {0.0, 0.0, 0.0, 0.0};
    }
}
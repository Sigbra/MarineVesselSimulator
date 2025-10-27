#include <casadi/casadi.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include "Control/non_lin_constrained_control_alloc.hpp"
#include "Models/model_utilities.hpp"
#include "Models/ran.hpp"
#include "Utilities/calculations.hpp"

using namespace casadi;

std::vector<double> NLOptControlAlloc(double tau_X, double tau_Y, double tau_N, double U, Eigen::Vector2d n, Eigen::Vector2d alpha, std::vector<bool> failstate) {

    // Lever arms from ran()
    double ly1_o = -0.79;
    double ly2_o =  0.79;
    double lx_o  = -1.17;
    // Eigen::Vector3d CO_offset = CO_Offset(U);
    // ly1_o -= CO_offset(1);    
    // ly2_o += CO_offset(1);    
    // lx_o  -= CO_offset(0);
    double pod_radius = 0.2;     

    // Constants from ran()     
    double n_max =  0.70;           
    double n_min = -0.70;            
    double alpha_max = M_PI/2; 
    double alpha_min = -M_PI/2;   

    Opti opti;

    // n1, alpha1, n2, alpha2
    int n_effectors = 4;
    MX vars = opti.variable(n_effectors);
    // Objective function
    MX J = 0;
    // Small randomness to avoid local minima
    double rand_small = ((double)rand() / RAND_MAX - 0.5) * 0.001;

    opti.set_initial(vars(0), n(0) + rand_small);
    opti.set_initial(vars(1), alpha(0));
    opti.set_initial(vars(2), n(1) + rand_small);
    opti.set_initial(vars(3), alpha(1));

    if (failstate[0]){
        opti.subject_to(vars(0) == 0);
        opti.subject_to(vars(1) == alpha(0));
    } else {
        opti.subject_to(vars(0) >= n_min);
        opti.subject_to(vars(0) <= n_max);
        opti.subject_to(vars(1) >= alpha_min);
        opti.subject_to(vars(1) <= alpha_max);
    }

    if (failstate[1]){
        opti.subject_to(vars(2) == 0);
        opti.subject_to(vars(3) == alpha(1));
    } else {
        opti.subject_to(vars(2) >= n_min);
        opti.subject_to(vars(2) <= n_max);
        opti.subject_to(vars(3) >= alpha_min);
        opti.subject_to(vars(3) <= alpha_max);
    }

    MX alpha1 = vars(1);
    MX alpha2 = vars(3);
    
    MX lx1  = lx_o - pod_radius * cos(alpha1);
    MX lx2  = lx_o - pod_radius * cos(alpha2);
    MX ly1 = ly1_o - pod_radius * sin(alpha1);
    MX ly2 = ly2_o - pod_radius * sin(alpha2);

    MX Thrust1 = ThrustFromRelativeN_MX(vars(0));
    MX Thrust2 = ThrustFromRelativeN_MX(vars(2));

    MX tau_X_model = Thrust1 * cos(alpha1) + Thrust2 * cos(alpha2);
    MX tau_Y_model = Thrust1 * sin(alpha1) + Thrust2 * sin(alpha2);
    MX tau_N_model = lx1 * Thrust1 * sin(alpha1) + lx2 * Thrust2 * sin(alpha2)
                   + ly1 * Thrust1 * cos(alpha1) + ly2 * Thrust2 * cos(alpha2);
    
    J = 0.5 * (pow(tau_X - tau_X_model, 2) +
               pow(tau_Y - tau_Y_model, 2) +
<<<<<<< HEAD
               pow(tau_N - tau_N_model, 2));
=======
               5*pow(tau_N - tau_N_model, 2));
>>>>>>> 35-pirnn-observer

    // - Penalty for both pods forward, leading to loss of sway control.
    // MX a1 = exp( -pow( abs(vars(1)), 2 ) / 0.1 ); 
    // MX a2 = exp( -pow( abs(vars(2)), 2 ) / 0.1 );
    // J += 5 * a1 * a2; // X * a1_max * a2_max = X
    
    // - Penalties for directing thrust into another pod slip stream 
    //   (effect not captured by the current ran model, but on the real vessel)
    //MX b1 = exp( -pow((vars(1) + M_PI/2), 2) / 0.2 ); 
    //J += 5 * b1; 

    //MX b2 = exp( -pow((vars(3) - M_PI/2), 2) / 0.2 ); 
    //J += 5 * b2;

    // Penalties for pods beeing +90 or -90 at the same time,
    // leading to loss of surge control because of slowly time variying dynamics
    // not captured by this optimalization method.
    //MX d1 = exp( -pow(abs(vars(1)) - M_PI/2, 2) / 0.1 );
    //MX d2 = exp( -pow(abs(vars(3)) - M_PI/2, 2) / 0.1 );
    //J += 10 * d1 * d2;

    // Penalty for large changes in propeller speed and azimuth angle
    MX d_n1 = vars(0) - n(0);
    MX d_n2 = vars(2) - n(1);
    MX d_alpha1 = alpha1 - alpha(0);
    MX d_alpha2 = alpha2 - alpha(1);
    J += 8*(dot(d_n1,d_n1) + dot(d_n1,d_n1)) 
         + 1*(dot(d_alpha1,d_alpha1) + dot(d_alpha2,d_alpha2)); 

    // Prefer azimuths centered around 0
    J += 0.5*(dot(alpha1,alpha1) + dot(alpha2, alpha2));

    opti.minimize(J);

    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 1000;  
    solver_opts["ipopt.tol"] = 0.001;       
    solver_opts["ipopt.acceptable_tol"] = 0.01;
    solver_opts["ipopt.acceptable_iter"] = 200;
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
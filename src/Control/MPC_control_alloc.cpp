#include <vector>
#include <Eigen/Dense>
#include "Models/ran.hpp"
#include "Models/model_utilities.hpp"
#include <casadi/casadi.hpp>

using namespace casadi;

std::vector<double> MPC_control_alloc(double tau_X, double tau_Y, double tau_N,
    double U, double T_n, double T_alpha,
    Eigen::Vector2d n_input, Eigen::Vector2d alpha_input, std::vector<bool> failstate) 
{
    double horizon = 4.0;
    double delta   = 0.2;
    int N = static_cast<int>(horizon/delta); 

    //Eigen::Vector3d CO_offset = CO_Offset(U);
    //double ly1 =  0.79 - CO_offset(1);    // Left pod lever arm
    //double ly2 = -0.79 + CO_offset(1);    // Right pod lever arm
    //double lx  = -1.17 - CO_offset(0);    // Pod locations in x
    double ly1 =  0.79;    
    double ly2 = -0.79;    
    double lx  = -1.17;  

    // Constants from ran()    
    double n_max =  0.75;           
    double n_min = -0.75;            
    double alpha_max = M_PI/2; 
    double alpha_min = -M_PI/2; 

    Opti opti;

    MX n_vars = opti.variable(2, N+1);
    MX alpha_vars = opti.variable(2, N+1);

    MX n_cmd = opti.variable(2, N);
    MX alpha_cmd = opti.variable(2, N);

    // Set initial guesses for optimization variables
    // opti.set_initial(n_vars, repmat(DM::vertcat({n_input(0), n_input(1)}), 1, N+1));
    // opti.set_initial(alpha_vars, repmat(DM::vertcat({alpha_input(0), alpha_input(1)}), 1, N+1));

    double rand_small = ((double)rand() / RAND_MAX - 0.5) * 0.001; 

    opti.set_initial(n_cmd, repmat(DM::vertcat({n_input(0) + rand_small, n_input(1) + rand_small}), 1, N));
    opti.set_initial(alpha_cmd, repmat(DM::vertcat({alpha_input(0), alpha_input(1)}), 1, N));
    
    
    opti.subject_to(n_vars(Slice(), 0) ==
    DM::vertcat(std::vector<DM>{DM(n_input(0)), DM(n_input(1))}));
    opti.subject_to(alpha_vars(Slice(), 0) ==
    DM::vertcat(std::vector<DM>{DM(alpha_input(0)), DM(alpha_input(1))}));

    // Constraint on change in n and alpha
    for (int k = 0; k < N; ++k) {
        MX n_next = n_vars(Slice(), k) + (delta/T_n) * (n_cmd(Slice(), k) - n_vars(Slice(), k));
        opti.subject_to( n_vars(Slice(), k+1) == n_next );

        MX alpha_next = alpha_vars(Slice(), k) + (delta/T_alpha) * (alpha_cmd(Slice(), k) - alpha_vars(Slice(), k));
        opti.subject_to( alpha_vars(Slice(), k+1) == alpha_next );
    }

    // Constraints on maximum and minimum values
    // for (int k = 0; k <= N-1; ++k) {
    //     // Control bounds
    //     opti.subject_to( n_cmd(Slice(), k) <= DM({n_max, n_max}) );
    //     opti.subject_to( n_cmd(Slice(), k) >= DM({n_min, n_min}) );
    //     opti.subject_to( alpha_cmd(Slice(), k) <= DM({alpha_max, alpha_max}) );
    //     opti.subject_to( alpha_cmd(Slice(), k) >= DM({alpha_min, alpha_min}) );
    // }
    // for (int k = 1; k <= N; ++k) {
    //     // State bounds
    //     opti.subject_to( n_vars(Slice(), k) <= DM({n_max, n_max}) );
    //     opti.subject_to( n_vars(Slice(), k) >= DM({n_min, n_min}) );
    //     opti.subject_to( alpha_vars(Slice(), k) <= DM({alpha_max, alpha_max}) );
    //     opti.subject_to( alpha_vars(Slice(), k) >= DM({alpha_min, alpha_min}) );
    // }
    for (int k = 0; k <= N-1; ++k) {
        for (int i = 0; i < 2; ++i) {
            if (failstate[i]) {
                opti.subject_to(n_cmd(i, k) == 0);
                opti.subject_to(alpha_cmd(i, k) == alpha_input(i));  // Use appropriate reference
            } else {
                opti.subject_to(n_cmd(i, k) <= n_max);
                opti.subject_to(n_cmd(i, k) >= n_min);
                opti.subject_to(alpha_cmd(i, k) <= alpha_max);
                opti.subject_to(alpha_cmd(i, k) >= alpha_min);
            }
        }
    }
    
    for (int k = 1; k <= N; ++k) {
        for (int i = 0; i < 2; ++i) {
            if (failstate[i]) {
                opti.subject_to(n_vars(i, k) == 0);
                opti.subject_to(alpha_vars(i, k) == alpha_input(i));  // Keep alpha fixed
            } else {
                opti.subject_to(n_vars(i, k) <= n_max);
                opti.subject_to(n_vars(i, k) >= n_min);
                opti.subject_to(alpha_vars(i, k) <= alpha_max);
                opti.subject_to(alpha_vars(i, k) >= alpha_min);
            }
        }
    }

    MX J = 0;
    for (int k = 0; k < N; ++k) {
        MX n1 = n_vars(0, k);
        MX n2 = n_vars(1, k);
        MX alpha1 = alpha_vars(0, k);
        MX alpha2 = alpha_vars(1, k);

        // Calculate thrusts
        MX Thrust1 = ThrustFromRelativeN_MX(n1);
        MX Thrust2 = ThrustFromRelativeN_MX(n2);

        // Mapping to forces and moments (From ran())
        MX tau_X_model = Thrust1 * cos(alpha1) + Thrust2 * cos(alpha2);
        MX tau_Y_model = Thrust1 * sin(alpha1) + Thrust2 * sin(alpha2);
        MX tau_N_model = lx * (Thrust1*sin(alpha1) + Thrust2*sin(alpha2))
                         -(ly1*Thrust1*cos(alpha1) + ly2*Thrust2*cos(alpha2));

        // Error cost
        J += 0.5 * (pow(tau_X - tau_X_model, 2) + 
                    pow(tau_Y - tau_Y_model, 2) + 
                    pow(tau_N - tau_N_model, 2));

        // - Penalty for both pods forward, leading to loss of sway control.
        // MX a1 = exp( -pow( abs(vars(1)), 2 ) / 0.1 ); 
        // MX a2 = exp( -pow( abs(vars(2)), 2 ) / 0.1 );
        // J += 5 * a1 * a2; // X * a1_max * a2_max = X
        
        // - Penalties for directing thrust into another pod slip stream 
        //   (effect not captured by the current ran model, but on the real vessel)
        MX b1 = exp( -pow((alpha1 + M_PI/2), 2) / 0.2 ); 
        J += 5 * b1; 

        MX b2 = exp( -pow((alpha2 - M_PI/2), 2) / 0.2 ); 
        J += 5 * b2;

        // Penalties for pods beeing +90 or -90 at the same time,
        // leading to loss of surge control because of slowly time variying dynamics
        // not captured by this optimalization method.
        MX d1 = exp( -pow(abs(alpha1) - M_PI/2, 2) / 0.1 );
        MX d2 = exp( -pow(abs(alpha2) - M_PI/2, 2) / 0.1 );
        J += 10 * d1 * d2;

        // Penalty for large changes in propeller speed and azimuth angle
        MX d_n1 = n_cmd(0, k) - n_vars(0, k);
        MX d_n2 = n_cmd(1, k) - n_vars(1, k);
        MX d_alpha1 = alpha_cmd(0, k) - alpha_vars(0, k);
        MX d_alpha2 = alpha_cmd(1, k) - alpha_vars(1, k);
        J += 5*(dot(d_n1,d_n1) + dot(d_n2,d_n2)) 
            +5*(dot(d_alpha1,d_alpha1) + dot(d_alpha2,d_alpha2));
    }

    // Optimization:
    opti.minimize(J);

    // Configure solver options
    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 2000;  
    solver_opts["ipopt.tol"] = 0.001;       
    solver_opts["ipopt.acceptable_tol"] = 0.01;
    solver_opts["ipopt.acceptable_iter"] = 400;
    opti.solver("ipopt", solver_opts);

    try {
        OptiSol sol = opti.solve();

        DM n_cmd_sol = opti.value(n_cmd);
        DM alpha_cmd_sol = opti.value(alpha_cmd);

        double n1_opt     = double(n_cmd_sol(0, 1));
        double n2_opt     = double(n_cmd_sol(1, 1));
        double alpha1_opt = double(alpha_cmd_sol(0, 1));
        double alpha2_opt = double(alpha_cmd_sol(1, 1));

        return {n1_opt, alpha1_opt, n2_opt, alpha2_opt};

    } catch (std::exception &e) {

        std::cerr << "Optimization failed: " << e.what() << std::endl;
        return {0.0, 0.0, 0.0, 0.0};
    }
}
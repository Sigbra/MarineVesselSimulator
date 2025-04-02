#include <vector>
#include <Eigen/Dense>
#include "Models/ran.hpp"
#include "Models/model_utilities.hpp"
#include <casadi/casadi.hpp>

using namespace casadi;

std::vector<double> MPC_control_alloc(double tau_X, double tau_Y, double tau_N,
                                      double U, double T_n, double T_alpha,
                                      Eigen::Vector2d n_input, Eigen::Vector2d alpha_input) 
{

    double horizon = 2.0;
    double delta   = 0.1;
    int N = static_cast<int>(horizon/delta); 

    Eigen::Vector3d CO_offset = CO_Offset(U);
    double ly1 =  1.1 - CO_offset(1);    // Left pod lever arm
    double ly2 = -1.1 + CO_offset(1);    // Right pod lever arm
    double lx  = -1.1 - CO_offset(0);    // Pod locations in x

    // Constants from ran()
    double g = 9.81;
    double k_pos = 200*g;         // Positive Bollard
    double k_neg = 200*g;         // Negative Bollard
    double n_max =  1;            // Relative propellar speed max (representing max positive revs)
    double n_min = -1;            // Relative propellar speed min (representing max negative revs)
    double alpha_max = M_PI/2; 
    double alpha_min = -M_PI/2;

    Opti opti;

    MX n_vars = opti.variable(2, N+1);
    MX alpha_vars = opti.variable(2, N+1);

    MX n_cmd = opti.variable(2, N);
    MX alpha_cmd = opti.variable(2, N);

    // Initial values; 
    opti.subject_to(n_vars(Slice(), 0) ==
    DM::vertcat(std::vector<DM>{DM(n_input(0)), DM(n_input(1))}));
    opti.subject_to(alpha_vars(Slice(), 0) ==
    DM::vertcat(std::vector<DM>{DM(alpha_input(0)), DM(alpha_input(1))}));

    // Constraint on change in n and alpha
    for (int k = 1; k < N; ++k) {
        MX n_next = n_vars(Slice(), k) + (delta/T_n) * (n_cmd(Slice(), k) - n_vars(Slice(), k));
        opti.subject_to( n_vars(Slice(), k+1) == n_next );

        MX alpha_next = alpha_vars(Slice(), k) + (delta/T_alpha) * (alpha_cmd(Slice(), k) - alpha_vars(Slice(), k));
        opti.subject_to( alpha_vars(Slice(), k+1) == alpha_next );
    }

    MX J = 0;
    double tau_weight = 10.0;  
    
    for (int k = 0; k < N; ++k) {
        MX n1 = n_vars(0, k);
        MX n2 = n_vars(1, k);
        MX alpha1 = alpha_vars(0, k);
        MX alpha2 = alpha_vars(1, k);

        // Calculate thrusts
        MX Thrust1 = if_else(n1 >= 0, k_pos * n1 * fabs(n1), k_neg * n1 * fabs(n1));
        MX Thrust2 = if_else(n2 >= 0, k_pos * n2 * fabs(n2), k_neg * n2 * fabs(n2));

        // Mapping to forces and moments (From ran())
        MX tau_X_model = Thrust1 * cos(alpha1) + Thrust2 * cos(alpha2);
        MX tau_Y_model = Thrust1 * sin(alpha1) + Thrust2 * sin(alpha2);
        MX tau_N_model = lx * (Thrust1*sin(alpha1) + Thrust2*sin(alpha2))
                        -(ly1*Thrust1*cos(alpha1) + ly2*Thrust2*cos(alpha2));
        
        // Error cost
        J = J + tau_weight * (pow(tau_X - tau_X_model, 2) + 
                              pow(tau_Y - tau_Y_model, 2) + 
                              pow(tau_N - tau_N_model, 2));
    }

    // Optimization:
    opti.minimize(J);

    // Add constraints
    for (int k = 0; k < N; ++k) {
        // Control bounds
        opti.subject_to( n_cmd(Slice(), k) <= DM({n_max, n_max}) );
        opti.subject_to( n_cmd(Slice(), k) >= DM({n_min, n_min}) );
        opti.subject_to( alpha_cmd(Slice(), k) <= DM({alpha_max, alpha_max}) );
        opti.subject_to( alpha_cmd(Slice(), k) >= DM({alpha_min, alpha_min}) );
    }
    for (int k = 1; k <= N; ++k) {
        // State bounds
        opti.subject_to( n_vars(Slice(), k) <= DM({n_max, n_max}) );
        opti.subject_to( n_vars(Slice(), k) >= DM({n_min, n_min}) );
        opti.subject_to( alpha_vars(Slice(), k) <= DM({alpha_max, alpha_max}) );
        opti.subject_to( alpha_vars(Slice(), k) >= DM({alpha_min, alpha_min}) );
    }

    // Set initial guesses for optimization variables
    opti.set_initial(n_vars, repmat(DM::vertcat({n_input(0), n_input(1)}), 1, N+1));
    opti.set_initial(alpha_vars, repmat(DM::vertcat({alpha_input(0), alpha_input(1)}), 1, N+1));

    opti.set_initial(n_cmd, repmat(DM::vertcat({0.1, 0.1}), 1, N));
    opti.set_initial(alpha_cmd, repmat(DM::vertcat({alpha_input(0), alpha_input(1)}), 1, N));

    // Configure solver options
    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 200;  
    solver_opts["ipopt.tol"] = 0.001;       
    solver_opts["ipopt.acceptable_tol"] = 0.01;
    solver_opts["ipopt.acceptable_iter"] = 50;
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
        
        return {n_input(0), alpha_input(0), n_input(1), alpha_input(1)};
    }
}
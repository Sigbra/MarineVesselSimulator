#include "Control/MPC_control_system.hpp"
#include "Utilities/plotting.hpp"
#include "Models/ran.hpp"
#include "Models/model_utilities.hpp"
#include <iostream>
#include "Eigen/Dense"

using namespace casadi;

MPC_Control_System::MPC_Control_System(int N, double dt)
    : N(N), dt(dt)
{
    X_prev_ = DM(); 
    n_cmd_prev_ = DM();
    alpha_cmd_prev_ = DM();
}

MX MPC_Control_System::f(const MX& X, const MX& tau, const MX& V_c, const MX& beta_c) {
    // States: [x, y, psi, u, v, r]
    MX x    = X(0);
    MX y    = X(1);
    MX psi  = X(2);
    MX u    = X(3);
    MX v    = X(4);
    MX r    = X(5);

    // ---------------------------
    // Kinematics
    // ---------------------------
    // x and y velocities are given in the global frame:
    MX dx   = u * cos(psi) - v * sin(psi);
    MX dy   = u * sin(psi) + v * cos(psi);
    MX dpsi = r;

    // ---------------------------
    // Current in body frame
    // ---------------------------
    MX Vcn = V_c * cos(beta_c);     // current north
    MX Vce = V_c * sin(beta_c);     // current east
    MX u_c =  Vcn * cos(psi) + Vce * sin(psi);
    MX v_c = -Vcn * sin(psi) + Vce * cos(psi);

    // ---------------------------
    // Dynamics (Lumped Model)
    // ---------------------------
    // To capture the hydrodynamic effects from the detailed model we introduce
    // effective parameters. These values and formulas represent a first-order 
    // approximation incorporating:
    //   - An effective surge mass m_eff (including added mass in surge)
    //   - An effective yaw inertia Izz_eff (including added yaw inertia)
    //   - Linear damping terms from the computed Xu, Yv and Nr
    //   - An extra cross-flow (or quadratic) drag in sway and an extra nonlinear yaw term
    //
    // The values below are constants based on the full model’s parameters.
    const double m_eff_surge   = 880.0;    // 880  effective surge mass (kg)
    const double m_eff_sway    = 1200.0;   // 1200 effective sway mass (kg) //1200
    const double Izz_eff       = 1250.0;   // 1250 effective yaw moment of inertia (kg·m²)
    const double T_surge = 1.5;        // maximum surge speed (m/s)
    const double T_sway  = 2;          // sway time constant (s)
    const double T_yaw   = 1.5;        // yaw time constant (s)
    
    // Damping coefficients (derived from expressions in the full model)
    double Xu = m_eff_surge / T_surge;                                           //                <--
    // A simplified sway damping (using the effective mass and a time constant)
    double Yv = m_eff_sway / T_sway;
    // Yaw damping coefficient (using effective inertia and yaw time constant).
    // Note: The full model uses a nonlinear yaw term.
    double Nr_linear = Izz_eff / T_yaw;

    // To include additional nonlinear drag:
    // (i) Cross-flow drag in sway is often modeled as quadratic in v.
    // (ii) Nonlinear yaw damping may be approximated by a term proportional to |r|*r.
    double D_y     = 0.5;  // cross-flow drag coefficient (to be tuned)
    double Nr_nl   = 10.0; // nonlinear yaw damping factor (tuned factor)

    MX tau_X = tau(0);
    MX tau_Y = tau(1);
    MX tau_N = tau(2);

    //-----------------------------
    // Relative velocities in body frame
    //-----------------------------
    MX u_rel = u - u_c;
    MX v_rel = v - v_c;

    // Compute a quadratic drag term in sway 
    MX drag_v = D_y * v_rel * fabs(v_rel);

    // Assemble the dynamic equations for the body velocities.
    // The effective dynamics lump together the thrust contributions, the damping, 
    // and any additional drag:
    MX du = (tau_X - Xu * u_rel) / m_eff_surge;
    MX dv = (tau_Y - Yv * v_rel - drag_v) / m_eff_sway;
    MX dr = (tau_N - Nr_linear * r - Nr_nl * fabs(r) * r) / Izz_eff;

    return MX::vertcat({dx, dy, dpsi, du, dv, dr});
}

MX control_allocation(const MX& n, const MX& alpha, double lx_o, double ly1_o, double ly2_o, double pod_radius) {
    // n, alpha are 2x1 vectors, for left and right pods.
    MX n1 = n(0), n2 = n(1);
    MX alpha1 = alpha(0), alpha2 = alpha(1);

    MX ly1 = ly1_o + pod_radius * cos(alpha1);
    MX ly2 = ly2_o + pod_radius * cos(alpha2);
    MX lx1  = lx_o - pod_radius * sin(alpha1);
    MX lx2  = lx_o - pod_radius * sin(alpha2);

    MX Thrust1 = ThrustFromRelativeN_MX(n1);
    MX Thrust2 = ThrustFromRelativeN_MX(n2);

    MX tau_X_model = Thrust1 * cos(alpha1) + Thrust2 * cos(alpha2);
    MX tau_Y_model = Thrust1 * sin(alpha1) + Thrust2 * sin(alpha2);
    MX tau_N_model = lx1 * Thrust1 * sin(alpha1) - ly1 * Thrust1 * cos(alpha1)
                   + lx2 * Thrust2 * sin(alpha2) - ly2 * Thrust2 * cos(alpha2);

    return MX::vertcat({tau_X_model, tau_Y_model, tau_N_model});
}


MX MPC_Control_System::rk4(const MX& Xk, const MX& tau, const MX& V_c, const MX& beta_c, const MX& dt) {
    MX k1 = f(Xk, tau, V_c, beta_c);
    MX k2 = f(Xk + (dt / 2) * k1, tau, V_c, beta_c);
    MX k3 = f(Xk + (dt / 2) * k2, tau, V_c, beta_c);
    MX k4 = f(Xk + dt * k3, tau, V_c, beta_c);
    return Xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

Function MPC_Control_System::oneStepDynamicsFunction() {
    MX Xk = MX::sym("Xk", 6);
    MX tau = MX::sym("tau", 3);
    MX V_c     = MX::sym("Vc");        
    MX beta_c  = MX::sym("beta_c");    

    MX k1 = f(Xk, tau, V_c, beta_c);
    MX k2 = f(Xk + (dt/2) * k1, tau, V_c, beta_c);
    MX k3 = f(Xk + (dt/2) * k2, tau, V_c, beta_c);
    MX k4 = f(Xk + dt * k3, tau, V_c, beta_c);

    MX X_next = Xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    return Function("oneStep", {Xk, tau, V_c, beta_c}, {X_next});
}

bool MPC_Control_System::solve(const std::vector<double>& x0, double x_s, double y_s, double x_d, double y_d, double psi_d, double Vc, double betac, Eigen::VectorXd n_init, Eigen::VectorXd alpha_init, std::vector<bool> failstate) {
    
    double U = std::sqrt(x0[3]*x0[3] + x0[4]*x0[4]); // Current speed from state x0
    
    // Lever arms from ran()
    double ly1_o = 0.79;
    double ly2_o = -0.79;
    double lx_o = -1.17;
    // Eigen::Vector3d CO_offset = CO_Offset(U);
    // ly1_o -= CO_offset(1);    
    // ly2_o += CO_offset(1);    
    // lx_o  -= CO_offset(0);
    double pod_radius = 0.2;  

    // Constants from ran()    
    double n_max =  0.75;           
    double n_min = -0.75;            
    double alpha_max = M_PI/2; 
    double alpha_min = -M_PI/2; 
    double T_n = 0.5, T_alpha = 0.5;
    
    // Create a fresh Opti object inside the solve function
    Opti opti;

    // Decision variables:
    // X: state trajectory (6 x (N+1)) 
    // U: control inputs (3 x N)
    MX X = opti.variable(6, N+1);

    MX V_c = opti.parameter();  // Desired surge speed
    MX beta_c = opti.parameter(); // Desired beta angle
    opti.set_value(V_c, Vc);
    opti.set_value(beta_c, betac);

    Function oneStepFunc = oneStepDynamicsFunction();

    // Constraint on first iteration values: X(:,0) should equal the current state x0.
    opti.subject_to(X(Slice(), 0) == DM(x0));
    // Constraiints on last iteration values.
    //opti.subject_to(X(3, N) == 0);  // Surge velocity (u) must be 0 at final time.
    //opti.subject_to(X(4, N) == 0);  // Sway velocity (v) must be 0 at final time.
    //opti.subject_to(X(5, N) == 0);  // Yaw rate (r) must be 0 at final time.

    // Pod command states
    MX n_vars = opti.variable(2, N+1);
    MX alpha_vars = opti.variable(2, N+1);
    // Pod command inputs
    MX n_cmd = opti.variable(2, N);
    MX alpha_cmd = opti.variable(2, N);

    //Warm start
    if (!X_prev_.is_empty()) {
        opti.set_initial(X, X_prev_);
    }
    if (!n_cmd_prev_.is_empty()) {
        opti.set_initial(n_vars, n_cmd_prev_);
    }
    if (!alpha_cmd_prev_.is_empty()) {
        opti.set_initial(alpha_vars, alpha_cmd_prev_);
    }

    opti.subject_to(n_vars(Slice(), 0) ==
    DM::vertcat(std::vector<DM>{DM(n_init(0)), DM(n_init(1))}));
    opti.subject_to(alpha_vars(Slice(), 0) ==
    DM::vertcat(std::vector<DM>{DM(alpha_init(0)), DM(alpha_init(1))}));

    // Constraint on change in n and alpha
    for (int k = 0; k < N; ++k) {
        MX n_next = n_vars(Slice(), k) + (dt/T_n) * (n_cmd(Slice(), k) - n_vars(Slice(), k));
        opti.subject_to( n_vars(Slice(), k+1) == n_next );

        MX alpha_next = alpha_vars(Slice(), k) + (dt/T_alpha) * (alpha_cmd(Slice(), k) - alpha_vars(Slice(), k));
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
                // opti.subject_to(n_cmd(i, k) == 0);
                // opti.subject_to(alpha_cmd(i, k) == alpha_init(i));  // Use appropriate reference
                opti.subject_to(n_cmd(i, k) <= 0);
                opti.subject_to(n_cmd(i, k) >= 0);
                opti.subject_to(alpha_cmd(i, k) <= alpha_init(i));
                opti.subject_to(alpha_cmd(i, k) >= alpha_init(i));
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
                // opti.subject_to(n_vars(i, k) == 0);
                // opti.subject_to(alpha_vars(i, k) == alpha_init(i));  // Keep alpha fixed
                opti.subject_to(n_vars(i, k) <= 0);
                opti.subject_to(n_vars(i, k) >= 0);
                opti.subject_to(alpha_vars(i, k) <= alpha_init(i));
                opti.subject_to(alpha_vars(i, k) >= alpha_init(i));
            } else {
                opti.subject_to(n_vars(i, k) <= n_max);
                opti.subject_to(n_vars(i, k) >= n_min);
                opti.subject_to(alpha_vars(i, k) <= alpha_max);
                opti.subject_to(alpha_vars(i, k) >= alpha_min);
            }
        }
    }

    // Define path start and goal points
    MX path_x_start = x_s;
    MX path_y_start = y_s;
    MX path_x_goal  = x_d;
    MX path_y_goal  = y_d;  

    // Compute unit direction vector along the path
    MX path_dx = path_x_goal - path_x_start;
    MX path_dy = path_y_goal - path_y_start;
    MX path_length = sqrt(path_dx*path_dx + path_dy*path_dy);
    MX path_dir_x = path_dx / path_length;
    MX path_dir_y = path_dy / path_length;



    MX cost = 0;
    for(int i = 0; i < N; ++i) {
        MX tau = control_allocation(n_vars(Slice(), i), alpha_vars(Slice(), i), lx_o, ly1_o, ly2_o, pod_radius);

        MX X_next = oneStepFunc({X(Slice(), i), tau, V_c, beta_c})[0];
        opti.subject_to(X(Slice(), i + 1) == X_next);

        // --- 1) cross-track error to infinite line ---
        MX rel_x = path_x_goal - X(0,i);
        MX rel_y = path_y_goal - X(1,i);
        MX proj_dist = rel_x*path_dir_x + rel_y*path_dir_y;    
        MX cross_x = rel_x - proj_dist*path_dir_x;
        MX cross_y = rel_y - proj_dist*path_dir_y;
        MX crosstrack_error_sq = sqrt(cross_x*cross_x + cross_y*cross_y + 1e-4);
    
        // --- 2) position error along the line (1 @ start → 0 @ goal) ---
        MX normalized_along_error = proj_dist / path_length;
        MX position_error_sq = sqrt(normalized_along_error * normalized_along_error + 1e-4); 
    
        // --- 3) heading error ---
        MX heading_error = psi_d - X(2,i);
        MX heading_error_sq = sqrt(heading_error * heading_error + 1e-4);

        if (failstate[0] || failstate[1]) { // Penalties when error state
            cost +=  11   * crosstrack_error_sq; //                                 <- Fix this
            cost += 140   * position_error_sq;
            cost +=  11.5 * heading_error_sq;  
        } 
        else { // Penalties in normal operation
            cost +=  9 * crosstrack_error_sq; //9
            cost += 12 * position_error_sq;  //12
            cost += 18 * heading_error_sq;  //18
        }

       
        MX d_n = n_cmd(Slice(), i) - n_vars(Slice(), i);
        MX d_alpha = alpha_cmd(Slice(), i) - alpha_vars(Slice(), i);
        cost += 15*dot(d_n, d_n) + 5*dot(d_alpha, d_alpha); // 5 and 1

        // Penalize large speed resulting in large discretization steps that the MPC can't handle
        MX U_i = sqrt(X(3,i)*X(3,i) + X(4,i)*X(4,i) + 1e-4);
        cost+= exp((U_i-0.8)/0.1); 

    } 
    
    opti.minimize(cost);

    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 2000;  
    solver_opts["ipopt.tol"] = 0.0001;       
    solver_opts["ipopt.acceptable_tol"] = 0.001;
    solver_opts["ipopt.acceptable_iter"] = 400;

    // Choose IPOPT as the solver.
    opti.solver("ipopt", solver_opts);

    try {
        // Solve the optimization problem
        OptiSol sol = opti.solve();

        DM n_cmd_sol = opti.value(n_cmd);
        DM alpha_cmd_sol = opti.value(alpha_cmd);
        DM X_sol = opti.value(X);

        n_opt.resize(2);
        n_opt << double(n_cmd_sol(0, 0)), double(n_cmd_sol(1, 0));

        alpha_opt.resize(2);
        alpha_opt << double(alpha_cmd_sol(0, 0)), double(alpha_cmd_sol(1, 0));

        DM sol_X = opti.value(X);
        DM sol_n = opti.value(n_vars);
        DM sol_alpha = opti.value(alpha_vars);

        X_prev_ = sol_X;
        n_cmd_prev_ = sol_n;
        alpha_cmd_prev_ = sol_alpha;

        return true;
    }
    catch (std::exception& e) {
        std::cerr << "MPC solve failed: " << e.what() << std::endl;
        return false;
    }
}

Eigen::VectorXd MPC_Control_System::get_n_opt() {
    return n_opt;
}

Eigen::VectorXd MPC_Control_System::get_alpha_opt() {
    return alpha_opt;
}



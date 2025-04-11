#include "Control/MPC_motion_control.hpp"
#include <iostream>

using namespace casadi;

MPC_Motion_Control::MPC_Motion_Control(int N, double dt, double m, double Iz, double Xu, double Yv, double Npsi, double Npsi_nl)
    : N(N), dt(dt), m(m), Iz(Iz), Xu(Xu), Yv(Yv), Npsi(Npsi), Npsi_nl(Npsi_nl) {
    // Do not build the Opti problem here—instead, we will build it inside solve() every time.
}

MX MPC_Motion_Control::f(const MX& X, const MX& tau) {
    // States: [x, y, psi, u, v, r]
    MX x    = X(0);
    MX y    = X(1);
    MX psi  = X(2);
    MX u    = X(3);
    MX v    = X(4);
    MX r    = X(5);

    // Controls: [tau_x, tau_y, tau_N]
    MX tau_x = tau(0);
    MX tau_y = tau(1);
    MX tau_N = tau(2);

    // ---------------------------
    // Kinematics
    // ---------------------------
    // x and y velocities are given in the global frame:
    MX dx   = u * cos(psi) - v * sin(psi);
    MX dy   = u * sin(psi) + v * cos(psi);
    MX dpsi = r;

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
    const double m_eff   = 800.0;     // effective surge mass (kg)
    const double Izz_eff = 1000.0;    // effective yaw moment of inertia (kg·m²)
    const double Umax    = 10.0;      // maximum surge speed (m/s)
    const double g       = 9.81;      // gravitational constant (m/s²)
    const double T_sway  = 1.0;       // sway time constant (s)
    const double T_yaw   = 1.0;       // yaw time constant (s)
    
    // Damping coefficients (derived from expressions in the full model)
    // For surge, Xu is computed using gravitational acceleration and Umax.
    double Xu = -200 / Umax;
    // A simplified sway damping (using the effective mass and a time constant)
    double Yv = -m_eff / T_sway;
    // Yaw damping coefficient (using effective inertia and yaw time constant).
    // Note: The full model uses a nonlinear yaw term.
    double Nr_linear = -Izz_eff / T_yaw;

    // To include additional nonlinear drag:
    // (i) Cross-flow drag in sway is often modeled as quadratic in v.
    // (ii) Nonlinear yaw damping may be approximated by a term proportional to |r|*r.
    double D_y     = 0.5;  // cross-flow drag coefficient (to be tuned)
    double Nr_nl   = 10.0; // nonlinear yaw damping factor (tuned factor)

    // For these simplified MPC dynamics we may neglect the current effects.
    // Current correction terms
    MX current_surge = 0;
    MX current_sway  = 0;

    // Compute a quadratic drag term in sway 
    MX drag_v = D_y * v * fabs(v);

    // Assemble the dynamic equations for the body velocities.
    // The effective dynamics lump together the thrust contributions, the damping, 
    // and any additional drag:
    MX du = (tau_x - Xu * u + current_surge) / m_eff;
    MX dv = (tau_y - Yv * v - drag_v + current_sway) / m_eff;
    MX dr = (tau_N - Nr_linear * r - Nr_nl * fabs(r) * r) / Izz_eff;

    return MX::vertcat({dx, dy, dpsi, du, dv, dr});
}


MX MPC_Motion_Control::rk4(const MX& Xk, const MX& tau, const MX& dt) {
    MX k1 = f(Xk, tau);
    MX k2 = f(Xk + (dt / 2) * k1, tau);
    MX k3 = f(Xk + (dt / 2) * k2, tau);
    MX k4 = f(Xk + dt * k3, tau);
    return Xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

bool MPC_Motion_Control::solve(const std::vector<double>& x0, double x_d, double y_d, double psi_d) {
    // Create a fresh Opti object inside the solve function
    Opti opti;

    // Decision variables:
    // X: state trajectory (6 x (N+1))
    // U: control inputs (3 x N)
    MX X = opti.variable(6, N + 1);
    MX U = opti.variable(3, N);

    // Build the reference trajectory as a constant replication of the desired state for all time steps.
    // We are only using the first 3 states (position and heading)
    MX X_ref = repmat(MX::vertcat({x_d, y_d, psi_d}), 1, N + 1);

    // Set initial condition: X(:,0) should equal the current state x0.
    // Convert x0 (std::vector<double>) to a DM:
    opti.subject_to(X(Slice(), 0) == DM(x0));

    // Set up the dynamics constraints over the horizon.
    MX cost = 0;
    for (int i = 0; i < N; ++i) {
        MX X_next = rk4(X(Slice(), i), U(Slice(), i), dt);
        opti.subject_to(X(Slice(), i + 1) == X_next);

        // Cost: weighted sum of error between predicted (x,y,psi) and reference, plus control effort.
        // Here we only consider the first three states.
        MX err_x   = (X(0, i) - X_ref(0, i));
        MX err_y   = (X(1, i) - X_ref(1, i));
        MX err_psi = (X(2, i) - X_ref(2, i));

        cost += 5*pow(err_x, 2) + 5*pow(err_y, 2) + 10*pow(err_psi, 2);
        cost += 0.01 * pow(U(0, i), 2) + 0.01 * pow(U(1, i), 2) + 0.01 * pow(U(2, i), 2);

    }
    // Optionally, you could also add a cost on the terminal state X(:, N)
    // MX err_term = X(Slice(0, 3), N) - X_ref(Slice(), N);
    // cost += 100 * dot(err_term, err_term);

    opti.minimize(cost);

    // (Optional) Add bounds on U, for example:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            opti.subject_to(U(j, i) <= 200);
            opti.subject_to(U(j, i) >= -200);
        }
    }

    Dict solver_opts;
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.max_iter"] = 2000;  
    solver_opts["ipopt.tol"] = 0.001;       
    solver_opts["ipopt.acceptable_tol"] = 0.01;
    solver_opts["ipopt.acceptable_iter"] = 400;

    // Choose IPOPT as the solver.
    opti.solver("ipopt", solver_opts);

    try {
        // Solve the optimization problem
        OptiSol sol = opti.solve();

        // Retrive X, Y, and psi from the solution


        // Retrieve the solution for U (the control trajectory) over the horizon.
        // We'll store them in our member vectors (resizing appropriately).
        tau_x_sol.resize(N);
        tau_y_sol.resize(N);
        tau_N_sol.resize(N);

        // In CasADi C++, to convert the MX solution to a standard vector, you can use nonzeros()
        for (int i = 0; i < N; i++) {
            std::vector<double> tau_i = sol.value(U(Slice(), i)).nonzeros();
            tau_x_sol[i] = tau_i[0];
            tau_y_sol[i] = tau_i[1];
            tau_N_sol[i] = tau_i[2];
        }

        return true;
    }
    catch (std::exception& e) {
        std::cerr << "MPC solve failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<double> MPC_Motion_Control::get_first_tau() {
    // Return the first control input from the solution trajectory
    if (!tau_x_sol.empty() && !tau_y_sol.empty() && !tau_N_sol.empty())
        return {tau_x_sol[0], tau_y_sol[0], tau_N_sol[0]};
    else
        return {0.0, 0.0, 0.0};
}

std::vector<double> MPC_Motion_Control::get_tau_x_horizon() {
    return tau_x_sol;
}

std::vector<double> MPC_Motion_Control::get_tau_y_horizon() {
    return tau_y_sol;
}

std::vector<double> MPC_Motion_Control::get_tau_N_horizon() {
    return tau_N_sol;
}

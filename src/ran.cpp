// ran.cpp
#include "ran.hpp"       // Declaration of ran() in dynamics.hpp
#include "utilities.hpp"      // For prototypes of helper functions (see below)
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

//-------------------------------------------------------------------
// Helper function: Skew-symmetric matrix (Smtrx)
//-------------------------------------------------------------------
Eigen::Matrix3d Smtrx(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S <<      0, -v(2),  v(1),
          v(2),      0, -v(0),
         -v(1),  v(0),      0;
    return S;
}

//-------------------------------------------------------------------
// Helper function: Transformation matrix Hmtrx
// Returns a 6x6 matrix to transform from the center of gravity (CG)
// to the coordinate origin (CO) using the relation:
//   H = [ I   -Smtrx(r) ]
//       [ 0      I      ]
// where r is a 3x1 vector.
//-------------------------------------------------------------------
Eigen::MatrixXd Hmtrx(const Eigen::Vector3d &r) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6,6);
    H.block(0,0,3,3) = Eigen::Matrix3d::Identity();
    H.block(0,3,3,3) = -Smtrx(r);
    H.block(3,3,3,3) = Eigen::Matrix3d::Identity();
    return H;
}

//-------------------------------------------------------------------
// Helper function: m2c (added-mass to Coriolis matrix)
// For this stub, we simply return a 6x6 zero matrix.
//-------------------------------------------------------------------
Eigen::MatrixXd m2c(const Eigen::MatrixXd &MA, const Eigen::VectorXd & /*nu_r*/) {
    return Eigen::MatrixXd::Zero(6,6);
}

//-------------------------------------------------------------------
// Helper function: addedMassSurge
// Stub: returns a value proportional to the mass.
//-------------------------------------------------------------------
double addedMassSurge(double m, double /*L*/, double /*rho*/) {
    return 0.1 * m; // Example: 10% of the mass (adjust as needed)
}

//-------------------------------------------------------------------
// Helper function: crossFlowDrag
// Stub: returns a zero 6x1 vector.
//-------------------------------------------------------------------
Eigen::VectorXd crossFlowDrag(double /*L*/, double /*B_pont*/, double /*T*/, const Eigen::VectorXd & /*nu_r*/) {
    return Eigen::VectorXd::Zero(6);
}

//-------------------------------------------------------------------
// Helper function: eulerang
// Returns the 3x3 kinematic transformation matrix from body angular
// rates to Euler angle rates (using the ZYX Euler angle convention).
//-------------------------------------------------------------------
Eigen::Matrix3d eulerang(double phi, double theta, double psi) {
    (void)psi; // psi is not used in this standard formulation
    Eigen::Matrix3d J;
    double cosTheta = std::cos(theta);
    double sinTheta = std::sin(theta);
    double tanTheta = std::tan(theta);
    double cosPhi = std::cos(phi);
    double sinPhi = std::sin(phi);
    
    J << 1, sinPhi * tanTheta, cosPhi * tanTheta,
         0, cosPhi,           -sinPhi,
         0, sinPhi / cosTheta, cosPhi / cosTheta;
    return J;
}

//-------------------------------------------------------------------
// Helper function: Rzyx
// Returns the rotation matrix (3x3) from the body-fixed frame to the
// inertial frame using the ZYX Euler angle convention.
//-------------------------------------------------------------------
Eigen::Matrix3d Rzyx(double phi, double theta, double psi) {
    double cphi = std::cos(phi),  sphi = std::sin(phi);
    double ctheta = std::cos(theta), stheta = std::sin(theta);
    double cpsi = std::cos(psi),   spsi = std::sin(psi);
    
    Eigen::Matrix3d R;
    R << cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi,
         spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi,
         -stheta,       ctheta * sphi,                      ctheta * cphi;
    return R;
}

//-------------------------------------------------------------------
// The ran() function
//
// Inputs:
//   x      - 12x1 state vector: [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
//   n      - 2x1 propeller speeds [n_left, n_right] (rad/s)
//   mp     - payload mass (kg)
//   rp     - 3x1 payload location (m)
//   V_c    - ocean current speed (m/s)
//   beta_c - ocean current direction (rad)
// Outputs (by reference):
//   xdot      - 12x1 time derivative of state vector
//   U         - Speed (m/s) computed as sqrt(u^2+v^2+w^2)
//   M_out     - 6x6 system mass matrix (MRB + added mass)
//   B_prop_out- 2x2 propeller input matrix (if no input arguments, this is returned)
//-------------------------------------------------------------------
void ran(const Eigen::VectorXd x, const Eigen::VectorXd n_input,
           double mp, const Eigen::Vector3d rp,
           double V_c, double beta_c,
           Eigen::VectorXd &xdot, double &U,
           Eigen::MatrixXd &M_out, Eigen::MatrixXd &B_prop_out)
{
    // Check dimensions
    if (x.size() != 12) {
        std::cerr << "Error: x vector must have dimension 12!" << std::endl;
        return;
    }
    if (n_input.size() != 2) {
        std::cerr << "Error: n vector must have dimension 2!" << std::endl;
        return;
    }
    
    // ---------------------------
    // Main physical constants
    // ---------------------------
    double g   = 9.81;                        // gravitational acceleration (m/s^2)
    double rho = 1025.0;                      // water density (kg/m^3)
    double L   = 2.0;                         // vessel length (m)
    double B   = 1.08;                        // vessel beam (m)
    double m   = 55.0;                        // vessel mass (kg)
    Eigen::Vector3d rg_hull(0.2, 0.0, -0.2);  // center of gravity for hull only
    double R44 = 0.4 * B;                     // radii of gyration in roll
    double R55 = 0.25 * L;                    // in pitch
    double R66 = 0.25 * L;                    // in yaw
    double T_sway = 1.0;                      // sway time constant (s)
    double T_yaw  = 1.0;                      // yaw time constant (s)
    double Umax   = 6 * 0.5144;               // maximum forward speed (m/s) (6 knots)
    
    // Data for one pontoon
    double B_pont  = 0.25;                    // pontoon beam (m)
    double y_pont  = 0.395;                   // lateral offset from centerline (m)
    double Cw_pont = 0.75;                    // waterline area coefficient
    double Cb_pont = 0.4;                     // block coefficient
    
    // ---------------------------
    // State extraction
    // ---------------------------
    // x = [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
    Eigen::VectorXd nu = x.segment(0,6);   // body velocities
    Eigen::VectorXd eta = x.segment(6,6);    // positions and Euler angles
    // Euler angles: phi = eta(3), theta = eta(4), psi = eta(5) (0-indexed)
    
    // Compute speed U from surge, sway, and heave components
    U = std::sqrt(nu(0)*nu(0) + nu(1)*nu(1) + nu(2)*nu(2));
    
    // ---------------------------
    // Ocean current and relative velocity
    // ---------------------------
    double psi = eta(5); // vessel yaw angle
    double u_c = V_c * std::cos(beta_c - psi);
    double v_c = V_c * std::sin(beta_c - psi);
    Eigen::VectorXd nu_c = Eigen::VectorXd::Zero(6);
    nu_c(0) = u_c;
    nu_c(1) = v_c;
    // Relative velocity: nu_r = nu - nu_c
    Eigen::VectorXd nu_r = nu - nu_c;
    
    // Split nu into translational (nu1) and rotational (nu2) parts:
    // nu1 = [u, v, w] and nu2 = [p, q, r]
    Eigen::Vector3d nu2 = nu.segment(3,3);
    
    // Compute current acceleration term: nu_c_dot = [ -Smtrx(nu2)*nu_c_head; zeros(3) ]
    Eigen::Vector3d nu_c_head = nu_c.segment(0,3);
    Eigen::Vector3d nu_c_dot_head = -Smtrx(nu2) * nu_c_head;
    Eigen::VectorXd nu_c_dot = Eigen::VectorXd::Zero(6);
    nu_c_dot.segment(0,3) = nu_c_dot_head;
    // (last three entries remain zero)
    
    // ---------------------------
    // Inertia and trim calculations
    // ---------------------------
    double nabla = (m + mp) / rho;  // displaced volume
    double T_draft = nabla / (2 * Cb_pont * B_pont * L);  // vessel draft
    
    // Inertia dyadic for hull only at CG: Ig_CG = m * diag(R44^2, R55^2, R66^2)
    Eigen::Matrix3d Ig_CG = Eigen::Matrix3d::Zero();
    Ig_CG(0,0) = R44 * R44;
    Ig_CG(1,1) = R55 * R55;
    Ig_CG(2,2) = R66 * R66;
    
    // Corrected center of gravity including payload:
    Eigen::Vector3d rg_corrected = (m * rg_hull + mp * rp) / (m + mp);
    
    // Total inertia matrix at CG:
    Eigen::Matrix3d Ig = Ig_CG - m * (Smtrx(rg_hull) * Smtrx(rg_hull))
                               - mp * (Smtrx(rp) * Smtrx(rp));
    
    // ---------------------------
    // Propeller data and control forces
    // ---------------------------
    double l1 = -y_pont;       // left propeller lever arm (m)
    double l2 = y_pont;        // right propeller lever arm (m)
    double k_pos = 0.02216 / 2.0;
    double k_neg = 0.01289 / 2.0;
    double n_max = std::sqrt((0.5 * 24.4 * g) / k_pos);
    double n_min = -std::sqrt((0.5 * 13.6 * g) / k_neg);
    
    // ---------------------------
    // Rigid-body (MRB) and Coriolis (CRB) matrices at the CG
    // ---------------------------
    Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d O3 = Eigen::Matrix3d::Zero();
    // MRB_CG = [ (m+mp)*I3,     O3;
    //            O3,         Ig ]
    Eigen::MatrixXd MRB_CG(6,6);
    MRB_CG.block(0,0,3,3) = (m + mp) * I3;
    MRB_CG.block(0,3,3,3) = O3;
    MRB_CG.block(3,0,3,3) = O3;
    MRB_CG.block(3,3,3,3) = Ig;
    
    // CRB_CG = [ (m+mp)*Smtrx(nu2),    O3;
    //            O3,              -Smtrx(Ig*nu2) ]
    Eigen::MatrixXd CRB_CG(6,6);
    CRB_CG.block(0,0,3,3) = (m + mp) * Smtrx(nu2);
    CRB_CG.block(0,3,3,3) = O3;
    CRB_CG.block(3,0,3,3) = O3;
    CRB_CG.block(3,3,3,3) = -Smtrx(Ig * nu2);
    
    // Transform from CG to CO using H = [I, -Smtrx(rg_corrected); 0, I]
    Eigen::MatrixXd H = Hmtrx(rg_corrected);
    Eigen::MatrixXd MRB = H.transpose() * MRB_CG * H;
    Eigen::MatrixXd CRB = H.transpose() * CRB_CG * H;
    
    // ---------------------------
    // Hydrodynamic added mass and Coriolis
    // ---------------------------
    double Xudot = -addedMassSurge(m, L, rho);
    double Yvdot = -1.5 * m;
    double Zwdot = -1.0 * m;
    double Kpdot = -0.2 * Ig(0,0);
    double Mqdot = -0.8 * Ig(1,1);
    double Nrdot = -1.7 * Ig(2,2);
    Eigen::MatrixXd MA = Eigen::MatrixXd::Zero(6,6);
    MA(0,0) = -Xudot;
    MA(1,1) = -Yvdot;
    MA(2,2) = -Zwdot;
    MA(3,3) = -Kpdot;
    MA(4,4) = -Mqdot;
    MA(5,5) = -Nrdot;
    // Added mass Coriolis matrix (stub)
    Eigen::MatrixXd CA = m2c(MA, nu_r);
    
    // System mass and Coriolis matrices
    Eigen::MatrixXd M_sys = MRB + MA;
    Eigen::MatrixXd C_sys = CRB + CA;
    
    // ---------------------------
    // Hydrostatic restoring (spring) coefficients
    // ---------------------------
    double Aw_pont = Cw_pont * L * B_pont;
    double I_T = 2 * (1.0/12.0) * L * std::pow(B_pont, 3) * (6 * std::pow(Cw_pont, 3) / ((1+Cw_pont)*(1+2*Cw_pont))) +
                 2 * Aw_pont * y_pont * y_pont;
    double I_L = 0.8 * 2 * (1.0/12.0) * B_pont * std::pow(L, 3);
    double KB = (1.0/3.0) * (5*T_draft/2.0 - 0.5 * nabla / (L * B_pont));
    double BM_T = I_T / nabla;
    double BM_L = I_L / nabla;
    double KM_T = KB + BM_T;
    double KM_L = KB + BM_L;
    double KG = T_draft - rg_corrected(2);
    double GM_T = KM_T - KG;
    double GM_L = KM_L - KG;
    
    double G33 = rho * g * (2 * Aw_pont);
    double G44 = rho * g * nabla * GM_T;
    double G55 = rho * g * nabla * GM_L;
    Eigen::MatrixXd G_CF = Eigen::MatrixXd::Zero(6,6);
    G_CF(2,2) = G33;
    G_CF(3,3) = G44;
    G_CF(4,4) = G55;
    
    // Transform hydrostatic matrix from the center-of-flotation frame to the CO.
    // Here we use Hmtrx with LCF = [-0.2, 0, 0]
    Eigen::Vector3d LCF_vec; LCF_vec << -0.2, 0, 0;
    Eigen::MatrixXd H2 = Hmtrx(LCF_vec);
    Eigen::MatrixXd G = H2.transpose() * G_CF * H2;

    // ---------------------------
    // Natural frequencies (for damping design)
    // ---------------------------
    double w3 = std::sqrt(G33 / M_sys(2,2));
    double w4 = std::sqrt(G44 / M_sys(3,3));
    double w5 = std::sqrt(G55 / M_sys(4,4));
    
    // ---------------------------
    // Linear damping terms
    // ---------------------------
    double Xu = -24.4 * g / Umax;
    double Yv = -M_sys(1,1) / T_sway;
    double Zw = -2 * 0.3 * w3 * M_sys(2,2);
    double Kp_damp = -2 * 0.2 * w4 * M_sys(3,3);
    double Mq_damp = -2 * 0.4 * w5 * M_sys(4,4);
    double Nr = -M_sys(5,5) / T_yaw;
    

    // Copy and saturate propeller speeds
    Eigen::VectorXd n = n_input;
    for (int i = 0; i < 2; i++) {
        if (n(i) > n_max)
            n(i) = n_max;
        else if (n(i) < n_min)
            n(i) = n_min;
    }
    
    // Control forces and moments, with saturated propeller speed
    Eigen::Vector2d Thrust;
    for (int i = 0; i < 2; i++) {
        if (n(i) > 0)
            Thrust(i) = k_pos * n(i) * std::abs(n(i));
        else
            Thrust(i) = k_neg * n(i) * std::abs(n(i));
    }

    // Control forces and moments
    // tau = [Thrust_left + Thrust_right; 0; 0; 0; 0; -l1*Thrust_left - l2*Thrust_right]
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(6);
    tau(0) = Thrust(0) + Thrust(1);
    tau(5) = -l1 * Thrust(0) - l2 * Thrust(1);

    //Linear damping using relative velocities + nonlinear yaw dampning
    double Xh = Xu * nu_r(0);
    double Yh = Yv * nu_r(1);
    double Zh = Zw * nu_r(2);
    double Kh = Kp_damp * nu_r(3);
    double Mh = Mq_damp * nu_r(4);
    double Nh = Nr * (1 + 10 * std::abs(nu_r(5))) * nu_r(5);
    Eigen::VectorXd tau_damp(6);

    tau_damp << Xh, Yh, Zh, Kh, Mh, Nh;
    
    // ---------------------------
    // Cross-flow drag (using strip theory; stub here)
    // ---------------------------
    Eigen::VectorXd tau_crossflow = crossFlowDrag(L, B_pont, T_draft, nu_r);
    
    // ---------------------------
    // Payload forces and moments
    // ---------------------------
    // f_payload = Rzyx(phi, theta, psi)' * [0; 0; mp*g]
    double phi = eta(3);
    double theta = eta(4);
    Eigen::Matrix3d Rzyx_mat = Rzyx(phi, theta, psi);
    Eigen::Vector3d f_payload = Rzyx_mat.transpose() * (Eigen::Vector3d() << 0, 0, mp * g).finished();
    Eigen::Vector3d m_payload = Smtrx(rp) * f_payload;
    Eigen::VectorXd g_0(6);
    g_0 << f_payload, m_payload;
    
    // ---------------------------
    // Trim condition (adjust equilibrium)
    // ---------------------------
    // eta_0 = [0; 0; inv(G(3:5,3:5)) * g_0(3:5); 0]
    // In 0-indexed C++, use block(2,2,3,3) and segment(2,3)
    Eigen::MatrixXd G_sub = G.block(2,2,3,3);
    Eigen::Vector3d g0_sub = g_0.segment(2,3);
    Eigen::Vector3d eta_0_sub = G_sub.inverse() * g0_sub;
    Eigen::VectorXd eta_0 = Eigen::VectorXd::Zero(6);
    eta_0(2) = eta_0_sub(0);
    eta_0(3) = eta_0_sub(1);
    eta_0(4) = eta_0_sub(2);
    // Adjust eta by shifting equilibrium
    eta = eta - eta_0;
    
    // ---------------------------
    // Kinematic transformation for position update
    // ---------------------------
    Eigen::Matrix3d J = eulerang(phi, theta, psi);
    
    // ---------------------------
    // Assemble state derivative (xdot)
    // ---------------------------
    // Debugging: Print dimensions and values of involved matrices and vectors
    std::cout << "tau: " << tau.transpose() << std::endl;
    std::cout << "tau_damp: " << tau_damp.transpose() << std::endl;
    std::cout << "tau_crossflow: " << tau_crossflow.transpose() << std::endl;
    std::cout << "C_sys * nu_r: " << (C_sys * nu_r).transpose() << std::endl;
    std::cout << "G * eta: " << (G * eta).transpose() << std::endl;

    Eigen::VectorXd force_term = tau + tau_damp + tau_crossflow - C_sys * nu_r - G * eta;
    std::cout << "force_term: " << force_term.transpose() << std::endl;

    Eigen::VectorXd acceleration = M_sys.fullPivLu().solve(force_term);
    std::cout << "acceleration: " << acceleration.transpose() << std::endl;

    Eigen::VectorXd xdot_vel = nu_c_dot + acceleration;
    std::cout << "xdot_vel: " << xdot_vel.transpose() << std::endl;

    Eigen::VectorXd xdot_pos = J * nu.head(3);
    std::cout << "xdot_pos: " << xdot_pos.transpose() << std::endl;

    // Assemble full 12x1 state derivative
    xdot.resize(12);
    xdot << xdot_vel, Eigen::VectorXd::Zero(3), xdot_pos;
    std::cout << "xdot: " << xdot.transpose() << std::endl;

    
    // ---------------------------
    // Set output mass matrix and B_prop matrix
    // ---------------------------
    M_out = M_sys;
    // If ran() is called with no arguments (n not provided) MATLAB returns B_prop.
    // Here we always compute B_prop as:
    B_prop_out = k_pos * (Eigen::MatrixXd(2,2) << 1, 1, y_pont, -y_pont).finished();
}

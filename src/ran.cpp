// ran.cpp
#include "ran.hpp"       
#include "utilities.hpp"     
#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>

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
// Helper function: Tzyx
//-------------------------------------------------------------------
Eigen::Matrix3d Tzyx(double phi, double theta) {
    Eigen::Matrix3d T;

    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double t_theta = std::tan(theta);
    double c_theta = std::cos(theta);

    T << 1, s_phi * t_theta, c_phi * t_theta,
         0, c_phi, -s_phi,
         0, s_phi / c_theta, c_phi / c_theta;

    return T;
}

//-------------------------------------------------------------------
// Helper function: m2c (added-mass to Coriolis matrix)
// Stub, return a 6x6 zero matrix.
//-------------------------------------------------------------------
Eigen::MatrixXd m2c(const Eigen::MatrixXd &MA, const Eigen::VectorXd & /*nu_r*/) {
    return Eigen::MatrixXd::Zero(6,6);
}

//-------------------------------------------------------------------
// Helper function: addedMassSurge
// Stub: returns a value proportional to the mass.
//-------------------------------------------------------------------
double addedMassSurge(double m, double /*L*/, double /*rho*/) {
    return 0.1 * m; 
}

// Function: Hoerner
// Computes the 2D Hoerner cross-flow form coefficient based on beam (B) and draft (T).
// The coefficient is obtained via linear interpolation from digitized data.
// Inputs:
//    B - beam (m)
//    T - draft (m)
// Output:
//    CY_2D - 2D Hoerner cross-flow form coefficient
double Hoerner(double B, double T) {
    // Compute the ratio B/(2*T)
    double ratio = B / (2.0 * T);
    
    // Digitized data: each pair is {B/(2*T), C_D}
    const std::vector<std::pair<double, double>> CD_DATA = {
        {0.0108623, 1.96608},
        {0.176606,  1.96573},
        {0.353025,  1.89756},
        {0.451863,  1.78718},
        {0.472838,  1.58374},
        {0.492877,  1.27862},
        {0.493252,  1.21082},
        {0.558473,  1.08356},
        {0.646401,  0.998631},
        {0.833589,  0.87959},
        {0.988002,  0.828415},
        {1.30807,   0.759941},
        {1.63918,   0.691442},
        {1.85998,   0.657076},
        {2.31288,   0.630693},
        {2.59998,   0.596186},
        {3.00877,   0.586846},
        {3.45075,   0.585909},
        {3.7379,    0.559877},
        {4.00309,   0.559315}
    };
    
    // If the ratio is above the highest value in the table, return the last value
    if (ratio >= CD_DATA.back().first) {
        return CD_DATA.back().second;
    }
    
    // If the ratio is below the first data point, return the first value.
    if (ratio <= CD_DATA.front().first) {
        return CD_DATA.front().second;
    }
    
    // Otherwise, search for the interval in which the ratio falls and interpolate.
    for (size_t i = 0; i < CD_DATA.size() - 1; ++i) {
        double x1 = CD_DATA[i].first;
        double x2 = CD_DATA[i+1].first;
        if (ratio >= x1 && ratio <= x2) {
            double y1 = CD_DATA[i].second;
            double y2 = CD_DATA[i+1].second;
            // Linear interpolation
            double t = (ratio - x1) / (x2 - x1);
            return y1 + t * (y2 - y1);
        }
    }
    
    // Fallback: Should never reach here.
    throw std::runtime_error("Interpolation error in Hoerner function.");
}

//-------------------------------------------------------------------
// Helper function: crossFlowDrag
//-------------------------------------------------------------------
Eigen::VectorXd crossFlowDrag(double L, double B, double T, const Eigen::VectorXd& nu_r) {
    // Check that nu_r has the correct size
    if (nu_r.size() < 6) {
        throw std::runtime_error("nu_r must be a 6-element vector.");
    }

    const double rho = 1025.0;   // density of water (kg/m^3)
    const double dx = L / 20.0;    // divide the craft into 20 strips
    const double Cd_2D = Hoerner(B, T);  // 2-D drag coefficient from Hoerner's curve

    double Yh = 0.0, Zh = 0.0, Mh = 0.0, Nh = 0.0;

    // Loop from -L/2 to L/2 with step dx.
    // Use a for loop; note that floating point comparisons might not hit L/2 exactly.
    for (double xL = -L/2.0; xL <= L/2.0 + 1e-6; xL += dx) {
        // Extract relevant components from nu_r (C++ indices):
        // v_r (sway velocity) is nu_r[1] (MATLAB nu_r(2))
        // w_r (heave velocity) is nu_r[2] (MATLAB nu_r(3))
        // q (pitch rate)      is nu_r[4] (MATLAB nu_r(5))
        // r (yaw rate)        is nu_r[5] (MATLAB nu_r(6))
        double v_r = nu_r(1);
        double w_r = nu_r(2);
        double q   = nu_r(4);
        double r   = nu_r(5);

        // Effective velocities at the strip (including rotational effects)
        double effective_v = v_r + xL * r;
        double effective_w = w_r + xL * q;
        double U_h = std::abs(effective_v) * effective_v;
        double U_v = std::abs(effective_w) * effective_w;

        // Accumulate cross-flow drag forces and moments
        Yh -= 0.5 * rho * T * Cd_2D * U_h * dx;       // sway force
        Zh -= 0.5 * rho * T * Cd_2D * U_v * dx;         // heave force
        Mh -= 0.5 * rho * T * Cd_2D * xL * U_v * dx;    // pitch moment
        Nh -= 0.5 * rho * T * Cd_2D * xL * U_h * dx;    // yaw moment
    }

    // Assemble the 6-DOF cross-flow drag vector: [0, Yh, Zh, 0, Mh, Nh]^T
    Eigen::VectorXd tau_crossflow(6);
    tau_crossflow << 0.0, Yh, Zh, 0.0, Mh, Nh;

    return tau_crossflow;
}

//-------------------------------------------------------------------
// Helper function: eulerang
// Returns the 3x3 kinematic transformation matrix from body angular
// rates to Euler angle rates (using the ZYX Euler angle convention).
//-------------------------------------------------------------------
Eigen::MatrixXd eulerang(double phi, double theta, double psi) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, 6);
    Eigen::Matrix3d J1 = Rzyx(phi, theta, psi);
    Eigen::Matrix3d J2 = Tzyx(phi, theta);
    

    J.block<3,3>(0,0) = J1;
    J.block<3,3>(3,3) = J2;
    //std::cout << "eulerang, J: \n" << J << std::endl;

    return J;
}

//-------------------------------------------------------------------
// Helper function: Rzyx
// Returns the rotation matrix (3x3) from the body-fixed frame to the
// inertial frame using the ZYX Euler angle convention.
//-------------------------------------------------------------------
Eigen::Matrix3d Rzyx(double phi, double theta, double psi) {
    // Compute trigonometric values
    double cphi = std::cos(phi);
    double sphi = std::sin(phi);
    double cth  = std::cos(theta);
    double sth  = std::sin(theta);
    double cpsi = std::cos(psi);
    double spsi = std::sin(psi);

    // Define rotation matrix R
    Eigen::Matrix3d R;
    R <<  cpsi * cth,  -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth,
          spsi * cth,   cpsi * cphi + sphi * sth * spsi,  -cpsi * sphi + sth * spsi * cphi,
          -sth,         cth * sphi,                       cth * cphi;

    return R;
}

std::vector<double> CO_frame(double L, double U) {
    double CO_min = L/2;
    double CO_max = L;
    double x = 0;
    double y = 0;
    double z = 0;
    return {x, y, z};
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
//   alpha  - 2x1 azimuth angles [alpha_left, alpha_right] (rad)
// Outputs (by reference):
//   xdot      - 12x1 time derivative of state vector
//   U         - Speed (m/s) computed as sqrt(u^2+v^2+w^2)
//   M_out     - 6x6 system mass matrix (MRB + added mass)
//   B    - 3x2 propeller input matrix 
//-------------------------------------------------------------------
void ran(const Eigen::VectorXd x, const Eigen::VectorXd n_input, const Eigen::VectorXd alpha_input,
           double mp, const Eigen::Vector3d rp, double V_c, double beta_c,
           Eigen::VectorXd &xdot, double &U, Eigen::MatrixXd &M_out, Eigen::MatrixXd &B)
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
    if (alpha_input.size() != 2) {
        std::cerr << "Error: alpha vector must have dimension 2!" << std::endl;
        return;
    }
    
    // ---------------------------
    // Main physical constants
    // ---------------------------
    double g    = 9.81;                        // gravitational acceleration (m/s^2)
    double rho  = 1025.0;                      // water density (kg/m^3)
    double L    = 5.0;                         // vessel length (m)
    double Beam = 3;                           // vessel beam (m)
    double m    = 800.0;                       // vessel mass (kg)
    Eigen::Vector3d rg_hull(-0.5, 0.0, -0.2);  // center of gravity for hull only
    double R44 = 0.4 * Beam;                   // radii of gyration in roll
    double R55 = 0.25 * L;                     // in pitch
    double R66 = 0.25 * L;                     // in yaw
    double T_sway = 1.0;                       // sway time constant (s)
    double T_yaw  = 1.0;                       // yaw time constant (s)
    double Umax   = 10;                        // maximum forward speed (m/s)
    
    // Data for one pontoon
    double Beam_pont  = 0.70;                 // pontoon beam (m)
    double y_pont  = 1.1;                     // lateral offset from centerline (m)
    double Cw_pont = 1;                       // waterline area coefficient
    double Cb_pont = 0.5;                     // block coefficient
    
    // ---------------------------
    // State extraction
    // ---------------------------
    // x = [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
    Eigen::VectorXd nu = x.segment(0,6);   // body velocities
    Eigen::VectorXd eta = x.segment(6,6);  // positions and Euler angles

    
    // Speed U from surge and sway components
    U = std::sqrt(nu(0)*nu(0) + nu(1)*nu(1));
    
    // ---------------------------
    // Ocean current and relative velocity
    // ---------------------------
    double psi = eta(5); 
    double u_c = V_c * std::cos(beta_c - psi);
    double v_c = V_c * std::sin(beta_c - psi);
    Eigen::VectorXd nu_c = Eigen::VectorXd::Zero(6);
    nu_c(0) = u_c;
    nu_c(1) = v_c;
    // Relative velocity:
    Eigen::VectorXd nu_r = nu - nu_c;
    
    // Translational (nu1) and rotational (nu2) parts of nu:
    // nu1 = [u, v, w] and nu2 = [p, q, r]
    Eigen::Vector3d nu2 = nu.segment(3,3);
    
    // Compute current acceleration term: nu_c_dot = [ -Smtrx(nu2)*nu_c_head; zeros(3) ]
    Eigen::Vector3d nu_c_head = nu_c.segment(0,3);
    Eigen::Vector3d nu_c_dot_head = -Smtrx(nu2) * nu_c_head;
    Eigen::VectorXd nu_c_dot = Eigen::VectorXd::Zero(6);
    nu_c_dot.segment(0,3) = nu_c_dot_head;
    
    
    // ---------------------------
    // Inertia and trim calculations
    // ---------------------------
    double nabla = (m + mp) / rho;  // displaced volume
    double T_draft = nabla / (2 * Cb_pont * Beam_pont * L);  // vessel draft
    
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
    // Azimuth pods / Pontoon data and control forces
    // ---------------------------
    double ly1 = y_pont;         // left pod lever arm (m)
    double ly2 = -y_pont;        // right pod lever arm (m)
    double lx  = -0.9;           // forward displacement of pods (m)
    double k_pos = 220*g;        // Positive Bollard
    double k_neg = 220*g;        // Negative Bollard
    double n_max =  1;           // relative propellar speed max 
    double n_min = -1;           // relative propellar speed min
    double alpha_max = M_PI/2;   // maximum azimuth angle (rad)
    double alpha_min = -M_PI/2;  // minimum azimuth angle (rad)
    
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
    double Aw_pont = Cw_pont * L * Beam_pont;
    double I_T = 2 * (1.0/12.0) * L * std::pow(Beam_pont, 3) * (6 * std::pow(Cw_pont, 3) / ((1+Cw_pont)*(1+2*Cw_pont))) +
                 2 * Aw_pont * y_pont * y_pont;
    double I_L = 0.8 * 2 * (1.0/12.0) * Beam_pont * std::pow(L, 3);
    double KB = (1.0/3.0) * (5*T_draft/2.0 - 0.5 * nabla / (L * Beam_pont));
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
    

    // Saturate and compute propeller speeds
    Eigen::VectorXd n = n_input;
    for (int i = 0; i < 2; i++) {
        if (n(i) > n_max)
            n(i) = n_max;
        else if (n(i) < n_min)
            n(i) = n_min;
    }

    // Saturate and compute azimuth angles
    Eigen::VectorXd alpha = alpha_input;
    for (int i = 0; i < 2; i++) {
        if (alpha(i) > alpha_max)
            alpha(i) = alpha_max;
        else if (alpha(i) < alpha_min)
            alpha(i) = alpha_min;
    }
    
    // Thrust for podded propellar (Fossen, Chapter 9.2.1)
    // T = p*D⁴*K_T(J_a) * |n|n ~= T_|n|n * |n|n - T|n|u_a * |n|u_a 
    // u_a = (1-w)*u, u - forward speed of ship, w - wake fraction (0.1-0.4)
    // T_|n|n = p*D⁴*a1   
    // T_|n|u_a = p*D³*a2
    // D - propellar diameter, p - water density, n - propeller revs, a - const value, 
    // Temporary Thrust calculation based on otter.m in the MssToolbox
    Eigen::Vector2d Thrust;
    for (int i = 0; i < 2; i++) {
        if (n(i) > 0)
            Thrust(i) = k_pos * n(i) * std::abs(n(i)); //Positive Thrust (thust, not direction of thrust)
        else
            Thrust(i) = k_neg * n(i) * std::abs(n(i)); //Nagavtive Thrust (thust, not direction of thrust)
    }

    // Control forces and moments. 
    //  Using ly1 and ly2, but what about lx?
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(6);
    tau(0) = Thrust(0) * cos(alpha(0)) + Thrust(1) * cos(alpha(1)); //X: Surge 
    tau(1) = Thrust(0) * sin(alpha(0)) + Thrust(1) * sin(alpha(1)); //Y: Sway 
    tau(5) = lx * (Thrust(0) * sin(alpha(0)) + Thrust(1) * sin(alpha(1)))
          - (ly1 * Thrust(0) * cos(alpha(0)) + ly2 * Thrust(1) * cos(alpha(1))); //N: Yaw

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
    // Cross-flow drag
    // ---------------------------
    Eigen::VectorXd tau_crossflow = crossFlowDrag(L, Beam_pont, T_draft, nu_r);
    
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
    Eigen::MatrixXd J = eulerang(phi, theta, psi);
    //std::cout << "J dimensions: " << J.rows() << "x" << J.cols() << std::endl;
    //std::cout << "nu.head(6) dimensions: " << nu.head(6).rows() << "x" << nu.head(6).cols() << std::endl;
    // ---------------------------
    // Assemble state derivative (xdot)
    // ---------------------------
    // Debugging: Print dimensions and values of involved matrices and vectors
    //std::cout << "tau: " << tau.transpose() << std::endl;
    //std::cout << "tau_damp: " << tau_damp.transpose() << std::endl;
    //std::cout << "tau_crossflow: " << tau_crossflow.transpose() << std::endl;
    //std::cout << "C_sys * nu_r: " << (C_sys * nu_r).transpose() << std::endl;
    //std::cout << "G * eta: " << (G * eta).transpose() << std::endl;

    Eigen::VectorXd force_term = tau + tau_damp + tau_crossflow - C_sys * nu_r - G * eta;
    //std::cout << "force_term: " << force_term.transpose() << std::endl;
    //std::cout << "force_term dimensions: " << force_term.rows() << "x" << force_term.cols() << std::endl;

    Eigen::VectorXd acceleration = M_sys.fullPivLu().solve(force_term);
    //std::cout << "acceleration: " << acceleration.transpose() << std::endl;
    //std::cout << "acceleration dimensions: " << acceleration.rows() << "x" << acceleration.cols() << std::endl;
    
    Eigen::VectorXd xdot_vel = nu_c_dot + acceleration;
    //std::cout << "xdot_vel: " << xdot_vel.transpose() << std::endl;
    //std::cout << "xdot_vel dimensions: " << xdot_vel.rows() << "x" << xdot_vel.cols() << std::endl;

    Eigen::VectorXd xdot_pos = J * nu.head(6);
    //std::cout << "xdot_pos: " << xdot_pos.transpose() << std::endl;
    
    // Assemble full 12x1 state derivative //xdot might be a logical issue
    xdot.resize(12);
    xdot << xdot_vel, xdot_pos;
    //std::cout << "xdot: " << xdot.transpose() << std::endl;
    //std::cout << "xdot dimensions: " << xdot.rows() << "x" << xdot.cols() << std::endl;

    
    // ---------------------------
    // Set output mass matrix M 
    // ---------------------------
    M_out = M_sys;
    //std::cout << "M_out: " << M_out << std::endl;

    //----------------------------
    // Input matrix B (3x4). 
    // B = T_e * K_e (Fossen Chapter 9.4 and 11.2)
    //----------------------------

    // T_e = [1,   0,  1,   0;
    //        0,   1,  0,   1;
    //        ly1, lx, ly2, lx]
    Eigen::MatrixXd T_e = Eigen::MatrixXd::Zero(3,4);
    // First row: contribution to surge (X)
    T_e(0,0) = 1;
    T_e(0,1) = 0;
    T_e(0,2) = 1;
    T_e(0,3) = 0;
    // Second row: contribution to sway (Y)
    T_e(1,0) = 0;
    T_e(1,1) = 1;
    T_e(1,2) = 0;
    T_e(1,3) = 1;
    // Third row: contribution to yaw moment (N)
    T_e(2,0) =  ly1;
    T_e(2,1) =  lx;
    T_e(2,2) =  ly2;
    T_e(2,3) =  lx;

    // K_e = [K1  0  0  0;
    //         0 K2  0  0;
    //         0  0 K3  0;
    //         0  0  0 K4]
    Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(4,4);
    //How should these be defined?                            !!!
    double K1 = 1; 
    double K2 = 1;
    double K3 = 1;
    double K4 = 1;
    K_e(0,0) = K1;
    K_e(1,1) = K2;
    K_e(2,2) = K3;
    K_e(3,3) = K4;

    B = T_e * K_e;
}


// Specialized RK4 integrator for the RAN model. 
void rk4_ran_step(Eigen::VectorXd& x, const Eigen::VectorXd& n, const Eigen::VectorXd& alpha,
    double mp, const Eigen::Vector3d& rp,
    double V_c, double beta_c, double h) {
    // Initialize output variables
    Eigen::VectorXd xdot(12);
    double U;
    Eigen::MatrixXd M_out(6, 6);
    Eigen::MatrixXd B(3, 4);

    // Compute k1
    ran(x, n, alpha, mp, rp, V_c, beta_c, xdot, U, M_out, B);
    Eigen::VectorXd k1 = h * xdot;
    //std::cout << "k1: " << k1.transpose() << std::endl;

    // Compute k2
    ran(x + 0.5 * k1, n, alpha, mp, rp, V_c, beta_c, xdot, U, M_out, B);
    Eigen::VectorXd k2 = h * xdot;
    //std::cout << "k2: " << k2.transpose() << std::endl;

    // Compute k3
    ran(x + 0.5 * k2, n, alpha, mp, rp, V_c, beta_c, xdot, U, M_out, B);
    Eigen::VectorXd k3 = h * xdot;
    //std::cout << "k3: " << k3.transpose() << std::endl;

    // Compute k4
    ran(x + k3, n, alpha, mp, rp, V_c, beta_c, xdot, U, M_out, B);
    Eigen::VectorXd k4 = h * xdot;
    //std::cout << "k4: " << k4.transpose() << std::endl;

    // Update state vector x
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
}
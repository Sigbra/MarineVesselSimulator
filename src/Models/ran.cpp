#include "Models/ran.hpp"       
#include "Utilities/calculations.hpp"     
#include "Models/model_utilities.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>

RAN::RAN() {
    g    = 9.81;                        
    rho  = 1025.0;                     
    L    = 5.46;                         
    Beam = 2.14;                           
    m    = 850.0;  

    R44 = 0.4  * Beam;                  
    R55 = 0.25 * L;                    
    R66 = 0.25 * L;   

    T_surge = 1.5; //Specifics unknown
    T_sway = 2;    //Specifics unknown                  
    T_yaw  = 1.5;  //Specifics unknown  

    Umax   = 8;      

    Beam_pont = 0.50;                 
    y_pont    = 0.56;                    
    Cw_pont   = 0.80; //Specifics unknown                     
    Cb_pont   = 0.50; //Specifics unknown
     
    // Relative propeller speed interval [-1, 1] limited to [-0.75, 0.75]
    // due to battery power limitation during bollard pull tests.
    n_max =  0.75;           
    n_min = -0.75; 

    alpha_max = M_PI/2;   
    alpha_min = -M_PI/2; 

    T_n = 0.4;     //Specifics unknown
    T_alpha = 0.8;

    ly1_o = -0.79;
    ly2_o = 0.79;
    lx_o = -1.17;
    pod_radius = 0.2;

    M = Eigen::MatrixXd::Zero(6, 6);
    B = Eigen::MatrixXd::Zero(3, 2);
    xdot = Eigen::VectorXd::Zero(12);

    propagate = true;

    n1_fail = false;
    n2_fail = false;

    double bollard_order = 5;
    thrust_coeffs = NOrderApprox("../data/bollard_pull_data.csv", bollard_order);
}

void RAN::update(const Eigen::VectorXd x, double mp, double V_c, double beta_c,
              double h, Eigen::Vector2d n, Eigen::Vector2d alpha)
{
    // Check dimensions
    if (x.size() != 12) {
        std::cerr << "Error: x vector must have dimension 12!" << std::endl;
        return;
    }

    // ---------------------------
    // State extraction
    // ---------------------------
    // x = [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
    // - Body velocities
    Eigen::VectorXd nu = x.segment(0,6);   
    // - Positions and Euler angles
    Eigen::VectorXd eta = x.segment(6,6); 

    // Speed U from surge and sway components
    U = std::sqrt(nu(0)*nu(0) + nu(1)*nu(1));

    //----------------------------
    // Coordinate frame updates
    // - Assuming CO = (0, 0, 0) means center of the boat.
    //----------------------------
    // - Coordinate origin (CO) offset update (Speed dependant)
    // Eigen::Vector3d CO_offset = CO_Offset(U);
    // // - CO to center of gravity for hull only.
    // Eigen::Vector3d rg_hull = Eigen::Vector3d( -0.5, 0.0, -0.2) - CO_offset;
    // // - CO to payload location (front cabin).
    // Eigen::Vector3d rp      = Eigen::Vector3d( 0.75, 0.0, -0.2) - CO_offset;
    // // - CO to longditudinal center of flotation (LCF).
    // Eigen::Vector3d LCF_vec = Eigen::Vector3d( -0.2, 0.0,  0.0) - CO_offset;

    Eigen::Vector3d rg_hull = Eigen::Vector3d( -0.75, 0.0, -0.2);
    Eigen::Vector3d rp      = Eigen::Vector3d( 0.75, 0.0,  -0.2);
    Eigen::Vector3d LCF_vec = Eigen::Vector3d( -0.15, 0.0,  0.1);
    
    // ---------------------------
    // Ocean current and relative velocity
    // ---------------------------
    double psi = eta(5); 
    double u_c = V_c * std::cos(beta_c - psi);
    double v_c = V_c * std::sin(beta_c - psi);
    Eigen::VectorXd nu_c = Eigen::VectorXd::Zero(6);
    nu_c(0) = u_c;
    nu_c(1) = v_c;
    // - Relative velocity:
    Eigen::VectorXd nu_r = nu - nu_c;
    
    // Translational (nu1) and rotational (nu2) parts of nu:
    // nu1 = [u, v, w] and nu2 = [p, q, r]
    Eigen::Vector3d nu2 = nu.segment(3,3);
    
    // Compute current acceleration term: 
    // nu_c_dot = [ -Smtrx(nu2)*nu_c_head; zeros(3) ]
    Eigen::Vector3d nu_c_head = nu_c.segment(0,3);
    Eigen::Vector3d nu_c_dot_head = -Smtrx(nu2) * nu_c_head;
    Eigen::VectorXd nu_c_dot = Eigen::VectorXd::Zero(6);
    nu_c_dot.segment(0,3) = nu_c_dot_head;

    // ---------------------------
    // Inertia and trim calculations
    // ---------------------------
    // - Displaced volume
    double nabla = (m + mp) / rho; 
    // - Vessel draft
    double T_draft = nabla / (2 * Cb_pont * Beam_pont * L);  
    
    // Inertia dyadic for hull only at CG: Ig_CG = m * diag(R44^2, R55^2, R66^2)
    Eigen::Matrix3d Ig_CG = Eigen::Matrix3d::Zero();
    Ig_CG(0,0) = R44 * R44;
    Ig_CG(1,1) = R55 * R55;
    Ig_CG(2,2) = R66 * R66;
    
    // Corrected center of gravity including payload:
    Eigen::Vector3d rg_corrected = (m * rg_hull + mp * rp) / (m + mp);
    
    // Total inertia matrix at CO:
    Eigen::Matrix3d Ig = Ig_CG - m * (Smtrx(rg_hull) * Smtrx(rg_hull))
                               - mp * (Smtrx(rp) * Smtrx(rp));

    // ---------------------------
    // Azimuth pods / Pontoon data and control forces
    // ---------------------------
    // left pod lever arm (m)
    // ly1 = ly1 - CO_offset(1);         
    // // right pod lever arm (m)
    // ly2 = l2y + CO_offset(1);        
    // // forward displacement of pods (m)
    // lx  = lx + CO_offset(0); 

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
    // Added mass Coriolis matrix 
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
    double Xu = - M_sys(0,0) / T_surge;
    double Yv = - M_sys(1,1) / T_sway;
    double Zw = -2 * 0.3 * w3 * M_sys(2,2);
    double Kp_damp = -2 * 0.2 * w4 * M_sys(3,3);
    double Mq_damp = -2 * 0.4 * w5 * M_sys(4,4);
    double Nr = -M_sys(5,5) / T_yaw;

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
    // Forces and Moments generated by pods                       
    // ---------------------------
    Eigen::VectorXd tau = tau_pods(n, alpha); // <- tau given by nn input

    // ---------------------------
    // Cross-flow drag
    // ---------------------------
    Eigen::VectorXd tau_crossflow = crossFlowDrag(L, Beam_pont, T_draft, nu_r);
    
    //----------------------------
    // Wave drift disturbances
    //----------------------------
    Eigen::VectorXd tau_wave = Eigen::VectorXd::Zero(6); 
    if (waves_on) {
        tau_wave(0) = tau_wave_body_cached(0); 
        tau_wave(1) = tau_wave_body_cached(1); 
        tau_wave(5) = tau_wave_body_cached(2);  
    }

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

    Eigen::VectorXd force_term = tau + tau_damp + tau_crossflow + tau_wave - C_sys * nu_r - G * eta;
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
    if(propagate){

        xdot.resize(12);
        xdot << xdot_vel, xdot_pos;

        // ---------------------------
        // Set output mass matrix M 
        // ---------------------------
        M = M_sys;

        //----------------------------
        // Input matrix B (3x2). 
        // (Fossen Chapter 9.4 and 11.2)
        //----------------------------
        // Eigen::MatrixXd B_ = Eigen::MatrixXd::Zero(3,4);
        // double F1 = 0;
        // double F2 = 0;

        // if (n(0) >= 0) {
        //     F1 = k_pos;
        // } else if (n(0) < 0) {
        //     F1 = k_neg;
        // }

        // if (n(1) >= 0) {
        //     F2 = k_pos;
        // } else if (n(1) < 0) {
        //     F2 = k_neg;
        // }

        // // First row: contribution to surge (X)
        // B_(0,0) = F1*cos(alpha(0));
        // B_(0,1) = F2*cos(alpha(1));

        // // Second row: contribution to sway (Y)
        // B_(1,0) = F1*sin(alpha(0));
        // B_(1,1) = F2*sin(alpha(1));

        // // Third row: contribution to yaw moment (N)
        // B_(2,0) =  F1*(lx*sin(alpha(0) - ly1*cos(alpha(0))));
        // B_(2,1) =  F2*(lx*sin(alpha(1) - ly2*cos(alpha(1))));

        // //std::cout << "B : \n" << B_ << std::endl;
        // B = B_;
            // Lever arms considering pod radius
        double lx1 = lx_o  - pod_radius*cos(alpha(0));
        double lx2 = lx_o  - pod_radius*cos(alpha(1));
        double ly1 = ly1_o - pod_radius*sin(alpha(0));         
        double ly2 = ly2_o - pod_radius*sin(alpha(1));  

        Eigen::Matrix<double,3,4> B_e;
        B_e <<  1,    0,    1,    0,
              0,    1,    0,    1,
             -ly1*cos(alpha[0]), lx1*sin(alpha[0]), -ly2*cos(alpha[1]), lx2*sin(alpha[1]);

        B = B_e;
    }
    else {
        xdot_rk4.resize(12);
        xdot_rk4 << xdot_vel, xdot_pos;
    }
}

void RAN::update_n(Eigen::Vector2d& n, Eigen::Vector2d n_c, double h){

    Eigen::Vector2d n_new = n + h/T_n * (n_c - n);

    for (int j = 0; j < n.size(); ++j) {
        if (n_new(j) > n_max) n_new(j) = n_max;
        else if (n_new(j) < n_min) n_new(j) = n_min;
    }

    if (n1_fail) {
        n_new(0) = 0; 
    }
    if (n2_fail) {
        n_new(1) = 0; 
    }

    n = n_new;
}

void RAN::update_alpha(Eigen::Vector2d& alpha, Eigen::Vector2d alpha_c, double h){
    
    Eigen::Vector2d alpha_new = alpha + h/T_alpha * (alpha_c - alpha);

    for (int j = 0; j < alpha.size(); ++j) {
        if (alpha_new(j) > alpha_max) alpha_new(j) = alpha_max;
        else if (alpha_new(j) < alpha_min) alpha_new(j) = alpha_min;
    }

    if (n1_fail) {
        alpha_new(0) = 0; 
    }
    if (n2_fail) {
        alpha_new(1) = 0; 
    }

    alpha = alpha_new;
}

Eigen::VectorXd RAN::tau_pods(Eigen::Vector2d n, Eigen::Vector2d alpha){
    // Check fail state:
    if (n1_fail) {
        n(0) = 0;
    }
    if (n2_fail) {
        n(1) = 0;
    }

    // Lever arms considering pod radius
    double lx1 = lx_o  - pod_radius*cos(alpha(0));
    double lx2 = lx_o  - pod_radius*cos(alpha(1));
    double ly1 = ly1_o - pod_radius*sin(alpha(0));         
    double ly2 = ly2_o - pod_radius*sin(alpha(1));        

    // Thrusts from podded propellars
    // TODO: Reduce thrust if slipstream from one propellar hits the other.
    // (Need to take lever arms and hydrodynamic interaction into account)
    Eigen::VectorXd Thrusts = ThrustsFromRealativeN(n, thrust_coeffs);

    // Control forces and moments. 
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(6);
    tau(0) = Thrusts(0) * cos(alpha(0)) + Thrusts(1) * cos(alpha(1)); //X: Combined Surge 
    tau(1) = Thrusts(0) * sin(alpha(0)) + Thrusts(1) * sin(alpha(1)); //Y: Combined Sway 
    tau(5) = lx1 * Thrusts(0) * sin(alpha(0)) + lx2 * Thrusts(1) * sin(alpha(1))  //N: Yaw pod 
           - ly1 * Thrusts(0) * cos(alpha(0)) - ly2 * Thrusts(1) * cos(alpha(1)); 

    return tau;
}

void RAN::rk4(Eigen::VectorXd& x, double mp, double V_c, double beta_c, double h, Eigen::Vector2d n, Eigen::Vector2d alpha)
{

   propagate = false;
   
   // Compute k1
   update(x, mp, V_c, beta_c, h, n, alpha);
   Eigen::VectorXd k1 = h * xdot_rk4;

   // Compute k2
   update(x + 0.5 * k1, mp, V_c, beta_c, h, n, alpha);
   Eigen::VectorXd k2 = h * xdot_rk4;

   // Compute k3
   update(x + 0.5 * k2, mp, V_c, beta_c, h, n, alpha);
   Eigen::VectorXd k3 = h * xdot_rk4;

   // Compute k4
   update(x + k3, mp, V_c, beta_c, h, n, alpha);
   Eigen::VectorXd k4 = h * xdot_rk4;

   // Update state vector x
   x += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;

   propagate = true;

}

std::vector<bool> RAN::check_failstate() {
    std::vector<bool> failstate;
    failstate.push_back(n1_fail);
    failstate.push_back(n2_fail);
    return failstate;
}

void RAN::select_failure_mode() {
    std::string input;

    while (true) {
        std::cout << "\nSelect operating mode:\n";
        std::cout << "1 - Fully operational\n";
        std::cout << "2 - Pod 1 failure\n";
        std::cout << "3 - Pod 2 failure\n";
        std::cout << "Enter your choice (1/2/3): ";
        std::cin >> input;

        if (input == "1") {
            recover_n1();
            recover_n2();
            std::cout << "\n";
            break; 
        } else if (input == "2") {
            fail_state_n1();
            std::cout << "\n";
            break; 
        } else if (input == "3") {
            fail_state_n2();
            std::cout << "\n";
            break; 
        } else {
            std::cout << "Invalid choice. Please try again.\n";
        }
    }
}

//----------------Wave environment-----------------------
//
// Method 3 (slide deck): per DOF
//   x1dot = x2
//   x2dot = -wn^2 x1 - 2*z*wn x2 + K*w
//   eta_w = x2
// with unit white noise w (no extra "sigw" tuning knob).
//
// Drift: Method 3 uses a Wiener process (random walk); no Td tuning knob.
//

// Back-compat setter: surge, sway, yaw + drift.
// Heave/roll/pitch get reasonable defaults derived from surge/sway.
void RAN::set_wave_params(double wn_u, double z_u, double K_u,
                          double wn_v, double z_v, double K_v,
                          double wn_r, double z_r, double K_r,
                          double sigma_drift_X_in, double sigma_drift_Y_in, double sigma_drift_N_in)
{
    // Map to 6-DOF arrays
    wn[0]  = wn_u;  zeta[0] = z_u;  Kg[0] = K_u;   // x (surge-like WF)
    wn[1]  = wn_v;  zeta[1] = z_v;  Kg[1] = K_v;   // y (sway-like WF)

    // Heave default: slightly stiffer & more damped than surge
    wn[2]  = std::max(1.0, 0.9 * wn_u);
    zeta[2]= std::max(0.35, z_u + 0.10);
    Kg[2]  = 0.5 * (K_u + K_v);

    // Roll/pitch defaults: small amplitude
    wn[3]  = std::max(0.8, 0.8 * wn_u);  zeta[3] = 0.25;  Kg[3] = 0.03; // φ
    wn[4]  = std::max(0.8, 0.8 * wn_v);  zeta[4] = 0.25;  Kg[4] = 0.03; // θ

    // Yaw from inputs
    wn[5]  = wn_r;  zeta[5] = z_r;  Kg[5] = K_r;   // ψ

    // Drift intensities (BODY frame) for Wiener process
    sigma_drift_X = sigma_drift_X_in;
    sigma_drift_Y = sigma_drift_Y_in;
    sigma_drift_N = sigma_drift_N_in;
}

// Full 6-DOF setter. Use this if you want explicit control of z, φ, θ too.
void RAN::set_wave_params6(
    double wn_x,     double z_x,     double K_x,
    double wn_y,     double z_y,     double K_y,
    double wn_z,     double z_z,     double K_z,
    double wn_phi,   double z_phi,   double K_phi,
    double wn_theta, double z_theta, double K_theta,
    double wn_psi,   double z_psi,   double K_psi,
    double sigma_drift_X_in, double sigma_drift_Y_in, double sigma_drift_N_in)
{
    wn   = {wn_x, wn_y, wn_z, wn_phi, wn_theta, wn_psi};
    zeta = {z_x , z_y , z_z , z_phi , z_theta , z_psi };
    Kg   = {K_x , K_y , K_z , K_phi , K_theta , K_psi };

    sigma_drift_X = sigma_drift_X_in;
    sigma_drift_Y = sigma_drift_Y_in;
    sigma_drift_N = sigma_drift_N_in;
}

// 2nd-order WF integrator for a single DOF (Method 3). Outputs eta_dot (x2dot) in xddot_out.
// State meaning (internal):
//   x[0] = x1 (auxiliary)
//   x[1] = x2 = eta_w
static inline void wf_step_pair(Eigen::Vector2d& x,
                                double wn, double z, double K,
                                double dt, std::mt19937_64& rng,
                                std::normal_distribution<double>& N01,
                                double& xddot_out)
{
    const double dt_safe     = std::max(1e-12, dt);
    const double inv_sqrt_dt = 1.0 / std::sqrt(dt_safe);

    // Unit-intensity white noise (continuous-time approximation)
    const double w = N01(rng) * inv_sqrt_dt;

    const double x1 = x[0];
    const double x2 = x[1]; // x2 = eta_w

    // Method 3 state derivatives
    const double x1dot = x2;
    const double x2dot = -(wn * wn) * x1 - 2.0 * z * wn * x2 + K * w;

    // Keep output variable for interface: now returns eta_dot = x2dot
    xddot_out = x2dot;

    // Forward Euler step
    x[0] = x1 + dt_safe * x1dot;
    x[1] = x2 + dt_safe * x2dot;
}

void RAN::wave_step_WF(double dt)
{
    if (!waves_on) {
        eta_w_6.setZero();
        etadot_w_6.setZero();
        etaddot_w_6.setZero();

        eta_w_EN.setZero();
        etadot_w_EN.setZero();
        etaddot_w_EN.setZero();
        return;
    }

    const double dt_safe = std::max(1e-12, dt);

    // Advance 6 DOFs: x, y, z, φ, θ, ψ
    for (int i = 0; i < 6; ++i) {
        double eta_dot_i = 0.0;

        // Method 3 step: eta_dot_i gets x2dot (since eta_w = x2)
        wf_step_pair(xw[i], wn[i], zeta[i], Kg[i], dt_safe, rng, N01, eta_dot_i);

        // Method 3 output mapping: eta_w = x2
        eta_w_6[i]    = xw[i][1];
        etadot_w_6[i] = eta_dot_i;

        // Numeric estimate (optional; inherently noisy with white-noise-driven models)
        etaddot_w_6[i] = (etadot_w_6[i] - etadot_w_prev[i]) / dt_safe;
        etadot_w_prev[i] = etadot_w_6[i];
    }

    // Back-compat 3-vectors: [x y ψ]
    eta_w_EN     << eta_w_6[0],     eta_w_6[1],     eta_w_6[5];
    etadot_w_EN  << etadot_w_6[0],  etadot_w_6[1],  etadot_w_6[5];
    etaddot_w_EN << etaddot_w_6[0], etaddot_w_6[1], etaddot_w_6[5];
}

// Drift: Method 3 (Wiener / random walk). BODY frame.
void RAN::wave_step_drift(double dt)
{
    if (!waves_on) { d_wave.setZero(); tau_wave_body_cached.setZero(); return; }

    const double dt_safe = std::max(1e-12, dt);
    const double sdt     = std::sqrt(dt_safe);

    // Wiener increments: dW ~ N(0, dt)
    const double dWX = sdt * N01(rng);
    const double dWY = sdt * N01(rng);
    const double dWN = sdt * N01(rng);

    // Random walk: d_{k+1} = d_k + sigma * dW
    d_wave[0] += sigma_drift_X * dWX;
    d_wave[1] += sigma_drift_Y * dWY;
    d_wave[2] += sigma_drift_N * dWN;

    // Cached constant drift over RK4 sub-steps
    tau_wave_body_cached = d_wave; // [X, Y, N] BODY
}
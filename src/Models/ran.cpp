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
    L    = 5.0;                         
    Beam = 3;                           
    m    = 800.0;  

    R44 = 0.4  * Beam;                  
    R55 = 0.25 * L;                    
    R66 = 0.25 * L;   

    T_sway = 1.0;                      
    T_yaw  = 1.0;     

    Umax   = 10;      

    Beam_pont = 0.70;                 
    y_pont    = 1.1;                    
    Cw_pont   = 1;                       
    Cb_pont   = 0.5;

    k_pos = 200;        
    k_neg = 200;       
    n_max =  1;           
    n_min = -1;           
    alpha_max = M_PI/2;   
    alpha_min = -M_PI/2; 

    T_n = 0.5;
    T_alpha = 0.5;

    n = Eigen::VectorXd::Zero(2);
    alpha = Eigen::VectorXd::Zero(2);

    M = Eigen::MatrixXd::Zero(6, 6);
    B = Eigen::MatrixXd::Zero(3, 2);
    xdot = Eigen::VectorXd::Zero(12);

    propagate = true;
}

void RAN::update(const Eigen::VectorXd x, double mp, double V_c, double beta_c,
              double h, Eigen::VectorXd n_c, Eigen::VectorXd alpha_c)
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
    Eigen::Vector3d CO_offset = CO_Offset(U);
    // - CO to center of gravity for hull only.
    Eigen::Vector3d rg_hull = Eigen::Vector3d( -0.5, 0.0, -0.2) - CO_offset;
    // - CO to payload location (front cabin).
    Eigen::Vector3d rp      = Eigen::Vector3d( 0.75, 0.0, -0.2) - CO_offset;
    // - CO to longditudinal center of flotation (LCF).
    Eigen::Vector3d LCF_vec = Eigen::Vector3d( -0.2, 0.0,  0.0) - CO_offset;
    
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
    
    // Total inertia matrix at CG:
    Eigen::Matrix3d Ig = Ig_CG - m * (Smtrx(rg_hull) * Smtrx(rg_hull))
                               - mp * (Smtrx(rp) * Smtrx(rp));

    // ---------------------------
    // Azimuth pods / Pontoon data and control forces
    // ---------------------------
    // left pod lever arm (m)
    double ly1 = y_pont - CO_offset(1);         
    // right pod lever arm (m)
    double ly2 = -y_pont + CO_offset(1);        
    // forward displacement of pods (m)
    double lx  = -1.1 + CO_offset(0);   

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
    double Xu = - 200 / Umax;
    double Yv = -M_sys(1,1) / T_sway;
    double Zw = -2 * 0.3 * w3 * M_sys(2,2);
    double Kp_damp = -2 * 0.2 * w4 * M_sys(3,3);
    double Mq_damp = -2 * 0.4 * w5 * M_sys(4,4);
    double Nr = -M_sys(5,5) / T_yaw;
    

    // Compute propeller speeds and angles
    if (propagate){
        update_n(n_c, h);
        update_alpha(alpha_c, h);
    }

    // Thrusts from podded propellars
    Eigen::VectorXd Thrusts = ThrustsFromRealativeN(n);

    // Control forces and moments. 
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(6);
    tau(0) = Thrusts(0) * cos(alpha(0)) + Thrusts(1) * cos(alpha(1)); //X: Surge 
    tau(1) = Thrusts(0) * sin(alpha(0)) + Thrusts(1) * sin(alpha(1)); //Y: Sway 
    tau(5) = lx * (Thrusts(0) * sin(alpha(0)) + Thrusts(1) * sin(alpha(1)))
          - (ly1 * Thrusts(0) * cos(alpha(0)) + ly2 * Thrusts(1) * cos(alpha(1))); //N: Yaw

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
        Eigen::MatrixXd B_ = Eigen::MatrixXd::Zero(3,2);
        double F1 = 0;
        double F2 = 0;

        if (n(0) >= 0) {
            F1 = k_pos;
        } else if (n(0) < 0) {
            F1 = k_neg;
        }

        if (n(1) >= 0) {
            F2 = k_pos;
        } else if (n(1) < 0) {
            F2 = k_neg;
        }

        // First row: contribution to surge (X)
        B_(0,0) = F1*cos(alpha(0));
        B_(0,1) = F2*cos(alpha(1));

        // Second row: contribution to sway (Y)
        B_(1,0) = F1*sin(alpha(0));
        B_(1,1) = F2*sin(alpha(1));

        // Third row: contribution to yaw moment (N)
        B_(2,0) =  F1*(lx*sin(alpha(0) - ly1*cos(alpha(0))));
        B_(2,1) =  F2*(lx*sin(alpha(1) - ly2*cos(alpha(1))));

        //std::cout << "B : \n" << B_ << std::endl;
        B = B_;

    }
    else {
        xdot_rk4.resize(12);
        xdot_rk4 << xdot_vel, xdot_pos;
    }
}

void RAN::update_n(Eigen::VectorXd n_c, double h){

    Eigen::VectorXd n_new = n + h/T_n * (n_c - n);

    for (int j = 0; j < n.size(); ++j) {
        if (n_new(j) > n_max) n_new(j) = n_max;
        else if (n_new(j) < n_min) n_new(j) = n_min;
    }

    n = n_new;
}

void RAN::update_alpha(Eigen::VectorXd alpha_c, double h){
    
    Eigen::VectorXd alpha_new = alpha + h/T_alpha * (alpha_c - alpha);

    for (int j = 0; j < alpha.size(); ++j) {
        if (alpha_new(j) > alpha_max) alpha_new(j) = alpha_max;
        else if (alpha_new(j) < alpha_min) alpha_new(j) = alpha_min;
    }

    alpha = alpha_new;
}

void RAN::rk4(Eigen::VectorXd& x, double mp, double V_c, double beta_c, double h, Eigen::VectorXd n_c, Eigen::VectorXd alpha_c)
{
   propagate = false;
   
   // Compute k1
   update(x, mp, V_c, beta_c, h, n_c, alpha_c);
   Eigen::VectorXd k1 = h * xdot_rk4;

   // Compute k2
   update(x + 0.5 * k1, mp, V_c, beta_c, h, n_c, alpha_c);
   Eigen::VectorXd k2 = h * xdot_rk4;

   // Compute k3
   update(x + 0.5 * k2, mp, V_c, beta_c, h, n_c, alpha_c);
   Eigen::VectorXd k3 = h * xdot_rk4;

   // Compute k4
   update(x + k3, mp, V_c, beta_c, h, n_c, alpha_c);
   Eigen::VectorXd k4 = h * xdot_rk4;

   // Update state vector x
   x += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;

   propagate = true;

}

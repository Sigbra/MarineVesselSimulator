#include "motion_control.hpp"
#include "utilities.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

//Super simple PID controllerfor DP
std::vector<double> tau_XYN_PID( 
    double h,
    double xn_d, double yn_d, double psi_d,
    double xn, double yn, double psi,
    double &z_xn, double &z_yn, double &z_psi,
    double &prev_error_xn, double &prev_error_yn, double &prev_error_psi) 
    {

    // Position error in NED frame
    double error_xn_NED =  xn_d - xn;
    double error_yn_NED =  yn_d - yn;

    // Position error in BODY frame
    // RT = [cos sin;
    //     -sin cos]
    double error_xn =  cos(psi) * error_xn_NED + sin(psi) * error_yn_NED;  // Surge error (m)
    double error_yn = -sin(psi) * error_xn_NED + cos(psi)* error_yn_NED;  // Sway error (m)

    // Compute heading error (BODY)
    double error_psi = ssa(psi_d - psi);

    // --- PID Gains ---
    // Surge and sway gains (N/m, N/(m·s), N·s/m)
    const double Kp_xn = 6.0;
    const double Ki_xn = 0.0; //Integral action calculations are faulty
    const double Kd_xn = 2.0;

    const double Kp_yn = 6.0;
    const double Ki_yn = 0.0;
    const double Kd_yn = 2.0;
    
    // Yaw gains (N·m/rad, N·m/(rad·s), N·m·s/rad)
    const double Kp_psi = 6.0;
    const double Ki_psi = 0.0;
    const double Kd_psi = 2.0;

    // --- Compute Derivative Terms ---
    double d_error_xn  = (error_xn - prev_error_xn)   / h;
    double d_error_yn  = (error_yn - prev_error_yn)   / h;
    double d_error_psi = (error_psi - prev_error_psi) / h;

    // --- PID Control Law ---
    double tau_X = (Kp_xn  * error_xn  + Kd_xn  * d_error_xn  + Ki_xn  * z_xn);   // Surge force (N)
    double tau_Y = (Kp_yn  * error_yn  + Kd_yn  * d_error_yn  + Ki_yn  * z_yn);   // Sway force (N)
    double tau_N = (Kp_psi * error_psi + Kd_psi * d_error_psi + Ki_psi * z_psi);  // Yaw moment (N·m)

    // --- Update Persistent Variables ---
    prev_error_xn  = error_xn;
    prev_error_yn  = error_yn;
    prev_error_psi = error_psi;

    return {tau_X, tau_Y, tau_N};
}

//PID heading autopilot (Nomoto model: M(6,6) = T/K)
std::vector<double> tau_XN_PID(Eigen::MatrixXd M, double psi, double z_psi, double psi_d, double r, double r_d, double a_d){
    
    // - Nomoto time constant
    double T = 1;
    // - Nomoto gain constant
    double K = T / M(5,5);

    // - Closed-loop natural frequency (rad/s) (Guess)
    double wn = 1.5;
    // - Closed-loop relative damping factor (-) (Guess)
    double zeta = 1.0;

    // - Proportional gain
    double Kp = M(5,5)*pow(wn, 2);
    // - Derivative gain
    double Kd = M(5,5) * (2 * zeta * wn - 1/T);
    // - Derivative time constant
    double Td = Kd / Kp;
    // - Integral time constant
    double Ti = 10 / wn;

    // Desired forces and moment
    double tau_X = 10; 
    double tau_Y = 0;  
    double tau_N = (T/K) * a_d + (1/K) * r_d -        
                    Kp * (ssa(psi - psi_d) + 
                    Td * (r - r_d) + (1/Ti) * z_psi); 

    return {tau_X, tau_Y, tau_N};
}


#include "motion_control.hpp"
#include "utilities.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

std::vector<double> SISO_linear_PID_Control(
    double h,
    double xn_d, double yn_d, double psi_d,
    double xn, double yn, double psi,
    double &z_xn, double &z_yn, double &z_psi,
    double &prev_error_xn, double &prev_error_yn, double &prev_error_psi) 
    {

    // Position error in NED frame
    double error_xn_NED =  xn - xn_d;
    double error_yn_NED =  yn - yn_d;

    // Position error in BODY frame
    double cos_psi = cos(psi);
    double sin_psi = sin(psi);
    double error_xn = cos_psi * error_xn_NED + sin_psi * error_yn_NED;   // Surge error (m)
    double error_yn = -sin_psi * error_xn_NED + cos_psi * error_yn_NED;  // Sway error (m)

    // Compute heading error (BODY)
    double error_psi = ssa(psi_d - psi);

    // --- PID Gains ---
    // Surge and sway gains (N/m, N/(m·s), N·s/m)
    const double Kp_xn = 5.0;
    const double Ki_xn = 0.0;
    const double Kd_xn = 1.0;

    const double Kp_yn = 0.0;
    const double Ki_yn = 0.0;
    const double Kd_yn = 0.0;
    
    // Yaw gains (N·m/rad, N·m/(rad·s), N·m·s/rad)
    const double Kp_psi = 10.0;
    const double Ki_psi = 0.1;
    const double Kd_psi = 1.0;

    // --- Compute Derivative Terms ---
    double d_error_xn  = (error_xn - prev_error_xn)   / h;
    double d_error_yn  = (error_yn - prev_error_yn)   / h;
    double d_error_psi = (error_psi - prev_error_psi) / h;

    // --- PID Control Law ---
    double tau_X = 0.5 * xn_d  - (Kp_xn  * error_xn  + Kd_xn  * d_error_xn  + Ki_xn  * z_xn);   // Surge force (N)
    double tau_Y = 0.5 * yn_d  - (Kp_yn  * error_yn  + Kd_yn  * d_error_yn  + Ki_yn  * z_yn);   // Sway force (N)
    double tau_N = 0.5 * psi_d - (Kp_psi * error_psi + Kd_psi * d_error_psi + Ki_psi * z_psi);  // Yaw moment (N·m)

    // --- Update Persistent Variables ---
    prev_error_xn  = error_xn;
    prev_error_yn  = error_yn;
    prev_error_psi = error_psi;

    return {tau_X, tau_Y, tau_N};
}

#ifndef MOTIONCONTROL_HPP
#define MOTIONCONTROL_HPP

#include <vector>
#include <string>

std::vector<double> tau_XYN_PID(
    double h,
    double xn_d, double yn_d, double psi_d,
    double xn, double yn, double psi,
    double &z_xn, double &z_yn, double &z_psi,
    double &prev_error_xn, double &prev_error_yn, double &prev_error_psi);

// Stub
std::vector<double> tau_XN_PID();

#endif 
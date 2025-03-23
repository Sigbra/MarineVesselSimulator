#include "Control/PID_position_motion_control.hpp"
#include "Utilities/calculations.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

//-------------------------
// PositionPIDController Implementation
//-------------------------
PositionPIDController::PositionPIDController(double Kp_xn, double Ki_xn, double Kd_xn,
    double Kp_yn, double Ki_yn, double Kd_yn,
    double Kp_psi, double Ki_psi, double Kd_psi)
: Kp_xn_(Kp_xn), Ki_xn_(Ki_xn), Kd_xn_(Kd_xn),
Kp_yn_(Kp_yn), Ki_yn_(Ki_yn), Kd_yn_(Kd_yn),
Kp_psi_(Kp_psi), Ki_psi_(Ki_psi), Kd_psi_(Kd_psi),
z_xn_(0.0), z_yn_(0.0), z_psi_(0.0),
prev_error_xn_(0.0), prev_error_yn_(0.0), prev_error_psi_(0.0)
{
}

std::vector<double> PositionPIDController::update(double h,
double xn_d, double yn_d, double psi_d,
double xn, double yn, double psi) {
// Compute position error in NED frame.
double error_xn_NED = xn_d - xn;
double error_yn_NED = yn_d - yn;

// Rotate errors to the BODY frame (using current heading psi)
double error_xn =  cos(psi) * error_xn_NED + sin(psi) * error_yn_NED;  // Surge error
double error_yn = -sin(psi) * error_xn_NED + cos(psi) * error_yn_NED;  // Sway error

// Compute heading error (wrapped to [-pi, pi])
double error_psi = ssa(psi_d - psi);

// Compute derivative terms.
double d_error_xn = (error_xn - prev_error_xn_) / h;
double d_error_yn = (error_yn - prev_error_yn_) / h;
double d_error_psi = (error_psi - prev_error_psi_) / h;

// Integrate errors (here you can later add low-pass filtering if desired)
z_xn_ += error_xn * h;
z_yn_ += error_yn * h;
z_psi_ += error_psi * h;

// PID Control Law for surge, sway, and yaw.
double tau_X = Kp_xn_ * error_xn + Kd_xn_ * d_error_xn + Ki_xn_ * z_xn_;
double tau_Y = Kp_yn_ * error_yn + Kd_yn_ * d_error_yn + Ki_yn_ * z_yn_;
double tau_N = Kp_psi_ * error_psi + Kd_psi_ * d_error_psi + Ki_psi_ * z_psi_;

// Store current errors for next derivative calculation.
prev_error_xn_ = error_xn;
prev_error_yn_ = error_yn;
prev_error_psi_ = error_psi;

return {tau_X, tau_Y, tau_N};
}

void PositionPIDController::reset() {
z_xn_ = z_yn_ = z_psi_ = 0.0;
prev_error_xn_ = prev_error_yn_ = prev_error_psi_ = 0.0;
}
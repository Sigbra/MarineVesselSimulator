#include "Control/PID_MIMO_motion_control.hpp"
#include "Utilities/calculations.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

MIMOPIDController::MIMOPIDController(){

    // PID states;
    nu_prev_error = Eigen::VectorXd::Zero(3);
    nu_der = Eigen::VectorXd::Zero(3);
    nu_z = Eigen::VectorXd::Zero(3);

    // Tunable params
    Omega_b = Eigen::MatrixXd::Zero(3, 3);
    Omega_b(0, 0) = 0.08;
    Omega_b(1, 1) = 0.08;
    Omega_b(2, 2) = 2;

    Z = Eigen::MatrixXd::Zero(3, 3);
    Z(0, 0) = 0.04;
    Z(1, 1) = 0.04;
    Z(2, 2) = 0.4;

    // Natural Frequencies
    Omega_n = Eigen::MatrixXd::Zero(3,3);
    for (int i=0; i < 3; ++i) {
        double w_bi = Omega_b(i, i);
        double z_i = Z(i, i);
        Omega_n(i, i) = w_bi / (sqrt(1-2*std::pow(z_i, 2) + sqrt(4*std::pow(z_i,4) - 4*std::pow(z_i, 2) + 2)));
    }
}


std::vector<double> MIMOPIDController::update(double h, double xn_d, double yn_d, double psi_d,
                                              Eigen::MatrixXd M, Eigen::VectorXd eta, Eigen::VectorXd nu)
{   
    assert(M.rows() >= 6 && M.cols() >= 6 && "Error: M must be at least 6x6!");
    assert(eta.size() >= 6 && nu.size() >= 6 && "Error: eta and nu must have at least 6 elements!");
    assert(h > 1e-6 && "Error: Time step h is too small!");

    // Extract current position and heading (NED frame)
    double xn = nu(0);
    double yn = nu(1);
    double psi = nu(5);

    // Compute position error in NED frame
    double error_xn_NED = xn_d - xn;
    double error_yn_NED = yn_d - yn;
    
    // Rotate errors to the BODY frame (using current heading psi)
    double error_xn =  cos(psi) * error_xn_NED + sin(psi) * error_yn_NED;  // Surge error
    double error_yn = -sin(psi) * error_xn_NED + cos(psi) * error_yn_NED;  // Sway error

    // Compute heading error (wrapped to [-pi, pi])
    double error_psi = ssa(psi_d - psi);

    // Error vector in body frame
    Eigen::VectorXd nu_error(3);
    nu_error << error_xn, error_yn, error_psi;
    
    // M for relevant states only
    Eigen::MatrixXd M3(3, 3);
    M3 << M(0, 0), M(0, 1), M(0, 5),
          M(1, 0), M(1, 1), M(1, 5),
          M(5, 0), M(5, 1), M(5, 5);

    // PID Gains
    Eigen::MatrixXd Kp = M3*Omega_n*Omega_n;
    Eigen::MatrixXd Kd = 2*M3*Z*Omega_n;     // No C or D compensation implemented here for now. 
    Eigen::MatrixXd Ki = (Kp*Omega_n)/10;

    //Low pass filtered derivative term
    double alpha = 0.2; //More filtering 0 < alpha < 1 No filtering.
    nu_der = alpha * ((nu_error - nu_prev_error) / h) + (1 - alpha) * nu_der;

    //Anti-windup clamping
    double integral_limit = 10.0;  // Limit for integral accumulation.
    nu_z += nu_error * h;
    nu_z = nu_z.cwiseMax(-integral_limit).cwiseMin(integral_limit);  // Clamping

    // tau PID for X, Y and N
    Eigen::VectorXd tau_XYN = Kp*nu_error + Kd*nu_der + Ki*nu_z;

    // Store error for next time
    nu_prev_error = nu_error;

    std::vector<double> tau_XYN_vec; 
    for (int i = 0; i < tau_XYN.size(); ++i) {
        tau_XYN_vec.push_back(tau_XYN(i)); 
    }

    return tau_XYN_vec;
}
    
void MIMOPIDController::reset(){
    nu_prev_error.setZero();
    nu_der.setZero();
    nu_z.setZero();
}
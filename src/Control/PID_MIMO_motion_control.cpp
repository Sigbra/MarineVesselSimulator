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
    pos_z = Eigen::Vector2d::Zero();

    // Tunable params
    Omega_b = Eigen::MatrixXd::Zero(3, 3);
    Omega_b(0, 0) = 0.2;  // ideal: 0.2
    Omega_b(1, 1) = 0.3;  // ideal: 0.3
    Omega_b(2, 2) = 1.1;   // ideal: 1.2

    Z = Eigen::MatrixXd::Zero(3, 3);
    Z(0, 0) = 0.8; 
    Z(1, 1) = 0.7; //ideal:0.7
    Z(2, 2) = 1.1; // ideal: 1.2

    // Natural Frequencies
    Omega_n = Eigen::MatrixXd::Zero(3,3);
    for (int i=0; i < 3; ++i) {
        double w_bi = Omega_b(i, i);
        double z_i = Z(i, i);
        Omega_n(i, i) = w_bi / (sqrt(1-2*std::pow(z_i, 2) + sqrt(4*std::pow(z_i,4) - 4*std::pow(z_i, 2) + 2)));
    }
}


std::vector<double> MIMOPIDController::update(
    double h,
    double xn_d, double yn_d, double psi_d,
    const Eigen::MatrixXd& M,
    const Eigen::VectorXd& eta,
    const Eigen::VectorXd& nu,
    double V_c, double beta_c)
{
    //—— 1) Sanity
    assert(M.rows() >= 6 && M.cols() >= 6 && "M must be 6×6+");
    assert(eta.size() >= 6 && nu.size() >= 6 && "eta, nu need 6 elems");
    assert(h > 1e-6 && "h too small");

    //—— 2) Extract pose from eta
    double xn  = eta(0);
    double yn  = eta(1);
    double psi = eta(5);

    //—— 3) Outer-loop: position error in NED
    double ex_N = xn_d - xn;
    double ey_N = yn_d - yn;
    double e_psi = ssa(psi_d - psi);

    // Rotate position error into BODY frame
    double c = std::cos(psi), s = std::sin(psi);
    double ex =  c*ex_N + s*ey_N;
    double ey = -s*ex_N + c*ey_N;

    //—— 4) Compute desired body‐frame velocities (simple P)
    const double Kpx    = 1;   // [1/s]
    const double Kpy    = 1;   // [1/s] 
    const double Kp_psi = 1;   // [rad/s per rad]
    
    double u_d = Kpx * ex;
    double v_d = Kpy * ey; 
    double r_d = Kp_psi * e_psi;

    //—— 5) Transform ocean current into BODY frame
    Eigen::Vector2d Vc_NED(V_c*std::cos(beta_c),
                           V_c*std::sin(beta_c));
    Eigen::Matrix2d Rbn;  // body ← NED
    Rbn <<  c,  s,
          - s,  c;
    Eigen::Vector2d Vc_body = Rbn * Vc_NED;

    //—— 6) Extract measured body velocities & form relative vel.
    double u = nu(0), v = nu(1), r = nu(5);
    double u_r = u - Vc_body(0);
    double v_r = v - Vc_body(1);
    double r_r = r;            // simplification: no current‐yaw

    //—— 7) Velocity error for inner PID
    Eigen::Vector3d nu_d_vec(u_d, v_d, r_d);
    Eigen::Vector3d nu_r_vec(u_r, v_r, r_r);
    Eigen::Vector3d nu_error = nu_d_vec - nu_r_vec;

    //—— 8) Build 3×3 mass submatrix M3
    Eigen::Matrix3d M3;
    M3 << M(0,0), M(0,1), M(0,5),
          M(1,0), M(1,1), M(1,5),
          M(5,0), M(5,1), M(5,5);

    //—— 9) Recompute your PID gains from M3, Omega_n, Z
    Eigen::Matrix3d Kp = M3 * Omega_n * Omega_n;
    Eigen::Matrix3d Kd = 2 * M3 * Z * Omega_n;
    Eigen::Matrix3d Ki = (Kp * Omega_n) / 10.0;

    //—— 10) Low-pass derivative
    double alpha = 0.1;
    nu_der = alpha * ((nu_error - nu_prev_error) / h)
           + (1 - alpha) * nu_der;

    //—— 11) Integrator w/ anti-windup
    constexpr double integral_limit = 10.0; //10
    nu_z += nu_error * h;
    nu_z = nu_z.cwiseMax(-integral_limit)
           .cwiseMin( integral_limit);

    //—— 12) Compute control τ = [X, Y, N]^T
    Eigen::Vector3d tau_XYN = Kp*nu_error
                           + Kd*nu_der
                           + Ki*nu_z;

    //—— 13) Save for next step
    nu_prev_error = nu_error;

    //—— 14) Return as std::vector
    return std::vector<double>{
        tau_XYN(0),
        tau_XYN(1),
        tau_XYN(2)
    };
}

    
void MIMOPIDController::reset(){
    nu_prev_error.setZero();
    nu_der.setZero();
    nu_z.setZero();
    pos_z.setZero();
}
#ifndef MOTIONCONTROL_HPP
#define MOTIONCONTROL_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>

//-------------------------
// Position PID Controller
//-------------------------
class PositionPIDController {
    public:
        // Constructor (you can later extend these to be configurable)
        PositionPIDController(double Kp_xn = 6.0, double Ki_xn = 0.0, double Kd_xn = 2.0,
                              double Kp_yn = 6.0, double Ki_yn = 0.0, double Kd_yn = 2.0,
                              double Kp_psi = 6.0, double Ki_psi = 0.0, double Kd_psi = 2.0);
    
        // Update method calculates control efforts (tau_X, tau_Y, tau_N)
        // h: time step
        // (xn_d, yn_d, psi_d): desired states (position in NED and heading)
        // (xn, yn, psi): measured states
        std::vector<double> update(double h,
                                   double xn_d, double yn_d, double psi_d,
                                   double xn, double yn, double psi);
    
        // Reset the internal (integral and derivative) state.
        void reset();
    
    private:
        // PID gains for surge (x), sway (y), and heading (psi)
        double Kp_xn_, Ki_xn_, Kd_xn_;
        double Kp_yn_, Ki_yn_, Kd_yn_;
        double Kp_psi_, Ki_psi_, Kd_psi_;
    
        // Persistent states for integral accumulation
        double z_xn_, z_yn_, z_psi_;
        // For derivative calculation: previous error values.
        double prev_error_xn_, prev_error_yn_, prev_error_psi_;
    };

//-------------------------
// Heading PID Controller (Nomoto model based)
//-------------------------
class HeadingPIDController {
    public:
        // Constructor: T: Nomoto time constant, wn: closed-loop natural frequency, zeta: damping ratio.
        HeadingPIDController(double T = 1, double wn = 0.5, double zeta = 2.0);
    
        // Update method calculates control outputs given the current state.
        // M: mass (and added-mass) matrix (Eigen::MatrixXd)
        // psi: measured heading, psi_d: desired heading,
        // r: measured yaw rate, r_d: desired yaw rate,
        // a_d: desired acceleration (feedforward term)
        // h: time step (for integrating error)
        std::vector<double> update(double h, const Eigen::MatrixXd& M,
                                   double psi, double psi_d, double r, double r_d, double a_d);
    
        // Reset the internal (integral) state.
        void reset();
    
    private:
        double T_, wn_, zeta_;
        // Persistent integral state for heading error
        double z_psi_;
    };

#endif 
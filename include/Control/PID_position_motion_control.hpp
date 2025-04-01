#ifndef MOTIONCONTROLPOSITION_HPP
#define MOTIONCONTROLPOSITION_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>

class PositionPIDController {
    public:

        PositionPIDController(double Kp_xn = 6.0, double Ki_xn = 0.0, double Kd_xn = 2.0,
                              double Kp_yn = 6.0, double Ki_yn = 0.0, double Kd_yn = 2.0,
                              double Kp_psi = 6.0, double Ki_psi = 0.0, double Kd_psi = 2.0);
    
        std::vector<double> update(double h,
                                   double xn_d, double yn_d, double psi_d,
                                   double xn, double yn, double psi);
    
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


// Based on Algorithm 15.2 in Fossen. 
class MIMOPIDController {
    public:
        // Defines Omega and Z
        // Computes w
        MIMOPIDController();

        // Compute P gain matrix
        // Compute D gain matrix
        // Compute I gain matrix
        std::vector<double> MIMOPIDController::update(Eigen::MatrixXd M, Eigen::VectorXd eta, Eigen::VectorXd nu, double h);
    
        void reset();
    
    private:

        // Matrix of bandwidths
        Eigen::MatrixXd Omega_b;

        // Matrix of damping ratios
        Eigen::MatrixXd Z;

        // Natural frequencies
        Eigen::MatrixXd Omega_n;

        //Derivative state;
        Eigen::VectorXd nu_der;
        //Integral state;
        Eigen::VectorXd nu_int;

        
    };

#endif 
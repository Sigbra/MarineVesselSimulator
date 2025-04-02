#ifndef MOTIONCONTROLPOSITION_HPP
#define MOTIONCONTROLPOSITION_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>

// Based on Algorithm 15.2 in Fossen. 
class MIMOPIDController {
    public:
        // Defines Omega and Z
        // Computes w
        MIMOPIDController();

        // Compute P gain matrix
        // Compute D gain matrix
        // Compute I gain matrix
        std::vector<double> update(double h, double xn_d, double yn_d, double psi_d,
                            Eigen::MatrixXd M, Eigen::VectorXd eta, Eigen::VectorXd nu);
    
        void reset();
    
    private:

        // Matrix of bandwidths
        Eigen::MatrixXd Omega_b;

        // Matrix of damping ratios
        Eigen::MatrixXd Z;

        // Natural frequencies
        Eigen::MatrixXd Omega_n;

        //Derivative state;
        Eigen::VectorXd nu_prev_error;
        Eigen::VectorXd nu_der;
        //Integral state;
        Eigen::VectorXd nu_z;

        
    };

#endif 
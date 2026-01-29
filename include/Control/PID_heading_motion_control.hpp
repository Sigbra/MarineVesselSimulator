#ifndef MOTIONCONTROLHEADING_HPP
#define MOTIONCONTROLHEADING_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>

//-------------------------
// Heading PID Controller (Nomoto model based)
//-------------------------
class HeadingPIDController {
    public:
        // Constructor: T: Nomoto time constant, wn: closed-loop natural frequency, zeta: damping ratio.
        HeadingPIDController(double T = 15, double wn = 0.4, double zeta = 1.0);
    
        // Update method calculates control outputs given the current state.
        // M: mass (and added-mass) matrix (Eigen::MatrixXd)
        // psi: measured heading, psi_d: desired heading,
        // r: measured yaw rate, r_d: desired yaw rate,
        // a_d: desired acceleration (feedforward term)
        double update(double h, const Eigen::MatrixXd& M,
                      double psi, double psi_d, double r, double r_d, double a_d);
    
        // Reset the internal (integral and previous error) state.
        void reset();
    
    private:
        double T_, wn_, zeta_;
        double z_psi_;
    };

#endif 
// ran.hpp
#ifndef RAN_HPP
#define RAN_HPP

#include <Eigen/Dense>

// Function to compute dynamics (ran)
void ran(const Eigen::VectorXd x, const Eigen::VectorXd n_input, const Eigen::VectorXd alpha_input,
    double mp, double V_c, double beta_c,
    Eigen::VectorXd &xdot, double &U, Eigen::MatrixXd &M_out, Eigen::MatrixXd &B);

// Specialized RK4 integrator for the RAN model
void rk4_ran_step(Eigen::VectorXd& x, const Eigen::VectorXd& n, const Eigen::VectorXd& alpha,
                    double mp, double V_c, double beta_c, double h);
                    
#endif // RAN_HPP

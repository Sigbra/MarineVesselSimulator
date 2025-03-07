// ran.hpp
#ifndef RAN_HPP
#define RAN_HPP

#include <Eigen/Dense>

// Function declarations for helper utilities

// Skew-symmetric matrix (Smtrx)
Eigen::Matrix3d Smtrx(const Eigen::Vector3d &v);

// Transformation matrix Hmtrx
Eigen::MatrixXd Hmtrx(const Eigen::Vector3d &r);

// Added mass to Coriolis matrix (m2c)
Eigen::MatrixXd m2c(const Eigen::MatrixXd &MA, const Eigen::VectorXd &nu_r);

// Added mass surge
double addedMassSurge(double m, double L, double rho);

// Cross-flow drag (returns zero vector for this stub)
Eigen::VectorXd crossFlowDrag(double L, double B_pont, double T, const Eigen::VectorXd &nu_r);

// Euler angle to body angular rate transformation matrix
Eigen::MatrixXd eulerang(double phi, double theta, double psi);

// Rotation matrix from body-fixed frame to inertial frame (ZYX Euler angles)
Eigen::Matrix3d Rzyx(double phi, double theta, double psi);

// Helper function to find CO based on length and speed. 
Eigen::Vector3d CO_Offset(double U);

// Converting the relative propellar revs to the real ones. 
std::vector<double> nReal(std::vector<double> n_relative);

// Calculating thrusts based on relative propellar revs (n).
Eigen::VectorXd ThrustsFromRealativeN(std::vector<double> n_r);

// Function to compute dynamics (ran)
void ran(const Eigen::VectorXd x, const Eigen::VectorXd n_input, const Eigen::VectorXd alpha_input,
    double mp, double V_c, double beta_c,
    Eigen::VectorXd &xdot, double &U, Eigen::MatrixXd &M_out, Eigen::MatrixXd &B);

// Specialized RK4 integrator for the RAN model
void rk4_ran_step(Eigen::VectorXd& x, const Eigen::VectorXd& n, const Eigen::VectorXd& alpha,
                    double mp, double V_c, double beta_c, double h);
                    
#endif // RAN_HPP

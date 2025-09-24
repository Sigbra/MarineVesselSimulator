// ran.hpp
#ifndef MODELUTILITIES_HPP
#define MODELUTILITIES_HPP

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

// Skew-symmetric matrix (Smtrx)
Eigen::Matrix3d Smtrx(const Eigen::Vector3d &v);

// Transformation matrix Hmtrx
Eigen::MatrixXd Hmtrx(const Eigen::Vector3d &r);

Eigen::Matrix3d Tzyx(double phi, double theta);

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
Eigen::VectorXd ThrustsFromRealativeN(Eigen::Vector2d n_r, Eigen::VectorXd coeffs);

casadi::MX ThrustFromRelativeN_MX(casadi::MX n_i);

Eigen::VectorXd NOrderApprox(const std::string& csv_file, int order);

#endif

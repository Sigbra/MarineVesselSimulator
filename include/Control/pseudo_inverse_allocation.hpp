#ifndef PSEUDO_INVERSE_ALLOCATION_HPP
#define PSEUDO_INVERSE_ALLOCATION_HPP

#include <Eigen/Dense>
#include <cmath>

// Input:
//   B     : 3×4 (or 3×2) allocation matrix
//   tau   : 3×1 desired wrench [X; Y; N]
//   k_pos : thrust gain for forward rotation
//   k_neg : thrust gain for reverse rotation
// Returns: 4×1 vector [n1; n2; alpha1; alpha2]
std::vector<double> pseudo_inverse_allocation(const std::vector<double>& tau_XYN, const Eigen::MatrixXd& B, double k_pos, double k_neg);

#endif
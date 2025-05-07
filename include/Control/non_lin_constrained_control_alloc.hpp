#ifndef NLOPTCONTROLALLOC_HPP
#define NLOPTCONTROLALLOC_HPP

#include <vector>
#include <Eigen/Dense>

std::vector<double> NLOptControlAlloc(double tau_X, double tau_Y, double tau_N, double U, Eigen::Vector2d n, Eigen::Vector2d alpha, std::vector<bool> failstate);

#endif
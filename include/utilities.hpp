#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>

struct Waypoints {
    std::vector<double> x;
    std::vector<double> y;
};

double ssa(double angle);

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata);

#endif // UTILITIES_HPP

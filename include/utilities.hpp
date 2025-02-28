#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>

struct Waypoints {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> angle;
};

double ssa(double angle);

double deg2rad(double degrees);

double rad2deg(double radians);

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata);

void plotTrajectory();

#endif // UTILITIES_HPP

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>

struct Point {
    double x;
    double y;
};

struct Waypoints {
    std::vector<double> x;
    std::vector<double> y;
};

double ssa(double angle);

double deg2rad(double degrees);

double rad2deg(double radians);

Waypoints addIntermediateWaypoints(const Waypoints& input, double space);

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata, std::string filename);

void plotTrajectory();

void plotStateErrors();

void plotAngles();

#endif // UTILITIES_HPP

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>
#include "path_generation.hpp"
// Basic 2D point structure
struct Point2D {
    double x;
    double y;
};

double ssa(double angle);

double deg2rad(double degrees);

double rad2deg(double radians);

// Updated to use the new Waypoints type
//Waypoints addIntermediateWaypoints(const Waypoints& input, double space);

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata, std::string filename);

void plotPath(const Waypoints &path);

void plotTrajectory();

void plotStateErrors();

void plotAngles();

#endif // UTILITIES_HPP

#ifndef PLOTTING_UTILITIES_HPP
#define PLOTTING_UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>
#include "Utilities/calculations.hpp"

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata, std::string filename);

void plotPath(const Waypoints &path);

void plotTrajectory();

void plotStateErrors();

void plotAngles();

#endif // UTILITIES_HPP

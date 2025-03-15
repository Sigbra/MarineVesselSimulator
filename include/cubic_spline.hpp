#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <vector>
#include "utilities.hpp"

double computeCurvature(double dx, double dy, double ddx, double ddy);

std::vector<Point> generateCubicSpline(const Waypoints& wp, double max_curvature, int num_samples = 100);

#endif // SPLINE_HPP


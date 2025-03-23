#include "Utilities/calculations.hpp"
#include <cmath>
#include <iostream>

double ssa(double angle)
{
    if (std::abs(angle) > 360) {
        std::cerr << "Warning: Unusually large angle value detected: " << angle << std::endl;
        angle = std::fmod(angle, 2 * M_PI); 
    }
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;
    return angle;
}

double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}

Waypoints addIntermediateWaypoints(const Waypoints& input, double space) {
    Waypoints output;
    
    // Check if there are any waypoints to process
    if (input.empty())
        return output;
    
    // Always include the first waypoint
    output.push_back(input[0]);
    
    // Process each pair of consecutive waypoints
    for (size_t i = 1; i < input.size(); ++i) {
        double x1 = input[i-1].x;
        double y1 = input[i-1].y;
        double x2 = input[i].x;
        double y2 = input[i].y;
        
        // Compute the Euclidean distance between the two waypoints
        double dist = std::hypot(x2 - x1, y2 - y1);
        
        // If the distance exceeds the spacing, add intermediate waypoints
        if (dist > space) {
            // Determine the number of segments required
            int num_segments = static_cast<int>(std::ceil(dist / space));
            // Insert intermediate waypoints along the line
            for (int seg = 1; seg < num_segments; ++seg) {
                double t = static_cast<double>(seg) / num_segments;
                double new_x = x1 + t * (x2 - x1);
                double new_y = y1 + t * (y2 - y1);
                output.push_back({new_x, new_y});
            }
        }
        
        // Add the original waypoint
        output.push_back(input[i]);
    }
    
    return output;
}
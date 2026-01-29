#ifndef STRAIGHT_LINE_PATH_HPP
#define STRAIGHT_LINE_PATH_HPP

#include "Utilities/calculations.hpp"
#include <cmath>
#include <vector>

//----------------------------------------------------------
// StraightLinePath Class Declaration
//----------------------------------------------------------
// Generates a simple path/trajectory by combining waypoints
// using straight line segments.
//----------------------------------------------------------
class StraightLinePath {
    public:
        StraightLinePath();
    
        void updateWaypoints(const Waypoints& waypoints);
    
        PathTrackingInfo getClosestPoint(const Vector2D vessel, int& wpt_index);
    
        std::vector<Vector2D> samplePath(double delta) const;
    
        void printParameters() const;
    
    private:
        Waypoints waypoints_;
    
        PathPoint projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel);
    
        double alongTrackError(Vector2D vessel, PathPoint path_point);
        
        double crossTrackError(Vector2D vessel, PathPoint path_point);
    };

#endif
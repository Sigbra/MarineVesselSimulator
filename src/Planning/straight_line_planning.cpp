#include "Planning/straight_line_planning.hpp"
#include "Utilities/calculations.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

StraightLinePath::StraightLinePath() {}

void StraightLinePath::updateWaypoints(const Waypoints& waypoints) {
    if (waypoints.size() < 2)
        throw std::runtime_error("At least two waypoints are required.");
    waypoints_ = waypoints;
}

double StraightLinePath::alongTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double x_e = (vessel.x - path_point.pos.x)*std::cos(lamda) 
                +(vessel.y - path_point.pos.y)*std::sin(lamda);
    return x_e;
}

double StraightLinePath::crossTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double y_e = -(vessel.x - path_point.pos.x)*std::sin(lamda) 
                 +(vessel.y - path_point.pos.y)*std::cos(lamda);
    return y_e;
}

PathTrackingInfo StraightLinePath::getClosestPoint(const Vector2D vessel_position, int &wpt_index) {
    
    Vector2D vessel_pos = vessel_position;
    Vector2D wpt_prev   = waypoints_[wpt_index-1];
    Vector2D wpt        = waypoints_[wpt_index];
    Vector2D wpt_next   = waypoints_[wpt_index+1];

    PathPoint line_prev_point;
    double x_e_line_prev;
    double y_e_line_prev;
    double line_prev_error;

    PathPoint line_point;
    double x_e_line;
    double y_e_line;
    double line_error;

    PathTrackingInfo tracking_info;

    line_prev_point = projectOntoLine(wpt_prev, wpt, vessel_position);
    x_e_line_prev = alongTrackError(vessel_position, line_prev_point);
    y_e_line_prev = crossTrackError(vessel_position, line_prev_point);
    line_prev_error = std::sqrt(x_e_line_prev*x_e_line_prev + y_e_line_prev*y_e_line_prev);

    line_point = projectOntoLine(wpt, wpt_next, vessel_position);
    x_e_line = alongTrackError(vessel_position, line_point); 
    y_e_line = crossTrackError(vessel_position, line_point);
    line_error = std::sqrt(x_e_line*x_e_line + y_e_line*y_e_line);

    if (line_prev_error < line_error){
        tracking_info.point = line_prev_point;
        tracking_info.x_e = x_e_line_prev;
        tracking_info.y_e = y_e_line_prev;
    } 
    else {
        tracking_info.point = line_point;
        tracking_info.x_e = x_e_line;
        tracking_info.y_e = y_e_line;
        if (wpt_index < waypoints_.size()-1){
            wpt_index++;
        }
    } 

    return tracking_info;
}


PathPoint StraightLinePath::projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel) {
    PathPoint pp;
    Vector2D AB = B - A;
    double t = ((vessel - A).dot(AB)) / (AB.dot(AB));
    t = std::max(0.0, std::min(1.0, t));
    pp.pos = A + AB * t;
    pp.dpos = AB.normalized();
    pp.ddpos = Vector2D(0.0, 0.0);
    return pp;
}


std::vector<Vector2D> StraightLinePath::samplePath(double delta) const {
    return waypoints_;
}

void StraightLinePath::printParameters() const {
    std::cout << "Total waypoints: " << waypoints_.size() << "\n";
    for (size_t i = 0; i < std::min(waypoints_.size(), size_t(5)); ++i) {
        std::cout << "Waypoint " << i << ": (" << waypoints_[i].x << ", " << waypoints_[i].y << ")\n";
    }
}
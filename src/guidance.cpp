#include "guidance.hpp"
#include "utilities.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

GuidanceMethod::GuidanceMethod(){}
    
int GuidanceMethod::selectMethod() {
    std::cout << "Choose Guidance Method:" << std::endl;
    for (size_t i = 0; i < methods.size(); ++i) {
        std::cout << i + 1 << ". " << methods[i] << std::endl;
    }
    
    int choice = 0;
    while (true) {
        std::cout << "Enter the number of your choice: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
        }
        else if (choice > 0 && choice <= static_cast<int>(methods.size())) {
            break;
        }
        else {
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    return choice;
}

//-------------------------------Dynamic Positioning-------------------------------

DynamicPositioning::DynamicPositioning(const Waypoints &waypoints, double switch_radius)
    : wpt(waypoints), R_switch(switch_radius), wpt_index(1) // start with segment from 0 to 1
{
    if (wpt.size() < 2) {
        throw std::runtime_error("At least two waypoints are required for dynamic positioning to calculate heading.");
    }
    if (R_switch < 0) {
        throw std::runtime_error("R_switch must be larger than zero.");
    }
}

std::tuple<double, double, double> DynamicPositioning::update(double xn, double yn) {
    // Compute the error between current position and the active (goal) waypoint.
    double pos_x_error = wpt[wpt_index].x - xn;
    double pos_y_error = wpt[wpt_index].y - yn;
    double pos_error_BODY = std::sqrt(pos_x_error * pos_x_error + pos_y_error * pos_y_error);

    // If the error is below the switching threshold and we haven't reached the final waypoint,
    // update the active waypoint index.
    if (pos_error_BODY < R_switch && wpt_index < wpt.size() - 1) {
        wpt_index++;
    }

    // For dynamic positioning, compute the desired states using the current segment.
    double x_start = wpt[wpt_index-1].x;
    double y_start = wpt[wpt_index-1].y;

    double x_goal = wpt[wpt_index].x;
    double y_goal = wpt[wpt_index].y;

    double xn_d = x_goal;
    double yn_d = y_goal;
    double psi_d = std::atan2(y_goal - y_start, x_goal - x_start);

    return std::make_tuple(xn_d, yn_d, psi_d);
}

void DynamicPositioning::reset() {
    wpt_index = 1; // Reset to the first segment.
}

void DynamicPositioning::updateWaypoints(const Waypoints &newWaypoints) {
    if (newWaypoints.empty()) {
        throw std::runtime_error("Waypoints list cannot be empty.");
    }
    
    wpt = newWaypoints;
    reset(); // Reset to the first waypoint
}

//----------------------------------ALOS----------------------------------

ALOS::ALOS(const Waypoints &waypoints, double Delta_h, double gamma_h, double h, double R_switch)
    : Delta_h_(Delta_h), gamma_h_(gamma_h), h_(h), R_switch_(R_switch),
      k_(0), beta_hat_(0.0), refPath_(waypoints)
{
    if (Delta_h_ < 0) {
        throw std::runtime_error("Delta_h must be larger than zero.");
    }
    if (gamma_h_ < 0) {
        throw std::runtime_error("gamma_h must be larger than zero.");
    }
    if (R_switch_ < 0) {
        throw std::runtime_error("R_switch must be larger than zero.");
    }
    if (refPath_.empty()) {
        throw std::runtime_error("Reference path is empty.");
    }
}

std::tuple<double, double, bool> ALOS::update(double x, double y) {
    
    if (refPath_.empty()) {
        throw std::runtime_error("Reference path is empty.");
    }

    bool at_last_waypoint = false;
    int n = static_cast<int>(refPath_.size());

    int closest_index = k_;
    double min_dist_sq = std::numeric_limits<double>::max();
    for (int i = k_; i < n; ++i) {
        double dx = x - refPath_[i].x;
        double dy = y - refPath_[i].y;
        double dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_index = i;
        }
    }
    k_ = closest_index;

    // Path-tangential angle (pi_h)
    double pi_h = 0.0;
    if (k_ < n - 1){
        // Current ref point to the next.
        pi_h = std::atan2(refPath_[k_ + 1].y - refPath_[k_].y, refPath_[k_ + 1].x - refPath_[k_].x);
    } else if (k_ > 0) {
        // At the end, use the segment from the previous point.
        pi_h = std::atan2(refPath_[k_].y - refPath_[k_ - 1].y, refPath_[k_].x - refPath_[k_ - 1].x);
    }

    // Transform to path-tangent frame.
    double dx = x - refPath_[k_].x;
    double dy = y - refPath_[k_].y;
    double x_e = dx * std::cos(pi_h) + dy * std::sin(pi_h);
    double y_e = -dx * std::sin(pi_h) + dy * std::cos(pi_h);

    // Compute the desired heading.
    double psi_ref = pi_h - beta_hat_ - std::atan(y_e / Delta_h_);

    // Update the crab angle estimate.
    beta_hat_ = beta_hat_ + h_ * gamma_h_ * Delta_h_ * y_e / std::sqrt(Delta_h_ * Delta_h_ + y_e * y_e);

    // Check if we've reached the last waypoint
    if (k_ == n - 1 && min_dist_sq < R_switch_ * R_switch_) {
        at_last_waypoint = true;
    }
    
    return std::make_tuple(psi_ref, y_e, at_last_waypoint);
}

void ALOS::reset() {
    k_ = 0;
    beta_hat_ = 0.0;
}

void ALOS::updatePath(const Waypoints &newPath) {
    if (newPath.empty()) {
        throw std::runtime_error("Reference path cannot be empty.");
    }
    
    refPath_ = newPath;
    reset(); // Reset to the first point in the path
}

//----------------------------LOS Observer----------------------------

LOSObserver::LOSObserver(double h, double K_f)
    : h(h), K_f(K_f), LOSangle(0.0), LOSrate(0.0) {
    T_f = 1.0 / (K_f + 2 * std::sqrt(K_f) + 1);
    xi = LOSangle - LOSrate;
}

void LOSObserver::update(double LOScommand) {
    double PHI = std::exp(-h / T_f);
    LOSangle += h * (LOSrate + K_f * ssa(LOScommand - LOSangle));
    xi = PHI * xi + (1 - PHI) * LOSangle;
    LOSrate = LOSangle - xi;
}

double LOSObserver::getLOSAngle() const {
    return LOSangle;
}

double LOSObserver::getLOSRate() const {
    return LOSrate;
}
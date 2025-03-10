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

DynamicPositioning::DynamicPositioning(const Waypoints &waypoints, double switch_radious)
    : wpt(waypoints), R_switch(switch_radious), wpt_index(1) // start with segment from 0 to 1
{
    if (wpt.x.size() < 2 || wpt.y.size() < 2) {
        throw std::runtime_error("At least two waypoints are required for dynamic positioning to calculate heading.");
    }
    if (R_switch < 0) {
        throw std::runtime_error("R_switch must be larger than zero.");
    }
}

std::tuple<double, double, double> DynamicPositioning::update(double xn, double yn) {
    // Compute the error between current position and the active (goal) waypoint.
    double pos_x_error = wpt.x[wpt_index] - xn;
    double pos_y_error = wpt.y[wpt_index] - yn;
    double pos_error_BODY = std::sqrt(pos_x_error * pos_x_error + pos_y_error * pos_y_error);

    // If the error is below the switching threshold and we haven't reached the final waypoint,
    // update the active waypoint index.
    if (pos_error_BODY < R_switch && wpt_index < wpt.x.size() - 1) {
        wpt_index++;
    }

    // For dynamic positioning, compute the desired states using the current segment.
    double x_start = wpt.x[wpt_index-1];
    double y_start = wpt.y[wpt_index-1];

    double x_goal = wpt.x[wpt_index];
    double y_goal = wpt.y[wpt_index];

    double xn_d = x_goal;
    double yn_d = y_goal;
    double psi_d = std::atan2(y_goal - y_start, x_goal - x_start);

    return std::make_tuple(xn_d, yn_d, psi_d);
}

void DynamicPositioning::reset() {
    wpt_index = 1; // Reset to the first segment.
}

//----------------------------------ALOS----------------------------------

ALOS::ALOS(const Waypoints &wpt, double Delta_h, double gamma_h, double h, double R_switch)
    : wpt_(wpt), Delta_h_(Delta_h), gamma_h_(gamma_h), h_(h), R_switch_(R_switch),
      k_(0), beta_hat_(0.0)
{
    // Ensure that there are at least two waypoints.
    if (wpt_.x.size() < 2 || wpt_.y.size() < 2) {
        throw std::runtime_error("At least two waypoints are required.");
    }
    // Compute the minimum distance between consecutive waypoints.
    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < wpt_.x.size() - 1; i++) {
        double dx = wpt_.x[i+1] - wpt_.x[i];
        double dy = wpt_.y[i+1] - wpt_.y[i];
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < min_dist) {
            min_dist = dist;
        }
    }
    if (R_switch_ > min_dist) {
        throw std::runtime_error("The distances between the waypoints must be larger than R_switch.");
    }
    if (R_switch_ < 0) {
        throw std::runtime_error("R_switch must be larger than zero.");
    }
    if (Delta_h_ < 0) {
        throw std::runtime_error("Delta_h must be larger than zero.");
    }
    // Initialize the active waypoint.
    xk_ = wpt_.x[k_];
    yk_ = wpt_.y[k_];
}

std::tuple<double, double, bool> ALOS::update(double x, double y) {
    int n = static_cast<int>(wpt_.x.size());
    double xk_next, yk_next;
    // Determine the next waypoint.
    if (k_ < n - 1) {
        xk_next = wpt_.x[k_ + 1];
        yk_next = wpt_.y[k_ + 1];
    } else {
        // At the last waypoint, define a far-off "next" waypoint using the bearing from the last two.
        double bearing = std::atan2(wpt_.y[n - 1] - wpt_.y[n - 2],
                                    wpt_.x[n - 1] - wpt_.x[n - 2]);
        double R = 1e10;
        xk_next = wpt_.x[n - 1] + R * std::cos(bearing);
        yk_next = wpt_.y[n - 1] + R * std::sin(bearing);
    }

    // Compute the path-tangential angle (pi_h) with respect to North.
    double pi_h = std::atan2(yk_next - yk_, xk_next - xk_);
    
    // Transform the error to the path-tangential frame.
    double x_e = (x - xk_) * std::cos(pi_h) + (y - yk_) * std::sin(pi_h);
    double y_e = -(x - xk_) * std::sin(pi_h) + (y - yk_) * std::cos(pi_h);

    // Check the switching criterion: if the remaining along-track distance is less than R_switch, switch to the next waypoint.
    double d = std::sqrt((xk_next - xk_) * (xk_next - xk_) + (yk_next - yk_) * (yk_next - yk_));
    if ((d - x_e < R_switch_) && (k_ < n - 1)) {
        k_ = k_ + 1;
        xk_ = wpt_.x[k_];
        yk_ = wpt_.y[k_];
    }

    // Avoid division by zero for Delta_h_.
    if (std::abs(Delta_h_) < 1e-9) {
        std::cerr << "Warning: Delta_h is too small, avoiding division by zero!" << std::endl;
    }
    
    // Compute the desired heading (psi_ref) using the adaptive LOS guidance law.
    double psi_ref = pi_h - beta_hat_ - std::atan(y_e / Delta_h_);

    // Update the crab angle estimate.
    beta_hat_ = beta_hat_ + h_ * gamma_h_ * Delta_h_ * y_e / std::sqrt(Delta_h_ * Delta_h_ + y_e * y_e);

    // Determine if the vehicle is at the last waypoint (within R_switch).
    bool at_last_waypoint = false;
    if (k_ == n - 1) {
        double dist_to_last = std::sqrt((x - xk_) * (x - xk_) + (y - yk_) * (y - yk_));
        at_last_waypoint = (dist_to_last < R_switch_);
    }

    return std::make_tuple(psi_ref, y_e, at_last_waypoint);
}

void ALOS::reset() {
    k_ = 0;
    beta_hat_ = 0.0;
    xk_ = wpt_.x[k_];
    yk_ = wpt_.y[k_];
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
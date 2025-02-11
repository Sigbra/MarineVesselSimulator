#include "ALOSpsi.hpp"
#include "utilities.hpp"  // Assumes ssa(double) is declared here.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <vector>

std::pair<double, double> ALOSpsi(double x, double y,
                                  double Delta_h, double gamma_h, double h,
                                  double R_switch,
                                  const Waypoints &wpt) {
    // Static (persistent) variables: active waypoint index, active waypoint coordinates, and crab angle estimate.
    static bool initialized = false;
    static int k;           // Active waypoint index (0-indexed)
    static double xk, yk;   // Active waypoint coordinates
    static double beta_hat; // Estimate of the crab (current) angle

    // Initialization (runs only on the first call, or after a manual reset via "clear" in MATLAB)
    if (!initialized) {
        // Ensure there are at least two waypoints.
        if (wpt.x.size() < 2 || wpt.y.size() < 2) {
            throw std::runtime_error("At least two waypoints are required.");
        }
        // Compute the minimum distance between consecutive waypoints.
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < wpt.x.size() - 1; i++) {
            double dx = wpt.x[i+1] - wpt.x[i];
            double dy = wpt.y[i+1] - wpt.y[i];
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        if (R_switch > min_dist) {
            throw std::runtime_error("The distances between the waypoints must be larger than R_switch.");
        }
        if (R_switch < 0) {
            throw std::runtime_error("R_switch must be larger than zero.");
        }
        if (Delta_h < 0) {
            throw std::runtime_error("Delta_h must be larger than zero.");
        }
        beta_hat = 0.0;       // Initialize crab angle estimate.
        k = 0;                // Set the first waypoint (index 0) as active.
        xk = wpt.x[k];
        yk = wpt.y[k];
        std::cout << "Active waypoint:\n";
        std::cout << "  (x" << (k+1) << ", y" << (k+1) << ") = (" << xk << ", " << yk << ")\n";
        initialized = true;
    }

    // Total number of waypoints.
    int n = static_cast<int>(wpt.x.size());
    double xk_next, yk_next;
    if (k < n - 1) {  // There is a next waypoint.
        xk_next = wpt.x[k + 1];
        yk_next = wpt.y[k + 1];
    } else {  // If at the last waypoint, define a far-off "next" waypoint using the bearing from the last two.
        double bearing = std::atan2(wpt.y[n - 1] - wpt.y[n - 2], wpt.x[n - 1] - wpt.x[n - 2]);
        double R = 1e10;
        xk_next = wpt.x[n - 1] + R * std::cos(bearing);
        yk_next = wpt.y[n - 1] + R * std::sin(bearing);
    }

    // Compute the path-tangential (azimuth) angle (pi_h) with respect to North.
    double pi_h = std::atan2(yk_next - yk, xk_next - xk);

    // Compute along-track error x_e and cross-track error y_e.
    double x_e = (x - xk) * std::cos(pi_h) + (y - yk) * std::sin(pi_h);
    double y_e = -(x - xk) * std::sin(pi_h) + (y - yk) * std::cos(pi_h);

    // Check the switching criterion: if the remaining along-track distance is less than R_switch, switch to the next waypoint.
    double d = std::sqrt((xk_next - xk) * (xk_next - xk) + (yk_next - yk) * (yk_next - yk));
    if ((d - x_e < R_switch) && (k < n - 1)) {
        k = k + 1;
        xk = xk_next;
        yk = yk_next;
        std::cout << "Active waypoint updated:\n";
        std::cout << "  (x" << (k+1) << ", y" << (k+1) << ") = (" << xk << ", " << yk << ")\n";
    }

    // Compute the desired heading angle (psi_ref) using the adaptive LOS (ALOS) guidance law:
    //   psi_ref = pi_h - beta_hat - atan( y_e / Delta_h )
    double psi_ref = pi_h - beta_hat - std::atan(y_e / Delta_h);

    // Propagate the crab angle estimate (beta_hat) to the next time step.
    beta_hat = beta_hat + h * gamma_h * Delta_h * y_e / std::sqrt(Delta_h * Delta_h + y_e * y_e);

    // Return psi_ref and cross-track error y_e.
    return std::make_pair(psi_ref, y_e);
}

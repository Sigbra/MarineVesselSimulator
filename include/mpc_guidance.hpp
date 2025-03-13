#ifndef MPC_GUIDANCE_HPP
#define MPC_GUIDANCE_HPP

#include <vector>
#include <tuple>
#include <casadi/casadi.hpp>
#include "utilities.hpp"

class MPCGuidance {
public:
    // Constructor
    MPCGuidance(double h, double r_max, double U_dot_max, double T_final);
    
    // Initialization with waypoints (also resets the MPC state)
    void setWaypoints(const Waypoints &waypoints);
    
    // Update method: given the current position (x, y), calculates the desired heading and speed
    // Also returns if we've reached the final waypoint
    std::tuple<double, double, bool> update(double x, double y);
    
    // Reset the internal state
    void reset();
    
    // Get solution variables
    const std::vector<double>& getChiTrajectory() const { return chi_d_trajectory_; }
    const std::vector<double>& getSpeedTrajectory() const { return U_d_trajectory_; }
    const std::vector<double>& getXTrajectory() const { return X_d_trajectory_; }
    const std::vector<double>& getYTrajectory() const { return Y_d_trajectory_; }
    
    // Get current waypoint index
    int getCurrentWaypointIdx() const { return current_waypoint_idx_; }
    
private:
    // Waypoints
    Waypoints waypoints_;
    bool waypoints_set_;
    
    // Sampling time
    double h_;
    
    // Constraints
    double r_max_;          // Maximum turning rate [rad/s]
    double U_dot_max_;      // Maximum acceleration [m/s^2]
    double T_final_;        // Final time [s]
    
    // Number of discretization steps
    int N_;
    
    // Current active waypoint index
    int current_waypoint_idx_;
    
    // Flag indicating if we've reached the final waypoint
    bool at_final_waypoint_;
    
    // Solution trajectories
    std::vector<double> chi_d_trajectory_;
    std::vector<double> U_d_trajectory_;
    std::vector<double> X_d_trajectory_;
    std::vector<double> Y_d_trajectory_;
    
    // Solver and related variables
    casadi::Function solver_;
    bool solver_initialized_;
    
    // Initialize the solver
    void initializeSolver();
    
    // Distance threshold to determine if we've reached the final waypoint
    double arrival_threshold_;
};

#endif // MPC_GUIDANCE_HPP 
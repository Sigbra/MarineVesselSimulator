#ifndef GUIDANCE_HPP
#define GUIDANCE_HPP

#include <string>
#include <vector>
#include "utilities.hpp"

class GuidanceMethod {
    public:
        // Constructor
        GuidanceMethod();
        
        // Displays the selection menu and returns the chosen method index
        int selectMethod();
    
    private:
        std::vector<std::string> methods = {
            "Dynamic Positioning (static heading ref)",
            "ALOS straight line path following",
            "MPC guidance"
        };
    };

class DynamicPositioning {
    public:
        // Constructor: initializes with the waypoint list and switching threshold.
        // The active segment is initially set from the first to the second waypoint.
        DynamicPositioning(const Waypoints &waypoints, double switch_radious);
    
        // Update method: given the current position (xn, yn), returns a vector with:
        // [desired_x, desired_y, desired_heading]
        std::tuple<double, double, double> update(double xn, double yn);
    
        // Reset the internal state (e.g., for a new mission).
        void reset();
    
    private:
        Waypoints wpt;
        double R_switch;
        // The active waypoint index; using the convention that the current segment is from index (dp_index-1) to dp_index.
        size_t wpt_index;
    };

class ALOS {
    public:
        // Constructor: initialize the guidance parameters and state using provided waypoints.
        ALOS(const Waypoints &wpt, double Delta_h, double gamma_h, double h, double R_switch);
    
        // Update the guidance law given the current vehicle position.
        // Returns a tuple containing:
        // - psi_ref: desired heading angle,
        // - y_e: cross-track error,
        // - at_last_waypoint: flag indicating if the vehicle is within R_switch of the last waypoint.
        std::tuple<double, double, bool> update(double x, double y);
    
        // Reset the internal state (e.g., for a new mission or simulation run).
        void reset();
    
    private:
        // Guidance parameters.
        Waypoints wpt_;
        double Delta_h_;
        double gamma_h_;
        double h_;
        double R_switch_;
    
        // State variables.
        int k_;           // Active waypoint index.
        double xk_;       // Active waypoint x-coordinate.
        double yk_;       // Active waypoint y-coordinate.
        double beta_hat_; // Estimate of the crab (current) angle.
    };

class LOSObserver {
    public:
        // Constructor
        LOSObserver(double h, double K_f);
    
        // Update function for the LOS observer
        void update(double LOScommand);
    
        // Getters for LOS angle and LOS rate
        double getLOSAngle() const;
        double getLOSRate() const;
    
    private:
        double h;        // Sampling time
        double K_f;      // Observer gain
        double T_f;      // Differentiator time constant
        double LOSangle; // Estimated LOS angle
        double LOSrate;  // Estimated LOS rate
        double xi;       // Internal differentiator state
    };

#endif // GUIDANCE_HPP
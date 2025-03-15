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
            "ALOS straight line path following"
        };
    };

class DynamicPositioning {
    public:
        // Constructor: initializes with the waypoint list and switching threshold.
        DynamicPositioning(const Waypoints &waypoints, double switch_radius);
    
        // Update method: given the current position (xn, yn), returns a vector with:
        // [desired_x, desired_y, desired_heading]
        std::tuple<double, double, double> update(double xn, double yn);
    
        // Update waypoints without recreating the object
        void updateWaypoints(const Waypoints &newWaypoints);
        
        // Alias for updateWaypoints to maintain consistent naming with ALOS
        void updatePath(const Waypoints &newPath) { updateWaypoints(newPath); }
        
        // Reset the internal state (e.g., for a new mission).
        void reset();
    
    private:
        Waypoints wpt;
        double R_switch;
        // The active waypoint index
        size_t wpt_index;
    };

class ALOS {
    public:
        // Constructor with waypoints
        ALOS(const Waypoints &waypoints, double Delta_h, double gamma_h, double h, double R_switch);
    
        // Update method
        std::tuple<double, double, bool> update(double x, double y);
        
        // Update path without recreating the object
        void updatePath(const Waypoints &newPath);
    
        void reset();
    
    private:
        // Guidance parameters.
        double Delta_h_;
        double gamma_h_;
        double h_;
        double R_switch_;
    
        // State variables.
        int k_;           // Current index on the reference path
        double beta_hat_; // Estimate of the crab (current) angle.

        Waypoints refPath_;
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
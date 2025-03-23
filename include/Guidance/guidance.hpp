#ifndef GUIDANCE_HPP
#define GUIDANCE_HPP

#include <string>
#include <vector>
#include "Utilities/utilities.hpp"

class GuidanceMethod {
    public:
        // Constructor
        GuidanceMethod();
        
        // Displays the selection menu and returns the chosen method index
        int selectMethod();
    
    private:
        std::vector<std::string> methods = {
            "Dynamic Positioning",
            "LOS",
            "ALOS"
        };
    };

std::tuple<double, double, double> DP(double path_x, double path_y,
                                      double prev_path_x, double prev_path_y,
                                    double xn, double yn);

std::tuple<double, double> LOS(double xn, double yn, double delta,
                               double path_x, double path_y,
                               double path_x_dot, double path_y_dot);

std::tuple<double, double> ALOS(double xn, double yn, double delta,
                                double path_x, double path_y,
                                double path_x_dot, double path_y_dot);

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
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
            "Station keeping (dynamic heading ref)",
            "Dynamic Positioning (static heading ref)"
        };
    };

std::vector<double> StationKeeping(Waypoints wpt, int wpt_index, double xn, double yn, double psi_d);

std::vector<double> DynamicPositioning(Waypoints wpt, int wpt_index);

#endif // GUIDANCE_HPP
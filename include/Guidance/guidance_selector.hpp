#ifndef GUIDANCESELECTOR_HPP
#define GUIDANCESELECTOR_HPP

#include <string>
#include <vector>

class GuidanceMethod {
    public:
        // Constructor
        GuidanceMethod();
        
        // Displays the selection menu and returns the chosen method index
        int selectMethod();
    
    private:
        std::vector<std::string> methods = {
            "DP:  Simple wpt guidance",
            "DP:  MPC guidance",
            "LOS  guidance",
            "ALOS guidance"
        };
    };

#endif
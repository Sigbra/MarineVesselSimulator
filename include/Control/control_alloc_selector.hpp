#ifndef CONTROLALLOCSELECTOR_HPP
#define CONTROLALLOCSELECTOR_HPP

#include <vector>
#include <string>

class ControlAllocationMethod {
    public:
        // Constructor
        ControlAllocationMethod();
        
        // Displays the selection menu and returns the chosen method index
        int selectMethod();
    
    private:
        std::vector<std::string> methods = {
            "Non-linear optimization",
            "Model Predictive Control (MPC)"
        };
    };

#endif
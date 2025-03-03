#ifndef CONTROLALLOCATION_HPP
#define CONTROLALLOCATION_HPP

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

std::vector<double> NLOptControlAlloc(double Fx_des, double Fy_des, double Mz_des);


#endif
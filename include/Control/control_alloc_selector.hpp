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
            "Pseudo-inverse allocation with azimuthg angle constraints",
            "Non-linear constrained optimization control allocation",
            "MPC control allocation (simmilar to above, but with rate constraints)",
            "MPC control system (DP)"
        };
    };

#endif
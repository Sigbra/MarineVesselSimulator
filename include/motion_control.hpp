#ifndef MOTIONCONTROL_HPP
#define MOTIONCONTROL_HPP

#include "utilities.hpp"
#include <vector>
#include <string>

class ControlMethod {
    public:
        // Constructor
        ControlMethod();
        
        // Displays the selection menu and returns the chosen method index
        int selectMethod();
    
    private:
        std::vector<std::string> methods = {
            "No desired Forces or Moments in Surge, Sway or Yaw",
            "Dynamic Positioning (DP)"
        };
    };

void NoDesiredForcesOrMoments(double &tau_X, double &tau_Y, double &tau_N);

void SISO_linear_PID_Control(double u, double v, double xn, double yn, double psi,
                             const Waypoints &wpt, int &wpt_index, double h,
                             double &tau_X, double &tau_Y, double &tau_N,
                             double &z_xn, double &z_yn, double &z_psi,
                             double &prev_error_xn, double &prev_error_yn, double &prev_error_psi);

#endif 
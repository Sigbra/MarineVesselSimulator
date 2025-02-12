#include <iostream>
#include <vector>
#include <cmath>
#include "utilities.hpp"
#include "ref_model.hpp"
#include "ALOSpsi.hpp"
#include "hermite_spline.hpp"
#include "crosstrack_hermite.hpp"
#include "los_observer.hpp"
#include "control_method.hpp"

int main() {
    // USER INPUTS
    double h = 0.05; // Sampling time [s]
    double T_final = 1000; // Final simulation time [s]
    
    // Ocean current
    double V_c = 0.3; // Ocean current speed (m/s)
    double beta_c = M_PI / 6; // Ocean current direction (rad)
    
    // Waypoints
    Waypoints wpt;
    wpt.x = {0, 0, 150, 150, -100, -100, 200};
    wpt.y = {0, 200, 200, -50, -50, 250, 250};
    
    // ALOS and ILOS parameters
    double Delta_h = 10;
    double gamma_h = 0.001;
    double kappa = 0.001;
    
    // Additional parameter for straight-line path following
    double R_switch = 5;
    double K_f = 0.3;
    
    // Initial heading
    double psi0 = atan2(wpt.y[1] - wpt.y[0], wpt.x[1] - wpt.x[0]);
    
    // Additional parameters for Hermite spline path following
    double Umax = 2;
    int idx_start = 1;
    SplineResult spline = hermiteSpline(wpt, Umax, h);
    
    // Reference model parameters
    double wn_d = 1.0;
    double zeta_d = 1.0;
    double r_max = M_PI / 18;
    
    // Control method selection
    std::vector<std::string> methods = {
        "PID heading autopilot, no path following",
        "ALOS path-following control using straight lines and waypoint switching",
        "ILOS path-following control using straight lines and waypoint switching",
        "ALOS path-following control using Hermite splines"
    };
    ControlMethod control(methods);
    int ControlFlag = control.selectMethod();
    
    // Initialize variables
    double psi_d = psi0, r_d = 0, a_d = 0;
    LOSObserver losObserver(h, K_f);
    
    for (double t = 0; t <= T_final; t += h) {
        // Guidance and control system
        switch (ControlFlag) {
            case 1: {
                double psi_ref = psi0;
                if (t > 100) psi_ref = 0;
                if (t > 500) psi_ref = -M_PI / 2;
                refModel(psi_d, r_d, a_d, psi_ref, r_max, zeta_d, wn_d, h, true);
                break;
            }
            case 2: {
                auto [psi_ref, _] = ALOSpsi(0, 0, Delta_h, gamma_h, h, R_switch, wpt);
                losObserver.update(psi_ref);
                psi_d = losObserver.getLOSAngle();
                r_d = losObserver.getLOSRate();
                break;
            }
            case 3: {
                // Implement ILOS guidance function call here
                break;
            }
            case 4: {
                // Implement Hermite spline path following here
                break;
            }
        }
        std::cout << "Time: " << t << " Psi_d: " << psi_d << " R_d: " << r_d << std::endl;
    }
    return 0;
}
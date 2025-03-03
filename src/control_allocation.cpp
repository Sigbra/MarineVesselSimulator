#include "control_allocation.hpp"
#include <casadi/casadi.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <map>
#include "utilities.hpp"

using namespace casadi;

using namespace casadi;

std::vector<double> NLOptControlAlloc(double tau_X, double tau_Y, double tau_N) {
    // Define decision variables using MX (purely symbolic)
    MX F1   = MX::sym("F1");      // Thruster 1 force
    MX phi1 = MX::sym("phi1");    // Thruster 1 azimuth angle
    MX F2   = MX::sym("F2");      // Thruster 2 force
    MX phi2 = MX::sym("phi2");    // Thruster 2 azimuth angle

    // Combine decision variables into a single column vector
    MX vars = vertcat(std::vector<MX>{F1, phi1, F2, phi2});
    
    // Thruster positions (for yaw moment computation)
    // (Assuming both thrusters are located 0.5 m from the centerline)
    double L1 = 0.5;  
    double L2 = 0.5;

    // Define expressions for the generated forces (all in MX):
    MX Fx = F1*cos(phi1) + F2*cos(phi2);
    MX Fy = F1*sin(phi1) + F2*sin(phi2);
    MX Mz = -0.5*(F1*sin(phi1) + F2*sin(phi2)) - F1*cos(phi1) + F2*cos(phi2);

    // Define equality constraints: computed force/moment minus desired must be zero.
    MX g = vertcat(std::vector<MX>{ Fx - tau_X,
                                    Fy - tau_Y,
                                    Mz - tau_N });
    
    // Define the objective function: minimize F1^2 + F2^2
    MX obj = F1*F1 + F2*F2 + 0.1*(phi1*phi1 + phi2*phi2);
    
    // Set up the NLP using an MXDict (as in the CasADi example)
    MXDict nlp;
    nlp["x"] = vertcat(std::vector<MX>{F1, phi1, F2, phi2});
    nlp["f"] = obj;
    nlp["g"] = g;

    Dict opts;
    opts["ipopt.print_level"] = 0;  // Suppress IPOPT solver output
    opts["print_time"] = false;     // Disable CasADi timing output
    opts["ipopt.sb"] = "yes";       // Suppress IPOPT banner
    opts["ipopt.file_print_level"] = 0;  // Disable output to file
    opts["verbose"] = false;        // Disable verbose mode in CasADi
    opts["ipopt.max_iter"] = 200;
    
    // Create solver instance using IPOPT; note that "ipopt" is cast to std::string.
    Function solver = nlpsol("solver", std::string("ipopt"), nlp, opts);

    // Set up bounds and initial guess using DM objects.
    // Decision variable bounds:
    DM lbx = DM({-20.0, -M_PI/2, -20.0, -M_PI/2});
    DM ubx = DM({100.0,  M_PI/2, 100.0,  M_PI/2});
    // Equality constraints: all set to zero.
    DM lbg = DM({0.0, 0.0, 0.0});
    DM ubg = DM({0.0, 0.0, 0.0});
    
    // Initial guess for decision variables
    DM x0 = DM({0.0, 0.0, 0.0, 0.0});
    
    // Build solver argument dictionary using DMDict
    DMDict solver_args = {
        {"x0", x0},
        {"lbx", lbx},
        {"ubx", ubx},
        {"lbg", lbg},
        {"ubg", ubg}
    };
    
    // Solve the NLP
    DMDict res = solver(solver_args);
    
    // Extract the solution from the result (res["x"] is a DM)
    DM sol = res["x"];
    std::vector<double> sol_vec = sol.nonzeros();
    
    // Return the optimal decision variables in order: {F1, phi1, F2, phi2}
    return sol_vec;
}


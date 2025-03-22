#include <casadi/casadi.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include "control_allocation.hpp"
#include "ran.hpp"
#include "utilities.hpp"

ControlAllocationMethod::ControlAllocationMethod(){}
    
int ControlAllocationMethod::selectMethod() {
    std::cout << "Choose Control Allocation Method:" << std::endl;
    for (size_t i = 0; i < methods.size(); ++i) {
        std::cout << i + 1 << ". " << methods[i] << std::endl;
    }
    
    int choice = 0;
    while (true) {
        std::cout << "Enter the number of your choice: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
        }
        else if (choice > 0 && choice <= static_cast<int>(methods.size())) {
            break;
        }
        else {
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    return choice;
}

std::vector<double> NLOptControlAlloc(double tau_X, double tau_Y, double tau_N, double U) {
    using namespace casadi;

    // Check for NaN inputs and replace with safe values
    if (std::isnan(tau_X)) tau_X = 0.0;
    if (std::isnan(tau_Y)) tau_Y = 0.0;
    if (std::isnan(tau_N)) tau_N = 0.0;
    if (std::isnan(U)) U = 0.1;  // Small positive value

    // Define symbolic decision variables
    MX n1 = MX::sym("F1");        // Thruster 1 force
    MX alpha1 = MX::sym("phi1");  // Thruster 1 angle
    MX n2 = MX::sym("F2");        // Thruster 2 force
    MX alpha2 = MX::sym("phi2");  // Thruster 2 angle

    // Combine decision variables
    MX vars = vertcat(n1, alpha1, n2, alpha2);

    // CO offset
    Eigen::Vector3d CO_offset = CO_Offset(U);
    double ly1 =  1.1 - CO_offset(1);    // Left pod lever arm
    double ly2 = -1.1 + CO_offset(1);    // Right pod lever arm
    double lx  = -1.1 - CO_offset(0);    // Pod locations in x

    // Constants from ran()
    double g = 9.81;
    double k_pos = 200*g;         // Positive Bollard
    double k_neg = 200*g;         // Negative Bollard
    double n_max =  1;            // Relative propellar speed max (representing max positive  revs)
    double n_min = -1;            // Relative propellar speed min (representing max negative revs)
    double alpha_max = M_PI/2; 
    double alpha_min = -M_PI/2;

    // Thrust Calcualtion 
    // - Symbolic version of ThrustsFromRelativeN() in ran.cpp for casadi.
    MX Thrust1 = if_else(n1 >= 0, k_pos * n1 * fabs(n1), k_neg * n1 * fabs(n1));
    MX Thrust2 = if_else(n2 >= 0, k_pos * n2 * fabs(n2), k_neg * n2 * fabs(n2));

    // Mapping to forces and moments (From ran())
    MX tau_X_model = Thrust1 * cos(alpha1) + Thrust2 * cos(alpha2);
    MX tau_Y_model = Thrust1 * sin(alpha1) + Thrust2 * sin(alpha2);
    MX tau_N_model = lx * (Thrust1*sin(alpha1) + Thrust2*sin(alpha2))
                    -(ly1*Thrust1*cos(alpha1) + ly2*Thrust2*cos(alpha2));

    // Objective Function
    // - Mean square error minimization of each tau component
    MX penalty_tau = 0.5  * pow(tau_X - tau_X_model, 2)
                   + 0.5  * pow(tau_Y - tau_Y_model, 2)
                   + 0.8  * pow(tau_N - tau_N_model, 2);
                   
    MX eff_alpha1 = if_else(n1 >= 0, alpha1, alpha1 + M_PI);
    MX eff_alpha2 = if_else(n2 >= 0, alpha2, alpha2 + M_PI);

    // - Penalty for both pods forward, leading to loss of sway control.
    MX a1 = exp( -pow( abs(eff_alpha1), 2 ) / 0.1 ); 
    MX a2 = exp( -pow( abs(eff_alpha2), 2 ) / 0.1 );
    MX penalty_both_zero = 1 * a1 * a2; // a1_max * a2_max = 1  
    
    // - Penalty for pods in complete opposite directions, leading to loss of surge control.
    MX b1 = exp( -pow((M_PI/2 - abs(eff_alpha1)), 2) / 0.1 ); 
    MX b2 = exp( -pow((M_PI/2 - abs(eff_alpha2)), 2) / 0.1 );
    MX penalty_opposite = 10 * b1 * b2; // b1_max * b2_max = 1  

    // - Penalty for pods pointing inwards, cancelling each other out.
    MX c1 = exp( -pow(eff_alpha1 - M_PI/2, 2) / 0.1 );
    MX c2 = exp( -pow(eff_alpha2 + M_PI/2, 2) / 0.1 );
    MX penalty_inward = 10 * c1 * c2; // c1_max * c2_max = 1 

    // Penalty for both pods beeing +90, leading to loss of surge control?
    MX d1 = exp( -pow(eff_alpha1 - M_PI/2, 2) / 0.1 );
    MX d2 = exp( -pow(eff_alpha2 - M_PI/2, 2) / 0.1 );
    MX penalty_both_plus_90 = 10 * d1 * d2;

    // Penalty for both pods beeing -90, leading to loss of surge control?
    MX e1 = exp( -pow(eff_alpha1 + M_PI/2, 2) / 0.1 );
    MX e2 = exp( -pow(eff_alpha2 + M_PI/2, 2) / 0.1 );
    MX penalty_both_minus_90 = 10 * e1 * e2;
    
    MX objective = penalty_tau
                 + penalty_both_zero
                 + penalty_opposite
                 + penalty_inward
                 + penalty_both_plus_90
                 + penalty_both_minus_90;

    // Set up NLP problem dictionary
    MXDict nlp = {{"x", vars}, {"f", objective}}; 
    Dict opts;
    opts["ipopt.print_level"] = 0;  // Suppress IPOPT solver output
    opts["print_time"] = false;     // Disable CasADi timing output
    opts["ipopt.sb"] = "yes";       // Suppress IPOPT banner
    opts["ipopt.file_print_level"] = 0;  // Disable output to file
    opts["verbose"] = false;        // Disable verbose mode in CasADi
    opts["ipopt.max_iter"] = 50;
    Function solver = nlpsol("thr_alloc_solver", "ipopt", nlp, opts);

    // Initial guess
    DM x0 = DM::zeros(4,1);
    x0(0) = 0.5;  //n1
    x0(1) = 0.0;  //alpha1
    x0(2) = 0.5;  //n2
    x0(3) = 0.0;  //alpha2

    DM lbx = DM::zeros(4, 1);
    lbx(0) = n_min;
    lbx(1) = alpha_min;
    lbx(2) = n_min;
    lbx(3) = alpha_min;

    DM ubx = DM::zeros(4, 1);
    ubx(0) = n_max;
    ubx(1) = alpha_max;
    ubx(2) = n_max;
    ubx(3) = alpha_max;

    // Solve the NLP
    std::map<std::string, DM> args;
    args["x0"]  = x0;
    args["lbx"] = lbx;
    args["ubx"] = ubx;

    std::map<std::string, DM> result = solver(args);    

    // Extract solution values
    DM sol = result.at("x");
    double n1_c   = sol(0).scalar();
    double alpha1_c = sol(1).scalar();
    double n2_c   = sol(2).scalar();
    double alpha2_c = sol(3).scalar();

    return {n1_c, alpha1_c, n2_c, alpha2_c};
}

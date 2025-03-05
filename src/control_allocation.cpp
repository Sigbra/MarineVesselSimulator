#include <casadi/casadi.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <map>
#include "control_allocation.hpp"
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

std::vector<double> NLOptControlAlloc(double tau_X, double tau_Y, double tau_N) {
    using namespace casadi;

    // Define symbolic decision variables
    MX n1 = MX::sym("F1");    // Thruster 1 force
    MX alpha1 = MX::sym("phi1"); // Thruster 1 angle
    MX n2 = MX::sym("F2");    // Thruster 2 force
    MX alpha2 = MX::sym("phi2"); // Thruster 2 angle

    // Combine decision variables
    MX vars = vertcat(n1, alpha1, n2, alpha2);

    //Constants from ran()
    double g = 9.81;
    double k_pos = 0.2216 / 2.0;  //Borrowed from otter
    double k_neg = 0.1289 / 2.0;  //Borrowed from otter
    double n_max =  std::sqrt((0.5 * 24.4 * g) / k_pos);  //Borrowed from otter // maximum propeller rev. (rad/s)
    double n_min = -std::sqrt((0.5 * 13.6 * g) / k_neg);  //Borrowed from otter // minimum propeller rev. (rad/s)
    double alpha_max = M_PI/2 - 0.01; 
    double alpha_min = -M_PI/2 + 0.01;
    double ly1 =  1.1;   
    double ly2 = -1.1;  
    double lx  =  1.4; 

    // Thrust Calcualtion
    MX Thrust1 = if_else(n1 >= 0, k_pos * n1 * fabs(n1), k_neg * n1 * fabs(n1));
    MX Thrust2 = if_else(n2 >= 0, k_pos * n2 * fabs(n2), k_neg * n2 * fabs(n2));

    // Mapping to forces and moments (From ran())
    MX tau_X_model = Thrust1 * cos(alpha1) + Thrust2 * cos(alpha2);
    MX tau_Y_model = Thrust1 * sin(alpha1) + Thrust2 * sin(alpha2);
    MX tau_N_model = lx * (Thrust1*sin(alpha1) + Thrust2*sin(alpha2))
                    -(ly1*Thrust1*cos(alpha1) + ly2*Thrust2*cos(alpha2));

    // Objective Function
    // - Mean square error minimization of each tau component
    MX penalty_tau = 0.1 * pow(tau_X - tau_X_model, 2)
                   + 0.1 * pow(tau_Y - tau_Y_model, 2)
                   + 0.5 * pow(tau_N - tau_N_model, 2);
                   
    // - Penalizing equal direction that can lead to loss of surge or sway control
    MX penalty_equal = 10/(abs(alpha1 - alpha2)+0.1);  
    
    // - Penalizing complete opposite directions leading to loss of surge control
    MX eff_alpha1 = if_else(n1 >= 0, alpha1, alpha1 + M_PI);
    MX eff_alpha2 = if_else(n2 >= 0, alpha2, alpha2 + M_PI);
    MX f1 = exp( -pow((M_PI/2 - abs(eff_alpha1)), 2) / 0.1 );
    MX f2 = exp( -pow((M_PI/2 - abs(eff_alpha2)), 2) / 0.1 );
    MX penalty_opposite = 10*f1*f2; //Penalize when both thrusters have bad surge efficiency simultaniously
    
    MX objective = penalty_tau + penalty_equal + penalty_opposite;

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
    x0(0) = 0.0;  //n1
    x0(1) = 0.0;  //alpha1
    x0(2) = 0.0;  //n2
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

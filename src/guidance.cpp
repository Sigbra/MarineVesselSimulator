#include "guidance.hpp"
#include "utilities.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

GuidanceMethod::GuidanceMethod(){}
    
int GuidanceMethod::selectMethod() {
    std::cout << "Choose Guidance Method:" << std::endl;
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

std::vector<double> StationKeeping(Waypoints wpt, int wpt_index, double xn, double yn, double psi_d) {
    double xn_d  = wpt.x[wpt_index];
    double yn_d  = wpt.y[wpt_index];
    double xn_error = xn_d - xn;
    double yn_error = yn_d - yn;

    // Small errors can give unstable psi_d, 
    // hence the previous value is kept as desired.
    if ((abs(xn_error) + abs(yn_error)) < 0.1) { 
        return {xn_d, yn_d, psi_d}; 
    } 

    psi_d = std::atan2(yn_d - yn, xn_d - xn);
    return {xn_d, yn_d, psi_d};
}

// Dynamic positioning
// Steers x, y and psi towards a goal.
std::vector<double> DynamicPositioning(std::vector<double> wpt_start,
                                       std::vector<double> wpt_goal,
                                       std::vector<double> current_position){

    // Current position
    double xn = current_position[0];
    double yn = current_position[1];

    // Start states
    double x_s = wpt_start[0];
    double y_s = wpt_start[1];

    // Goal states  
    double x_g = wpt_goal[0];
    double y_g = wpt_goal[1];
    double psi_g = std::atan2(y_g - y_s, y_g - y_s);

    // Calculating the step size for the desired position. 
    double dx = x_g - xn;
    double dy = y_g - yn;
    double abs_dist_to_goal = std::hypot(dx, dy);     
    double step_size = std::min(1.0, abs_dist_to_goal);

    // Desired states limited by 1m in distance
    double xn_d = xn + step_size * std::cos(psi_g);
    double yn_d = yn + step_size * std::sin(psi_g);
    double psi_d  = psi_g;

    return {xn_d, yn_d, psi_d};
}
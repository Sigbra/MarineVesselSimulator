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
std::vector<double> DynamicPositioning(Waypoints pd_points, int pd_points_index){

    double x_start = pd_points.x[pd_points_index-1];
    double y_start = pd_points.y[pd_points_index-1];

    double x_goal = pd_points.x[pd_points_index];
    double y_goal = pd_points.y[pd_points_index];

    double xn_d = x_goal;
    double yn_d = y_goal;
    double psi_d = std::atan2(y_goal - y_start, x_goal - x_start);

    return {xn_d, yn_d, psi_d};
}
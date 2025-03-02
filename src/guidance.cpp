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

std::vector<double> StationKeeping(Waypoints wpt, int wpt_index, double xn, double yn, double psi) {
    //Desired position in NED
    double xn_d = wpt.x[wpt_index];
    double yn_d = wpt.y[wpt_index];
    double psi_d = std::atan2(yn_d - yn, xn_d - xn);
    
    // Position error in NED
    double error_xn_global = xn_d - xn;
    double error_yn_global = yn_d - yn;

    // Position error in BODY frame
    double cos_psi = cos(psi);
    double sin_psi = sin(psi);
    double x_d_body = cos_psi * error_xn_global + sin_psi * error_yn_global;  
    double y_d_body = -sin_psi * error_xn_global + cos_psi * error_yn_global; 

    // Heading error in BODY frame
    double psi_d_body = ssa(psi_d - psi);

    return {x_d_body, y_d_body, psi_d_body};
}

std::vector<double> DynamicPositioning(Waypoints wpt, int wpt_index, double xn, double yn) {
    double xn_d  = wpt.x[wpt_index];
    double yn_d  = wpt.y[wpt_index];
    double psi_d = wpt.angle[wpt_index];
    return {xn_d, yn_d, psi_d};
}
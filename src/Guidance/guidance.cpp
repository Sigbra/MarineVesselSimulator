#include "Guidance/guidance.hpp"
#include "Utilities/utilities.hpp"
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

//---------------------Dynamic Positioning--------------------------

std::tuple<double, double, double> DP(double xn, double yn,
                                      double path_x, double path_y,
                                      double prev_path_x, double prev_path_y)
{
double xn_d = path_x;
double yn_d = path_y;
double psi_d = std::atan2(path_y - prev_path_y, path_x - prev_path_x);

return std::make_tuple(xn_d, yn_d, psi_d);
}


//------------------------------LOS----------------------------------

std::tuple<double, double> LOS(double xn, double yn, double delta,
                             double path_x, double path_y,
                             double path_x_dot, double path_y_dot)
{
    // Check for invalid path derivatives (nearly zero)
    const double epsilon = 1e-6;
    if (std::abs(path_x_dot) < epsilon && std::abs(path_y_dot) < epsilon) {
        // Fall back to a direct course to path point if derivatives are too small
        double psi_ref = std::atan2(path_y - yn, path_x - xn);
        double y_e = 0.0;  // Not meaningful in this case
        return std::make_tuple(psi_ref, y_e);
    }

    // Compute the tangent angle (psi_p) of the path at the given point.
    double psi_p = std::atan2(path_y_dot, path_x_dot);

    // Compute the error vector from the path point to the vessel.
    double error_x = xn - path_x;
    double error_y = yn - path_y;

    // Compute the cross-track error as the projection of the error vector onto the normal.
    double y_e = -std::sin(psi_p) * error_x + std::cos(psi_p) * error_y;

    // Guidance law using a lookahead distance Delta.
    double psi_ref = psi_p - std::atan(y_e / delta);

    return std::make_tuple(psi_ref, y_e);
}

//--------------------------------ALOS--------------------------------
std::tuple<double, double> ALOS(double xn, double yn, double delta,
                                double path_x, double path_y,
                                double path_x_dot, double path_y_dot)
{
    return std::make_tuple(0.0, 0.0);
}

//----------------------------LOS Observer----------------------------

LOSObserver::LOSObserver(double h, double K_f)
    : h(h), K_f(K_f), LOSangle(0.0), LOSrate(0.0) {
    T_f = 1.0 / (K_f + 2 * std::sqrt(K_f) + 1);
    xi = LOSangle - LOSrate;
}

void LOSObserver::update(double LOScommand) {
    double PHI = std::exp(-h / T_f);
    LOSangle += h * (LOSrate + K_f * ssa(LOScommand - LOSangle));
    xi = PHI * xi + (1 - PHI) * LOSangle;
    LOSrate = LOSangle - xi;
}

double LOSObserver::getLOSAngle() const {
    return LOSangle;
}

double LOSObserver::getLOSRate() const {
    return LOSrate;
}
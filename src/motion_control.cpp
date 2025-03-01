#include "motion_control.hpp"
#include "utilities.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
#include <limits>


ControlMethod::ControlMethod(){}
    
int ControlMethod::selectMethod() {
    std::cout << "Choose Control Method:" << std::endl;
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

void NoDesiredForcesOrMoments(double &tau_X, double &tau_Y, double &tau_N) {
    tau_X = std::numeric_limits<double>::quiet_NaN();
    tau_Y = std::numeric_limits<double>::quiet_NaN();
    tau_N = std::numeric_limits<double>::quiet_NaN();
}

void SISO_linear_PID_Control(double u, double v, double xn, double yn, double psi,
    const Waypoints &wpt, int &wpt_index, double h,
    double &tau_X, double &tau_Y, double &tau_N,
    double &z_xn, double &z_yn, double &z_psi,
    double &prev_error_xn, double &prev_error_yn, double &prev_error_psi) {

    // Retrieve current waypoint and desired heading from the waypoint list
    double xn_d  = wpt.x[wpt_index];
    double yn_d  = wpt.y[wpt_index];
    double psi_d = wpt.angle[wpt_index];  // desired heading (rad)

    // Compute the position error in the global frame
    double error_xn_global = xn_d - xn;
    double error_yn_global = yn_d - yn;

    // Transform the global position error into the vessel's body-fixed frame.
    double cos_psi = cos(psi);
    double sin_psi = sin(psi);
    double error_xn = cos_psi * error_xn_global + sin_psi * error_yn_global;   // Surge error (m)
    double error_yn = -sin_psi * error_xn_global + cos_psi * error_yn_global;  // Sway error (m)

    // Compute heading error 
    double error_psi = ssa(psi_d - psi);

    // --- PID Gains ---
    // Surge and sway gains (N/m, N/(m·s), N·s/m)
    const double Kp_xn = 500.0;
    const double Ki_xn = 20.0;
    const double Kd_xn = 1000.0;
    const double Kp_yn = 500.0;
    const double Ki_yn = 20.0;
    const double Kd_yn = 1000.0;
    
    // Yaw gains (N·m/rad, N·m/(rad·s), N·m·s/rad)
    const double Kp_psi = 200.0;
    const double Ki_psi = 10.0;
    const double Kd_psi = 150.0;

    // --- Compute Derivative Terms ---
    double d_error_xn  = (error_xn - prev_error_xn)   / h;
    double d_error_yn  = (error_yn - prev_error_yn)   / h;
    double d_error_psi = (error_psi - prev_error_psi) / h;

    // --- PID Control Law ---
    tau_X = Kp_xn  * error_xn  + Kd_xn  * d_error_xn  + Ki_xn  * z_xn;   // Surge force (N)
    tau_Y = Kp_yn  * error_yn  + Kd_yn  * d_error_yn  + Ki_yn  * z_yn;   // Sway force (N)
    tau_N = Kp_psi * error_psi + Kd_psi * d_error_psi + Ki_psi * z_psi;  // Yaw moment (N·m)

    // --- Update Persistent Variables ---
    prev_error_xn  = error_xn;
    prev_error_yn  = error_yn;
    prev_error_psi = error_psi;
}

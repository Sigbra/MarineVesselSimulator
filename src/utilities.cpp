#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "matplotlibcpp.h"
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib> // for getenv
#include "utilities.hpp"
#include <filesystem>

namespace plt = matplotlibcpp;

double ssa(double angle)
{
    // Check for extremely large or small values
    if (std::abs(angle) > 360) {
        std::cerr << "Warning: Unusually large angle value detected: " << angle << std::endl;
        // Optionally clamp angle if it's too large
        angle = std::fmod(angle, 2 * M_PI); // Reduce large values by modulo operation
    }
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;
    return angle;
}

double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}

// Function to get the path to the 'Optimal-Constraint-Thruster-Allocation' repository dynamically
std::string getRepositoryPath() {
    const char* home = std::getenv("HOME");  // Get the user's home directory
    if (home) {
        return std::string(home) + "/Optimal-Constraint-Thruster-Allocation/";
    } else {
        std::cerr << "HOME environment variable is not set!" << std::endl;
        exit(1);
    }
}

// Store simulation data to a file
void storeSimulationData(const Eigen::MatrixXd& simdata, std::string filename) {
    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / filename;

    std::ofstream file(filepath);
    if (file.is_open()) {
        // Write each row of simulation data to the file
        for (int i = 0; i < simdata.rows(); ++i) {
            for (int j = 0; j < simdata.cols(); ++j) {
                file << simdata(i, j);
                if (j < simdata.cols() - 1) file << ",";
            }
            file << std::endl;
        }
        file.close();
        std::cout << "Simulation data stored to: " << filepath << std::endl;
    } else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }
}

// This function reads the CSV file and plots xn vs. yn with heading arrows
void plotTrajectory() {
    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "simdata_vessel_states.csv";
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    std::vector<double> xn, yn, psi;
    std::string line;
    
    // Read CSV data: expecting at least 12 columns (xn at column 7, yn at column 8, psi at column 12)
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> values;
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (const std::invalid_argument&) {
                // Skip invalid entries
            }
        }
        
        if (values.size() >= 16) {
            xn.push_back(values[6]);  
            yn.push_back(values[7]);   
            psi.push_back(values[11]); 
        }
    }
    file.close();
    
    // Ensure we have data before plotting
    if (xn.empty() || yn.empty() || psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    // Plot the trajectory
    plt::figure_size(800, 600);
    plt::plot(xn, yn, "b-");
    plt::xlabel("x(t)");
    plt::ylabel("y(t)");
    plt::title("Vessel Trajectory with Heading Angles");
    
    // Prepare data for quiver (vector field plot)
    std::vector<double> u, v;  // dx, dy components of arrows
    double arrowLength = 0.2;  // Scale factor for arrows
    std::vector<double> xq, yq;
    
    for (size_t i = 0; i < xn.size(); i += 400) { 
        xq.push_back(xn[i]);
        yq.push_back(yn[i]);
        u.push_back(arrowLength * cos(psi[i]));
        v.push_back(arrowLength * sin(psi[i]));
    }
    
    // Use quiver-like representation with arrows
    plt::quiver(xq, yq, u, v);
    
    plt::grid(true);
    plt::show();
}

//Plot trajectory with catamaran boat and azimuth thruster information. 
// void plotTrajectory() {
//     // Build file path
//     std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "simdata_vessel_states.csv";
//     if (!std::filesystem::exists(filepath)) {
//         std::cerr << "File does not exist: " << filepath << std::endl;
//         return;
//     }
    
//     std::ifstream file(filepath);
//     if (!file.is_open()) {
//         std::cerr << "Error: Unable to open " << filepath << std::endl;
//         return;
//     }
    
//     // Vectors for vessel states and thruster data
//     std::vector<double> xn, yn, psi;
//     std::vector<double> pod1_speed, pod2_speed, pod1_azimuth, pod2_azimuth;
//     std::string line;
    
//     // Expecting at least 16 columns:
//     //   xn: column 7 (index 6)
//     //   yn: column 8 (index 7)
//     //   psi: column 12 (index 11)
//     //   pod1_speed: column 13 (index 12)
//     //   pod2_speed: column 14 (index 13)
//     //   pod1_azimuth: column 15 (index 14)
//     //   pod2_azimuth: column 16 (index 15)
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::vector<double> values;
//         std::string cell;
//         while (std::getline(ss, cell, ',')) {
//             try {
//                 values.push_back(std::stod(cell));
//             } catch (const std::invalid_argument&) {
//                 // Skip invalid entries
//             }
//         }
//         if (values.size() >= 16) {
//             xn.push_back(values[6]);
//             yn.push_back(values[7]);
//             psi.push_back(values[11]);
//             pod1_speed.push_back(values[12]);
//             pod2_speed.push_back(values[13]);
//             pod1_azimuth.push_back(values[14]);
//             pod2_azimuth.push_back(values[15]);
//         }
//     }
//     file.close();
    
//     if (xn.empty() || yn.empty() || psi.empty()) {
//         std::cerr << "Error: No valid data found in " << filepath << std::endl;
//         return;
//     }
    
//     // Set up the plot
//     plt::figure_size(800, 600);
//     plt::xlabel("x (m)");
//     plt::ylabel("y (m)");
//     plt::title("Catamaran Trajectory with Azimuth Thrusters");
//     plt::grid(true);
    
//     // Plot vessel trajectory (blue line)
//     plt::plot(xn, yn, "b-");
    
//     // Define catamaran geometry in the boat's body frame.
//     // We'll represent each hull as a rectangle.
//     // Parameters (all in meters):
//     double hullLength = 5.0;        // length of each hull (from back to front)
//     double hullWidth  = 0.25;       // vertical dimension of each hull
//     double pontoonSeparation = 1.0; // distance between the centers of the two hulls.
//     // Thus, the left hull will be centered at (anything, +pontoonSeparation/2)
//     // and the right hull at (anything, -pontoonSeparation/2).
    
//     // Thruster parameters:
//     double thrusterScale = 0.2;  // scale factor converting propeller speed to a drawn line length
    
//     // Lambda: transform a point from the boat's body frame (bx, by) to global frame.
//     auto transform = [&](double x0, double y0, double psi_val, double bx, double by) -> std::pair<double, double> {
//          double gx = x0 + cos(psi_val) * bx - sin(psi_val) * by;
//          double gy = y0 + sin(psi_val) * bx + cos(psi_val) * by;
//          return {gx, gy};
//     };
    
//     // For a subset of data points, draw the boat and its thrusters.
//     for (size_t i = 0; i < xn.size(); i += 500) {
//         double x0 = xn[i];
//         double y0 = yn[i];
//         double psi_val = psi[i];
        
//         // Define left hull rectangle in the boat's body frame.
//         // Corners (ordered to form a closed polygon):
//         // Back left, Front left, Front right, Back right, then back to Back left.
//         double leftCenterY = pontoonSeparation / 2.0;
//         std::vector<double> leftHullX_body = { -hullLength, 0, 0, -hullLength, -hullLength };
//         std::vector<double> leftHullY_body = { leftCenterY - hullWidth/2, leftCenterY - hullWidth/2,
//                                                leftCenterY + hullWidth/2, leftCenterY + hullWidth/2,
//                                                leftCenterY - hullWidth/2 };
        
//         // Define right hull rectangle.
//         double rightCenterY = -pontoonSeparation / 2.0;
//         std::vector<double> rightHullX_body = { -hullLength, 0, 0, -hullLength, -hullLength };
//         std::vector<double> rightHullY_body = { rightCenterY - hullWidth/2, rightCenterY - hullWidth/2,
//                                                 rightCenterY + hullWidth/2, rightCenterY + hullWidth/2,
//                                                 rightCenterY - hullWidth/2 };
        
//         // Transform the hulls to global coordinates.
//         std::vector<double> leftHullX, leftHullY;
//         for (size_t j = 0; j < leftHullX_body.size(); j++) {
//             auto pt = transform(x0, y0, psi_val, leftHullX_body[j], leftHullY_body[j]);
//             leftHullX.push_back(pt.first);
//             leftHullY.push_back(pt.second);
//         }
//         std::vector<double> rightHullX, rightHullY;
//         for (size_t j = 0; j < rightHullX_body.size(); j++) {
//             auto pt = transform(x0, y0, psi_val, rightHullX_body[j], rightHullY_body[j]);
//             rightHullX.push_back(pt.first);
//             rightHullY.push_back(pt.second);
//         }
        
//         // Draw the hulls.
//         plt::plot(leftHullX, leftHullY, "k-");
//         plt::plot(rightHullX, rightHullY, "k-");
        
//         // Draw the deck as a line connecting the front midpoints of the hulls.
//         // Front midpoint of left hull: (0, leftCenterY)
//         // Front midpoint of right hull: (0, rightCenterY)
//         auto frontLeft = transform(x0, y0, psi_val, 0, leftCenterY);
//         auto frontRight = transform(x0, y0, psi_val, 0, rightCenterY);
//         std::vector<double> deckX = { frontLeft.first, frontRight.first };
//         std::vector<double> deckY = { frontLeft.second, frontRight.second };
//         plt::plot(deckX, deckY, "k-");
        
//         // Draw the azimuth thrusters as red lines.
//         // They are placed at the back center of each hull.
//         // Left hull back center: (-hullLength, leftCenterY)
//         auto leftHullBack = transform(x0, y0, psi_val, -hullLength, leftCenterY);
//         // Right hull back center: (-hullLength, rightCenterY)
//         auto rightHullBack = transform(x0, y0, psi_val, -hullLength, rightCenterY);
        
//         double thrusterAngleLeft = psi_val + pod1_azimuth[i];
//         double thrusterAngleRight = psi_val + pod2_azimuth[i];
        
//         double leftThrusterEndX = leftHullBack.first + thrusterScale * pod1_speed[i] * cos(thrusterAngleLeft);
//         double leftThrusterEndY = leftHullBack.second + thrusterScale * pod1_speed[i] * sin(thrusterAngleLeft);
//         std::vector<double> thrusterLeftX = { leftHullBack.first, leftThrusterEndX };
//         std::vector<double> thrusterLeftY = { leftHullBack.second, leftThrusterEndY };
//         plt::plot(thrusterLeftX, thrusterLeftY, "r-");
        
//         double rightThrusterEndX = rightHullBack.first + thrusterScale * pod2_speed[i] * cos(thrusterAngleRight);
//         double rightThrusterEndY = rightHullBack.second + thrusterScale * pod2_speed[i] * sin(thrusterAngleRight);
//         std::vector<double> thrusterRightX = { rightHullBack.first, rightThrusterEndX };
//         std::vector<double> thrusterRightY = { rightHullBack.second, rightThrusterEndY };
//         plt::plot(thrusterRightX, thrusterRightY, "r-");
//     }
    
//     plt::show();
// }





// This function reads the CSV file containing state error data and plots error_x, error_y, and error_psi versus time.
void plotStateErrors() {
    // Build the file path for the state errors CSV file
    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "simdata_state_errors.csv";
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }
    
    // Vectors to hold time and state errors
    std::vector<double> time, error_x, error_y, error_psi;
    std::string line;
    
    // Read CSV data: 
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> values;
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (const std::invalid_argument&) {
                // Skip any invalid entries
            }
        }
        
        if (values.size() >= 4) {
            time.push_back(values[0]);
            error_x.push_back(values[1]);
            error_y.push_back(values[2]);
            error_psi.push_back(values[3]);
        }
    }
    file.close();
    
    // Check if data was successfully loaded
    if (time.empty() || error_x.empty() || error_y.empty() || error_psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    // Plot state errors versus time
    plt::figure_size(800, 600);
    plt::named_plot("Error x position", time, error_x, "r-");
    plt::named_plot("Error y position", time, error_y, "g-");
    plt::named_plot("Error psi",        time, error_psi, "b-");
    plt::xlabel("Time (s)");
    plt::ylabel("State Error");
    plt::title("State Errors over Time");
    plt::legend();
    plt::grid(true);
    plt::show();
}

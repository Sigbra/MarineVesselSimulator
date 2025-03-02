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
        
        if (values.size() >= 12) {
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

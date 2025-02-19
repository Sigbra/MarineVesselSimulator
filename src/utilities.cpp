#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include <cstdlib> // for getenv
#include "utilities.hpp"


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


// Function to get the path to the 'Optimal-Constraint-Thruster-Allocation' repository dynamically
std::string getRepositoryPath() {
    const char* home = std::getenv("HOME");  // Get the user's home directory
    if (home) {
        return std::string(home) + "/Optimal-Constraint-Thruster-Allocation/data/simulation_data.csv";
    } else {
        std::cerr << "HOME environment variable is not set!" << std::endl;
        exit(1);
    }
}

// Store simulation data to a file
void storeSimulationData(const Eigen::MatrixXd& simdata) {
    std::string filepath = getRepositoryPath();  // Get the dynamic file path

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

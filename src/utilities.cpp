#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "matplotlibcpp.h"
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib> 
#include "utilities.hpp"
#include <filesystem>

namespace plt = matplotlibcpp;

double ssa(double angle)
{
    if (std::abs(angle) > 360) {
        std::cerr << "Warning: Unusually large angle value detected: " << angle << std::endl;
        angle = std::fmod(angle, 2 * M_PI); 
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

Waypoints addIntermediateWaypoints(const Waypoints& input, double space) {
    Waypoints output;
    
    // Check if there are any waypoints to process.
    if (input.x.empty() || input.y.empty() || input.x.size() != input.y.size())
        return output;
    
    // Always include the first waypoint.
    output.x.push_back(input.x[0]);
    output.y.push_back(input.y[0]);
    
    // Process each pair of consecutive waypoints.
    for (size_t i = 1; i < input.x.size(); ++i) {
        double x1 = input.x[i - 1];
        double y1 = input.y[i - 1];
        double x2 = input.x[i];
        double y2 = input.y[i];
        
        // Compute the Euclidean distance between the two waypoints.
        double dist = std::hypot(x2 - x1, y2 - y1);
        
        // If the distance exceeds 5 meters, add intermediate waypoints.
        if (dist > space) {
            // Determine the number of segments required so that each segment is <= 5 meters.
            int num_segments = static_cast<int>(std::ceil(dist / space));
            // Insert intermediate waypoints along the line.
            for (int seg = 1; seg < num_segments; ++seg) {
                double t = static_cast<double>(seg) / num_segments;
                double new_x = x1 + t * (x2 - x1);
                double new_y = y1 + t * (y2 - y1);
                output.x.push_back(new_x);
                output.y.push_back(new_y);
            }
        }
        
        // Add the original waypoint.
        output.x.push_back(x2);
        output.y.push_back(y2);
    }
    
    return output;
}

std::string getRepositoryPath() {
    const char* home = std::getenv("HOME");  // Get the user's home directory
    if (home) {
        return std::string(home) + "/Optimal-Constraint-Thruster-Allocation/";
    } else {
        std::cerr << "HOME environment variable is not set!" << std::endl;
        exit(1);
    }
}

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


void plotTrajectory() {

    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "simdata.csv";
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
        
        if (values.size() >= 24) {
            xn.push_back(values[7]);  
            yn.push_back(values[8]);   
            psi.push_back(values[12]); 
        }
    }
    file.close();
    
    if (xn.empty() || yn.empty() || psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
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


void plotStateErrors() {

    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "simdata.csv";
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }
    
    std::vector<double> time, error_x, error_y, error_psi;
    std::string line;
     
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

        if (values.size() >= 24) {
            double t = values[0];
            double xn = values[7];
            double yn = values[8];
            double psi = values[12];
            double xn_d = values[13];
            double yn_d = values[14];
            double psi_d = values[15];

            time.push_back(t);
            error_x.push_back(xn_d - xn);
            error_y.push_back(yn_d - yn);
            error_psi.push_back(ssa(psi_d - psi));
        }
        
    }
    file.close();
    
    if (time.empty() || error_x.empty() || error_y.empty() || error_psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

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

void plotAngles() {

    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "simdata.csv";
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }
    
    std::vector<double> time, psi_d, psi, angle1_error, angle2_error;
    std::string line;
    
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

        if (values.size() >= 24) {
            time.push_back(values[0]);
            psi.push_back(values[12]);
            psi_d.push_back(values[15]);
            angle1_error.push_back(values[20]-values[22]);
            angle2_error.push_back(values[21]-values[23]);
        }
        
    }
    file.close();
    
    if (time.empty() || psi_d.empty() || psi.empty() || angle1_error.empty() || angle2_error.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    plt::figure_size(800, 600);
    plt::named_plot("psi", time, psi, "r-");
    plt::named_plot("psi_d", time, psi_d, "g-");
    plt::named_plot("Left pod angle error", time, angle1_error, "b--");
    plt::named_plot("Right pod angle error", time, angle2_error, "m--");
    plt::xlabel("Time (s)");
    plt::ylabel("Angle");
    plt::title("Heading difference and pod angle errors");
    plt::legend();
    plt::grid(true);
    plt::show();
}
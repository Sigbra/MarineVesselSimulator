#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib> 
#include <filesystem>
#include "matplotlibcpp.h"
#include <Python.h>

#include "Utilities/plotting.hpp"
#include "Utilities/calculations.hpp"

namespace plt = matplotlibcpp;

std::string getRepositoryPath() {
    const char* home = std::getenv("HOME");  // Get the user's home directory
    if (home) {
        return std::string(home) + "/MarineVesselSimulator/";
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

void plotPath(const Waypoints& path) {
    // Check if the path is empty.
    if (path.empty()) {
        std::cerr << "Warning: The path is empty. Nothing to plot." << std::endl;
        return;
    }
    
    // Extract x and y coordinates.
    std::vector<double> x, y;
    for (const auto &pt : path) {
        x.push_back(pt.x);
        y.push_back(pt.y);
    }
    
    try {
        plt::figure();
        plt::plot(x, y, "b-");  
        plt::title("Fermat Spiral Path");
        plt::xlabel("X");
        plt::ylabel("Y");
        plt::grid(true);
        plt::axis("equal");    
        plt::show();
    } catch (const std::exception &e) {
        std::cerr << "Exception during plotting: " << e.what() << std::endl;
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

    std::vector<Vector2D> positions;
    std::vector<double> psi;
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
            positions.push_back(Vector2D(values[7], values[8]));
            psi.push_back(values[12]);
        }
    }
    file.close();
    
    if (positions.empty() || psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    std::vector<double> xn, yn;
    for (const auto& pos : positions) {
        xn.push_back(pos.x);
        yn.push_back(pos.y);
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
            double psi = rad2deg(values[12]);
            double xn_d = values[13];
            double yn_d = values[14];
            double psi_d = rad2deg(values[15]);

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
    plt::axis("equal");
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
    
    std::vector<double> time, psi_d, psi;
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
        }
        
    }
    file.close();
    
    if (time.empty() || psi_d.empty() || psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    plt::figure_size(800, 600);
    plt::named_plot("psi", time, psi, "r-");
    plt::named_plot("psi_d", time, psi_d, "g-");
    plt::xlabel("Time (s)");
    plt::ylabel("Angle");
    plt::title("Heading difference");
    plt::legend();
    plt::grid(true);
    plt::show();
}          

RealTimePlotter::RealTimePlotter() {
    plt::figure();   
    plt::xlabel("x(t)");
    plt::ylabel("y(t)");
    plt::title("Vessel Trajectory with Heading Angles");
    plt::axis("equal");
    plt::grid(true);
    plt::xlim(-50, 100);
    plt::ylim(-50, 100);
}

RealTimePlotter::~RealTimePlotter() {
    plt::close();
}

void RealTimePlotter::setSampledPath(const Waypoints& path) {
    m_path_x.clear();
    m_path_y.clear();
    for (const auto& pt : path) {
        m_path_x.push_back(pt.x);
        m_path_y.push_back(pt.y);
    }
    if (!m_path_x.empty() && !m_path_y.empty()) {
        plt::plot(m_path_x, m_path_y, "k-"); 
        plt::draw();
        plt::pause(0.001);
    }
}


void RealTimePlotter::updatePlot(double x, double y, double psi_value, double arrowLength,
    double x_closest, double y_closest) {

m_x.push_back(x);
m_y.push_back(y);
m_psi.push_back(psi_value);

plt::cla();

if (!m_path_x.empty() && !m_path_y.empty()) {
plt::plot(m_path_x, m_path_y, "k-");  
}

plt::plot(m_x, m_y, "b-");

double u_val = arrowLength * cos(psi_value);
double v_val = arrowLength * sin(psi_value);
std::vector<double> arrowX = { x };
std::vector<double> arrowY = { y };
std::vector<double> u_vec  = { u_val };
std::vector<double> v_vec  = { v_val };
plt::quiver(arrowX, arrowY, u_vec, v_vec);

std::vector<double> cpX = { x_closest };
std::vector<double> cpY = { y_closest };
plt::plot(cpX, cpY, "go"); 

std::vector<double> curPosX = { x };
std::vector<double> curPosY = { y };
plt::plot(curPosX, curPosY, "ro");

plt::xlabel("x(t)");
plt::ylabel("y(t)");
plt::title("Vessel Trajectory with Heading Angles");
plt::grid(true);

plt::draw();
plt::pause(0.01);
}

void RealTimePlotter::finalizePlot(const std::string& filename) {
    if (!filename.empty()) {
        plt::save(filename);
    }
    plt::show();
    plt::close();
}

void plot_points(const std::vector<Vector2D>& vessels, const std::vector<Vector2D>& projections) {
    std::vector<double> vessel_x, vessel_y;
    std::vector<double> proj_x, proj_y;

    for (const auto& v : vessels) {
        vessel_x.push_back(v.x);
        vessel_y.push_back(v.y);
    }

    for (const auto& p : projections) {
        proj_x.push_back(p.x);
        proj_y.push_back(p.y);
    }

    // Plot vessel positions
    plt::scatter(vessel_x, vessel_y, 50.0, {{"color", "red"}, {"label", "Vessel"}});

    // Plot projected points onto the spiral
    plt::scatter(proj_x, proj_y, 30.0, {{"color", "blue"}, {"label", "Projection"}});

    // Set labels and legend
    plt::xlabel("X Position");
    plt::ylabel("Y Position");
    plt::legend();

    // Show the plot
    plt::show();
}
#ifndef PLOTTING_UTILITIES_HPP
#define PLOTTING_UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>
#include "Utilities/calculations.hpp"

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata, std::string filename);

void storeWaypointChangeTimes(const std::vector<double>& times, const std::string& filename);

void plotPath(const Waypoints &path);

void plotTrajectory();

void plotStateErrors();

void plotAngles();

void plotPropellerSpeeds();

void plotAlphas();

void plot_points(const std::vector<Vector2D>& vessels, const std::vector<Vector2D>& projections);

void plotClosestPointErrors();

std::vector<double> loadWaypointChangeTimes();

//void addWaypointChangeLines(const std::vector<double>& change_times);

class RealTimePlotter {
    public:
        RealTimePlotter();
        
        ~RealTimePlotter();

        void setSampledPath(const Waypoints& path);

        void draw_vessel(double x, double y, double theta);

        void updatePlot(double x, double y, double psi_value, double arrowLength, 
                        std::vector<double> guidance_x, std::vector<double> guidance_y);

        void finalizePlot(const std::string& filename = "");
    
    private:
        std::vector<double> m_x;
        std::vector<double> m_y;
        std::vector<double> m_psi;

        std::vector<double> m_path_x;
        std::vector<double> m_path_y;
    };

#endif // UTILITIES_HPP

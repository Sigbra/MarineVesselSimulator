#ifndef PLOTTING_UTILITIES_HPP
#define PLOTTING_UTILITIES_HPP

#include <vector>
#include <Eigen/Dense>
#include "Utilities/calculations.hpp"

std::string getRepositoryPath();

void storeSimulationData(const Eigen::MatrixXd& simdata, std::string filename);

void storeWaypointChangeTimes(const std::vector<double>& times, const std::string& filename);

void plotPath(const Waypoints& wpt, const Waypoints &path);

void plotTrajectory(const Waypoints& wpt, const Waypoints& path);

void plotStateErrors();

void plotAngles();

void plotPropellerSpeeds();

void plotAlphas();

void plotTau();

void plot_points(const std::vector<Vector2D>& vessels, const std::vector<Vector2D>& projections);

void plotClosestPointErrors();

std::vector<double> loadWaypointChangeTimes();

<<<<<<< HEAD
=======
static void decimate_in_place(std::vector<double>& t,
                              std::vector<double>& x,
                              std::vector<double>& y,
                              std::vector<double>& z,
                              size_t max_pts);

void plotIMUAccel();

void plotIMUGyro();

void plotStateEstimateErrors();

>>>>>>> 35-pirnn-observer
//void addWaypointChangeLines(const std::vector<double>& change_times);

class RealTimePlotter {
    public:
        RealTimePlotter();
        ~RealTimePlotter();

        void setSampledPath(const Waypoints& path);

        // existing
        void updatePlot(double x, double y, double psi_value, double arrowLength,
                        std::vector<double> guidance_x, std::vector<double> guidance_y);

        // NEW: real + estimated in one plot
        void updatePlot(double x, double y, double psi_value,
                        double x_est, double y_est, double psi_est,
                        double arrowLength,
                        const std::vector<double>& guidance_x,
                        const std::vector<double>& guidance_y);

        void finalizePlot(const std::string& filename = "");

    private:
        // let the drawer accept a style/color
        void draw_vessel(double x, double y, double theta, const std::string& style = "r-");

        // existing buffers
        std::vector<double> m_x, m_y, m_psi;
        std::vector<double> m_path_x, m_path_y;

        // NEW: estimated buffers
        std::vector<double> m_x_est, m_y_est, m_psi_est;
};


#endif // PLOTTING_UTILITIES_HPP

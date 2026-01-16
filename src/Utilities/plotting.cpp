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

void storeWaypointChangeTimes(const std::vector<double>& times, const std::string& filename)
{
    namespace fs = std::filesystem;

    // 1) Build path and ensure the data/ directory exists
    fs::path dir = fs::path(getRepositoryPath()) / "data";
    if (!fs::exists(dir) && !fs::create_directories(dir)) {
    std::cerr << "Error: could not create directory " << dir << "\n";
    return;
    }

    fs::path filepath = dir / filename;

    // 2) Open the file (truncate/overwrite)
    std::ofstream ofs(filepath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
    std::cerr << "Error opening waypoint times file for writing: "
    << filepath << "\n";
    return;
    }

    // 3) Write each time on its own line
    for (double t : times) {
    ofs << t << "\n";
    }
    // 4) Explicit close (also flushes)
    ofs.close();

    std::cout << "Waypoint change times stored to: "
    << filepath << std::endl;
}

void plotPath(const Waypoints& wpt, const Waypoints& path) {
    plt::rcparams({
        {"font.size",       "14"},    // base font size for all text
        //{"font.family",     "sans-serif"},
        {"axes.titlesize",  "16"},    // title size
        {"axes.labelsize",  "14"},    // axis‐label size
        //{"axes.labelweight","bold"},  // axis‐label weight
        {"xtick.labelsize", "12"},    // x‐tick label size
        {"ytick.labelsize", "12"}     // y‐tick label size
    });  

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

    std::vector<double> wx, wy;
    wx.reserve(wpt.size());
    wy.reserve(wpt.size());
    for (auto &wp : wpt) {
        wx.push_back(wp.x);
        wy.push_back(wp.y);
    }
    
    try {
        plt::figure();
        plt::plot(x, y, "b-");  
        plt::plot(wx, wy, "ro");  
        plt::title("Path");
        plt::xlabel("x [m]");
        plt::ylabel("y [m]");
        plt::grid(true);
        plt::axis("equal");    
        plt::show();
    } catch (const std::exception &e) {
        std::cerr << "Exception during plotting: " << e.what() << std::endl;
    }
}


void plotTrajectory(const Waypoints& wpt, const Waypoints& path) {
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

    std::vector<double> wx, wy;
    wx.reserve(wpt.size());
    wy.reserve(wpt.size());
    for (auto &wp : wpt) {
        wx.push_back(wp.x);
        wy.push_back(wp.y);
    }

    std::vector<double> px, py;
    px.reserve(path.size());
    py.reserve(path.size());
    for (auto &p : path) {
        px.push_back(p.x);
        py.push_back(p.y);
    }
    
    plt::figure_size(800, 800);
    plt::plot(px, py, "r-"); 
    plt::plot(wx, wy, "ro");  
    plt::plot(xn, yn, "b-");
    plt::xlabel("x(t) [m]");
    plt::ylabel("y(t) [m]");
    plt::title("Vessel Path with Heading Angles");
    
    // Prepare data for quiver (vector field plot)
    std::vector<double> u, v;  // dx, dy components of arrows
    double arrowLength = 0.5;  // Scale factor for arrows
    std::vector<double> xq, yq;
    
    for (size_t i = 0; i < xn.size(); i += 2000) { 
        xq.push_back(xn[i]);
        yq.push_back(yn[i]);
        u.push_back(arrowLength * sin(psi[i]));
        v.push_back(arrowLength * cos(psi[i]));
    }
    
    // Use quiver-like representation with arrows
    plt::quiver(xq, yq, u, v);
    plt::axis("equal");
    plt::grid(true);
    plt::show();
}

void plotTrajectories(const Waypoints& wpt, const Waypoints& path)
{
    namespace fs = std::filesystem;

    fs::path filepath = fs::path(getRepositoryPath()) / "data" / "simdata.csv";
    if (!fs::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    // Robust row parser that preserves column positions
    auto parseRow = [](const std::string& line) {
        std::vector<double> values;
        values.reserve(64);

        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (...) {
                values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        return values;
    };

    std::vector<double> x_true, y_true, psi_true;
    std::vector<double> x_hat,  y_hat,  psi_hat;

    std::string line;
    while (std::getline(file, line)) {
        auto values = parseRow(line);

        // Need up to index 45 (0-based) for psi_hat
        if (values.size() < 46) {
            continue;
        }

        const double xt   = values[7];
        const double yt   = values[8];
        const double psit = values[12];

        const double xh   = values[40];
        const double yh   = values[41];
        const double psih = values[45];

        if (std::isfinite(xt) && std::isfinite(yt) && std::isfinite(psit) &&
            std::isfinite(xh) && std::isfinite(yh) && std::isfinite(psih))
        {
            x_true.push_back(xt);
            y_true.push_back(yt);
            psi_true.push_back(psit);

            x_hat.push_back(xh);
            y_hat.push_back(yh);
            psi_hat.push_back(psih);
        }
    }
    file.close();

    if (x_true.empty()) {
        std::cerr << "Error: No valid true/estimated data found in " << filepath << std::endl;
        return;
    }

    // Waypoints and planned path
    std::vector<double> wx, wy;
    wx.reserve(wpt.size());
    wy.reserve(wpt.size());
    for (const auto& wp : wpt) { wx.push_back(wp.x); wy.push_back(wp.y); }

    std::vector<double> px, py;
    px.reserve(path.size());
    py.reserve(path.size());
    for (const auto& p : path) { px.push_back(p.x); py.push_back(p.y); }

    // Build quiver data (downsample)
    const size_t stride = 2000;     // adjust to taste
    const double arrowLength = 0.5;

    std::vector<double> xq_t, yq_t, u_t, v_t;
    std::vector<double> xq_h, yq_h, u_h, v_h;

    xq_t.reserve((x_true.size() + stride - 1) / stride);
    yq_t.reserve(xq_t.capacity());
    u_t.reserve(xq_t.capacity());
    v_t.reserve(xq_t.capacity());

    xq_h.reserve(xq_t.capacity());
    yq_h.reserve(xq_t.capacity());
    u_h.reserve(xq_t.capacity());
    v_h.reserve(xq_t.capacity());

    for (size_t i = 0; i < x_true.size(); i += stride) {
        // Your convention: psi=0 -> North, increasing clockwise, ENU with x=East, y=North
        // Heading vector in (East, North) is [sin(psi), cos(psi)].
        xq_t.push_back(x_true[i]);
        yq_t.push_back(y_true[i]);
        u_t.push_back(arrowLength * std::sin(psi_true[i]));
        v_t.push_back(arrowLength * std::cos(psi_true[i]));

        xq_h.push_back(x_hat[i]);
        yq_h.push_back(y_hat[i]);
        u_h.push_back(arrowLength * std::sin(psi_hat[i]));
        v_h.push_back(arrowLength * std::cos(psi_hat[i]));
    }

    plt::figure_size(900, 900);

    // Lines/markers
    plt::named_plot("Planned path", px, py, "r-");
    plt::named_plot("Waypoints", wx, wy, "ro");
    plt::named_plot("True pos", x_true, y_true, "b-");
    plt::named_plot("Estimated pos", x_hat, y_hat, "g--");

    // Quiver arrows (two sets)
    // Note: matplotlib-cpp forwards to matplotlib; styling support for quiver varies by wrapper version.
    // If your wrapper supports a style string, you can pass it; if not, these calls still work.
    plt::quiver(xq_t, yq_t, u_t, v_t);  // true headings
    plt::quiver(xq_h, yq_h, u_h, v_h);  // estimated headings

    plt::xlabel("x(t) [m]");
    plt::ylabel("y(t) [m]");
    plt::title("True vs Estimated Trajectory with Heading Arrows");
    plt::axis("equal");
    plt::grid(true);
    plt::legend();
    plt::show();
}

void plotHeadingComparison()
{
    namespace fs = std::filesystem;

    fs::path filepath = fs::path(getRepositoryPath()) / "data" / "simdata.csv";
    if (!fs::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    auto parseRow = [](const std::string& line) {
        std::vector<double> values;
        values.reserve(64);
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (...) {
                values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        return values;
    };

    // Wrap angle in degrees to [-180, 180)
    auto wrapDeg180 = [](double deg) {
        deg = std::fmod(deg + 180.0, 360.0);
        if (deg < 0.0) deg += 360.0;
        return deg - 180.0;
    };

    std::vector<double> t, psi_true_deg, psi_hat_deg;

    std::string line;
    size_t k = 0;
    while (std::getline(file, line)) {
        auto values = parseRow(line);
        if (values.size() < 46) { ++k; continue; }

        const double psit = values[12];  // true psi (rad)
        const double psih = values[45];  // est  psi (rad)

        const double ti = std::isfinite(values[0]) ? values[0] : static_cast<double>(k);

        if (std::isfinite(psit) && std::isfinite(psih)) {
            const double psit_deg = wrapDeg180(psit * 180.0 / M_PI);
            const double psih_deg = wrapDeg180(psih * 180.0 / M_PI);

            t.push_back(ti);
            psi_true_deg.push_back(psit_deg);
            psi_hat_deg.push_back(psih_deg);
        }

        ++k;
    }
    file.close();

    if (psi_true_deg.empty()) {
        std::cerr << "Error: No valid heading data found in " << filepath << std::endl;
        return;
    }

    plt::figure_size(900, 400);
    plt::named_plot("psi true [deg]", t, psi_true_deg, "b-");
    plt::named_plot("psi estimated [deg]", t, psi_hat_deg, "g--");
    plt::xlabel("time [s] (or sample index)");
    plt::ylabel("heading psi [deg]");
    plt::title("True vs Estimated Heading (wrapped)");
    plt::ylim(-190.0, 190.0);
    plt::grid(true);
    plt::legend();
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
            error_psi.push_back(psi_d - psi);
        }
        
    }
    file.close();
    
    if (time.empty() || error_x.empty() || error_y.empty() || error_psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Load waypoint change times
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        // Create a vertical line at time t
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {-200, 200};  // Very large range to ensure visibility
        
        // Plot a simple black dashed line
        plt::plot(x_line, y_line, "k-");
    }

    plt::named_plot("Error x position [m]", time, error_x, "r-");
    plt::named_plot("Error y position [m]", time, error_y, "g-");
    plt::named_plot("Error $\\psi$ heading [deg]", time, error_psi, "b-");
    
    plt::xlabel("Time [s]");
    plt::ylabel("State Error");
    plt::title("State Errors over Time");
    plt::ylim(-50, 50);
    plt::legend();
    plt::grid(true);
    plt::show();
}

void plotEndVelocities() {
    std::filesystem::path filepath =
        std::filesystem::path(getRepositoryPath()) / "data" / "simdata.csv";

    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    std::vector<double> time, Ve, Vn, Vd;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> values;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (const std::invalid_argument&) {
                // Skip invalid entries (e.g. header text)
            }
        }

        // Need at least up to psi at index 12
        if (values.size() >= 13) {
            const double t     = values[0];
            const double u     = values[1];
            const double v     = values[2];
            const double w     = values[3];
            const double phi   = values[10];
            const double theta = values[11];
            const double psi   = values[12];

            const Mat3 Rnb = RnbFromEuler(phi, theta, psi); // Body -> END (E,N,D)
            const Eigen::Vector3d v_b(u, v, w);
            const Eigen::Vector3d v_end = Rnb * v_b;

            time.push_back(t);
            Ve.push_back(v_end(0));  // East
            Vn.push_back(v_end(1));  // North
            Vd.push_back(v_end(2));  // Down
        }
    }

    file.close();

    if (time.empty() || Ve.empty() || Vn.empty() || Vd.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Load waypoint change times (same as your other plots)
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    // Compute a reasonable y-range for waypoint vertical lines
    double ymin = Ve[0], ymax = Ve[0];
    auto update_minmax = [&](const std::vector<double>& v) {
        for (double x : v) { ymin = std::min(ymin, x); ymax = std::max(ymax, x); }
    };
    update_minmax(Ve);
    update_minmax(Vn);
    update_minmax(Vd);

    const double pad = 0.1 * std::max(1e-6, (ymax - ymin));
    ymin -= pad;
    ymax += pad;

    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {ymin, ymax};
        plt::plot(x_line, y_line, "k-");
    }

    plt::named_plot("$V_e$ [m/s]", time, Ve, "r-");
    plt::named_plot("$V_n$ [m/s]", time, Vn, "g-");
    plt::named_plot("$V_d$ [m/s]", time, Vd, "b-");

    plt::xlabel("Time [s]");
    plt::ylabel("END Velocity [m/s]");
    plt::title("END Velocities over Time");
    plt::ylim(ymin, ymax);
    plt::legend();
    plt::grid(true);
    plt::show();
}

void plotEndVelocitiesVsEstimates() {
    std::filesystem::path filepath =
        std::filesystem::path(getRepositoryPath()) / "data" / "simdata.csv";

    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    std::vector<double> time;

    // "True" END velocities computed from (u,v,w,phi,theta,psi)
    std::vector<double> Ve, Vn, Vd;

    // Estimated END velocities read from columns 73..75
    std::vector<double> Ve_est, Vn_est, Vd_est;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> values;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (const std::invalid_argument&) {
                // Skip non-numeric (e.g. header)
            }
        }

        // Need at least:
        //  - psi at index 12
        //  - v_end_est at indices 73..75
        if (values.size() >= 76) {
            const double t     = values[0];

            // From x stored in columns 1..12:
            const double u     = values[1];
            const double v     = values[2];
            const double w     = values[3];
            const double phi   = values[10];
            const double theta = values[11];
            const double psi   = values[12];

            const Mat3 Rnb = RnbFromEuler(phi, theta, psi); // Body -> END (E,N,D)
            const Eigen::Vector3d v_b(u, v, w);
            const Eigen::Vector3d v_end = Rnb * v_b;

            // Estimated END velocity logged at columns 73..75
            const double ve_hat = values[73];
            const double vn_hat = values[74];
            const double vd_hat = values[75];

            time.push_back(t);

            Ve.push_back(v_end(0));
            Vn.push_back(v_end(1));
            Vd.push_back(v_end(2));

            Ve_est.push_back(ve_hat);
            Vn_est.push_back(vn_hat);
            Vd_est.push_back(vd_hat);
        }
    }
    file.close();

    if (time.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Waypoint change times (optional vertical lines like your other plot)
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    // Compute y-limits from both true and estimated series
    auto minmax_of = [](const std::vector<double>& v, double& mn, double& mx) {
        for (double x : v) { mn = std::min(mn, x); mx = std::max(mx, x); }
    };

    double ymin = Ve[0], ymax = Ve[0];
    minmax_of(Ve, ymin, ymax);     minmax_of(Vn, ymin, ymax);     minmax_of(Vd, ymin, ymax);
    minmax_of(Ve_est, ymin, ymax); minmax_of(Vn_est, ymin, ymax); minmax_of(Vd_est, ymin, ymax);

    const double pad = 0.1 * std::max(1e-6, (ymax - ymin));
    ymin -= pad; ymax += pad;

    plt::figure_size(2480, 620);

    // Vertical waypoint lines
    for (double tline : wpt_change_times) {
        std::vector<double> x_line = {tline, tline};
        std::vector<double> y_line = {ymin, ymax};
        plt::plot(x_line, y_line, "k-");
    }

    // True vs estimated (estimated dashed)
    plt::named_plot("$V_e$ [m/s]",      time, Ve,     "r-");
    plt::named_plot("$\\hat V_e$ [m/s]", time, Ve_est, "r--");

    plt::named_plot("$V_n$ [m/s]",      time, Vn,     "g-");
    plt::named_plot("$\\hat V_n$ [m/s]", time, Vn_est, "g--");

    plt::named_plot("$V_d$ [m/s]",      time, Vd,     "b-");
    plt::named_plot("$\\hat V_d$ [m/s]", time, Vd_est, "b--");

    plt::xlabel("Time [s]");
    plt::ylabel("END Velocity [m/s]");
    plt::title("END Velocities: True vs Estimated");
    plt::ylim(ymin, ymax);
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
            psi.push_back(rad2deg(values[12]));
            psi_d.push_back(rad2deg(values[15]));
        }
        
    }
    file.close();
    
    if (time.empty() || psi_d.empty() || psi.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    // Load waypoint change times
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();
    
    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        // Create a vertical line at time t
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {-200, 200};  // Very large range to ensure visibility
        
        // Plot a simple black dashed line
        plt::plot(x_line, y_line, "k-");
    }

    plt::named_plot("$\\psi$", time, psi, "r-");
    plt::named_plot("$\\psi_{\\mathrm{desired}}$", time, psi_d, "g-"); 
    
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [deg]");
    plt::title("$\\psi$ vs $\\psi_{\\mathrm{desired}}$");
    plt::ylim(0, 180);
    plt::legend();
    plt::grid(true);
    plt::show();
}      

void plotPropellerSpeeds() {
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

    std::vector<double> time, n1, n2, nc1, nc2;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> vals;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { vals.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip non-numeric */ }
        }
        if (vals.size() >= 20) {
            time.push_back(vals[0]);
            nc1 .push_back(vals[16]);  // n_c(0)
            nc2 .push_back(vals[17]);  // n_c(1)
            n1  .push_back(vals[18]);  // n(0)
            n2  .push_back(vals[19]);  // n(1)
        }
    }
    file.close();

    if (time.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Load waypoint change times
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        // Create a vertical line at time t
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {-200, 200};  // Very large range to ensure visibility
        
        // Plot a simple black dashed line
        plt::plot(x_line, y_line, "k-");
    }

    plt::named_plot("$n_1$ commanded", time, nc1, "C0-");
    plt::named_plot("$n_2$ commanded", time, nc2, "C2-");
    plt::named_plot("$n_1$ actual",    time, n1,  "C3-");
    plt::named_plot("$n_2$ actual",    time, n2,  "C1-");
    
    plt::xlabel("Time [s]");
    plt::ylabel("Relative propeller speed $n$");
    plt::title("Actual vs. Commanded Propeller Speeds");
    plt::ylim(-1.1, 1.1);
    plt::legend();
    plt::grid(true);
    plt::show();
}

void plotAlphas() {
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

    std::vector<double> time, a1, a2, ac1, ac2;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> vals;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { vals.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip non-numeric */ }
        }
        if (vals.size() >= 24) {
            time.push_back(vals[0]);
            ac1.push_back(rad2deg(vals[20]));  // alpha_c(0) → degrees
            ac2.push_back(rad2deg(vals[21]));  // alpha_c(1)
            a1 .push_back(rad2deg(vals[22]));  // alpha(0)
            a2 .push_back(rad2deg(vals[23]));  // alpha(1)
        }
    }
    file.close();

    if (time.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Load waypoint change times
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        // Create a vertical line at time t
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {-200, 200};  // Very large range to ensure visibility
        
        // Plot a simple black dashed line
        plt::plot(x_line, y_line, "k-");
    }

    plt::named_plot("$\\alpha_1$ commanded", time, ac1, "C0-");
    plt::named_plot("$\\alpha_2$ commanded", time, ac2, "C2-");
    plt::named_plot("$\\alpha_1$ actual",    time, a1,  "C3-");
    plt::named_plot("$\\alpha_2$ actual",    time, a2,  "C1-");
    
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [deg]");
    plt::title("Actual vs. Commanded $\\alpha$ Angles");
    plt::ylim(-100, 100);
    plt::legend();
    plt::grid(true);
    plt::show();
}

void plotTau() {
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

    std::vector<double> time, tauX, tauY, tauN;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> vals;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { vals.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip non-numeric */ }
        }
        if (vals.size() >= 30) {
            time.push_back(vals[0]);
            // actual τ_X, τ_Y, τ_N now at 27..29
            tauX.push_back(vals[27]);
            tauY.push_back(vals[28]);
            tauN.push_back(vals[29]);
        }
    }
    file.close();

    if (time.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Load waypoint change times
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        // Create a vertical line at time t
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {-200, 200};  // Very large range to ensure visibility
        
        // Plot a simple black dashed line
        plt::plot(x_line, y_line, "k-");
    }

    plt::named_plot("$\\tau_N$", time, tauN, "C1-");
    plt::named_plot("$\\tau_Y$", time, tauY, "C2-");
    plt::named_plot("$\\tau_X$", time, tauX, "C0-");
    
    plt::xlabel("Time [s]");
    plt::ylabel("Tau");
    plt::title("$\\tau$ over Time");
    plt::ylim(-400, 400);
    plt::legend();
    plt::grid(true);
    plt::show();
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
    plt::title("Projections onto Spiral Path");
    plt::xlabel("x Position [m]");
    plt::ylabel("y Position [m]");
    plt::legend();

    // Show the plot
    plt::show();
}

void plotClosestPointErrors() {
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
    
    std::vector<double> time, x_e, y_e, position_e;
    std::string line;
     
    double total_x_e = 0.0;
    double total_y_e = 0.0;
    double prev_time = 0.0;

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

        if (values.size() >= 34) {
            double t = values[0];
            double xe = values[32];  // closest.x_e
            double ye = values[33];  // closest.y_e

            double h = (t - prev_time);
            
            time.push_back(t);
            x_e.push_back(xe);
            y_e.push_back(ye);
            position_e.push_back(std::sqrt(xe * xe + ye * ye));

            total_x_e += std::abs(xe)*h;
            total_y_e += std::abs(ye)*h;

            prev_time = t;
        }
    }
    file.close();
    
    if (time.empty() || x_e.empty() || y_e.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }
    
    // Load waypoint change times
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    plt::figure_size(2480, 620);

    // Add waypoint change lines
    for (double t : wpt_change_times) {
        // Create a vertical line at time t
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {-200, 200};  // Very large range to ensure visibility
        
        // Plot a simple black dashed line
        plt::plot(x_line, y_line, "k-");
    }

    if (!time.empty()) {
    double t_start = time.front();
    double t_end   = time.back();
    plt::plot(
        std::vector<double>{t_start, t_end},
        std::vector<double>{0.0, 0.0},
        "k-"
    );
    }

    plt::named_plot("Cross-track error", time, y_e, "r-"); //g- <- changed from green for better visibility
    
    plt::xlabel("Time [s]");
    plt::ylabel("Error [m]");
    plt::title("Cross-track error over time");
    plt::legend();
    plt::grid(true);
    plt::show();
}

static void decimate_in_place(std::vector<double>& t,
                              std::vector<double>& x,
                              std::vector<double>& y,
                              std::vector<double>& z,
                              size_t max_pts = 20000)
{
    if (t.size() <= max_pts) return;
    const size_t n = t.size();
    const size_t stride = (n + max_pts - 1) / max_pts;

    auto decimate = [&](std::vector<double>& v) {
        std::vector<double> out; out.reserve((n + stride - 1) / stride);
        for (size_t i = 0; i < n; i += stride) out.push_back(v[i]);
        v.swap(out);
    };
    decimate(t); decimate(x); decimate(y); decimate(z);
}

void plotIMUAccel()
{
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

    std::vector<double> t, ax, ay, az;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v; v.reserve(49);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { v.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip */ }
        }
        if (v.size() >= 49) {                 // we need cols 0 and 46..48
            t.push_back(v[0]);
            ax.push_back(v[46]);
            ay.push_back(v[47]);
            az.push_back(v[48]);
        }
    }
    file.close();

    if (t.empty()) {
        std::cerr << "Error: No valid accel data found in " << filepath << std::endl;
        return;
    }

    // Keep plots responsive for long runs
    decimate_in_place(t, ax, ay, az, 20000);

    // Optional: waypoint change markers
    std::vector<double> wpt_change_times;
    try { wpt_change_times = loadWaypointChangeTimes(); } catch (...) {}

    plt::figure_size(2400, 600);

    // Draw waypoint lines (light gray)
    for (double tt : wpt_change_times) {
        std::vector<double> xl = {tt, tt};
        std::vector<double> yl = {-1000, 1000};
        plt::plot(xl, yl, {{"color","0.7"},{"linestyle","--"},{"linewidth","1"}});
    }

    plt::named_plot("ax [m/s^2]", t, ax, "r-");
    plt::named_plot("ay [m/s^2]", t, ay, "g-");
    plt::named_plot("az [m/s^2]", t, az, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Specific force [m/s^2]");
    plt::title("IMU Accelerometer");
    plt::grid(true);
    plt::legend();
    plt::show();
}

void plotIMUGyro()
{
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

    std::vector<double> t, wx, wy, wz;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v; v.reserve(52);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { v.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip */ }
        }
        if (v.size() >= 52) {                 // we need cols 0 and 49..51
            t.push_back(v[0]);
            wx.push_back(v[49]);
            wy.push_back(v[50]);
            wz.push_back(v[51]);
        }
    }
    file.close();

    if (t.empty()) {
        std::cerr << "Error: No valid gyro data found in " << filepath << std::endl;
        return;
    }

    decimate_in_place(t, wx, wy, wz, 20000);

    // Optional: waypoint change markers
    std::vector<double> wpt_change_times;
    try { wpt_change_times = loadWaypointChangeTimes(); } catch (...) {}

    plt::figure_size(2400, 600);

    for (double tt : wpt_change_times) {
        std::vector<double> xl = {tt, tt};
        std::vector<double> yl = {-1000, 1000};
        plt::plot(xl, yl, {{"color","0.7"},{"linestyle","--"},{"linewidth","1"}});
    }

    plt::named_plot("wx [rad/s]", t, wx, "r-");
    plt::named_plot("wy [rad/s]", t, wy, "g-");
    plt::named_plot("wz [rad/s]", t, wz, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Angular rate [rad/s]");
    plt::title("IMU Gyroscope");
    plt::grid(true);
    plt::legend();
    plt::show();
}

void plotQuaternionQnb()
{
    using std::size_t;

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

    std::vector<double> t, qw, qx, qy, qz;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v; v.reserve(60);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { v.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip non-numerics */ }
        }
        // need cols 0 and 52..55
        if (v.size() >= 56) {
            t .push_back(v[0]);
            qw.push_back(v[52]);
            qx.push_back(v[53]);
            qy.push_back(v[54]);
            qz.push_back(v[55]);
        }
    }
    file.close();

    if (t.empty()) {
        std::cerr << "Error: No valid quaternion data found in " << filepath << std::endl;
        return;
    }

    // --- Local decimator: keeps at most max_pts samples (t + 4 series) ---
    auto decimate_t4 = [](std::vector<double>& tt,
                          std::vector<double>& s1,
                          std::vector<double>& s2,
                          std::vector<double>& s3,
                          std::vector<double>& s4,
                          size_t max_pts = 20000)
    {
        const size_t N = tt.size();
        if (N == 0 || N <= max_pts) return;

        const double step = static_cast<double>(N) / static_cast<double>(max_pts);
        std::vector<size_t> keep;
        keep.reserve(max_pts);
        for (size_t k = 0; k < max_pts; ++k) {
            size_t idx = static_cast<size_t>(std::floor(k * step));
            if (idx >= N) idx = N - 1;
            if (!keep.empty() && idx == keep.back()) continue; // avoid duplicates
            keep.push_back(idx);
        }

        auto apply = [&](std::vector<double>& v) {
            std::vector<double> out; out.reserve(keep.size());
            for (size_t idx : keep) out.push_back(v[idx]);
            v.swap(out);
        };

        apply(tt);
        apply(s1);
        apply(s2);
        apply(s3);
        apply(s4);
    };

    // Keep plots responsive for long runs
    decimate_t4(t, qw, qx, qy, qz, 20000);

    // Optional: waypoint change markers (safe if function is absent/throws)
    std::vector<double> wpt_change_times;
    try { wpt_change_times = loadWaypointChangeTimes(); } catch (...) {}

    plt::figure_size(2400, 600);

    // Draw waypoint lines (light gray)
    for (double tt : wpt_change_times) {
        std::vector<double> xl = {tt, tt};
        std::vector<double> yl = {-1.2, 1.2};
        plt::plot(xl, yl, {{"color","0.7"},{"linestyle","--"},{"linewidth","1"}});
    }

    plt::named_plot("qw", t, qw, "k-");  // black
    plt::named_plot("qx", t, qx, "r-");
    plt::named_plot("qy", t, qy, "g-");
    plt::named_plot("qz", t, qz, "b-");

    plt::xlabel("Time [s]");
    plt::ylabel("Quaternion components (q_nb)");
    plt::title("Attitude Quaternion Components over Time");
    plt::grid(true);
    plt::legend();
    plt::show();
}


void plotStateEstimateErrors() {
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

    auto clamp = [](double x, double lo, double hi){ return std::max(lo, std::min(hi, x)); };
    auto rad2deg = [](double r){ return r * 180.0 / M_PI; };

    // Custom END rotation (ZYX) — matches your project’s convention
    auto RnbFromEuler_END = [](double phi, double theta, double psi){
        const double cphi = std::cos(phi), sphi = std::sin(phi);
        const double cth  = std::cos(theta), sth = std::sin(theta);
        const double cpsi = std::cos(psi),   spsi = std::sin(psi);
        Eigen::Matrix3d R;
        // Row 1: East
        R(0,0) =  spsi * cth;
        R(0,1) =  cpsi * cphi + sphi * sth * spsi;
        R(0,2) = -cpsi * sphi + sth * spsi * cphi;
        // Row 2: North
        R(1,0) =  cpsi * cth;
        R(1,1) = -spsi * cphi + cpsi * sth * sphi;
        R(1,2) =  spsi * sphi + cpsi * cphi * sth;
        // Row 3: Down
        R(2,0) = -sth;
        R(2,1) =  cth * sphi;
        R(2,2) =  cth * cphi;
        return R;
    };

    std::vector<double> time, err_pos_total, err_vel_total, err_ori_deg, err_omega_degps;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> values;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try { values.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip header/invalid */ }
        }

        // Need up to estimated psi at index 45
        if (values.size() >= 46) {
            // SIMDATA (0-based):
            // 0: t
            // 1..12:  x = [u v w p q r x y z phi theta psi]
            // 34..45: x_est = [u v w p q r x y z phi theta psi]

            const double t = values[0];

            // True states
            const double u  = values[1];
            const double v  = values[2];
            const double w  = values[3];
            const double p  = values[4];
            const double q  = values[5];
            const double r  = values[6];
            const double xn = values[7];
            const double yn = values[8];
            const double zn = values[9];
            const double phi    = values[10];
            const double theta  = values[11];
            const double psi    = values[12];

            // Estimated states
            const double u_est  = values[34];
            const double v_est  = values[35];
            const double w_est  = values[36];
            const double p_est  = values[37];
            const double q_est  = values[38];
            const double r_est  = values[39];
            const double x_est  = values[40];
            const double y_est  = values[41];
            const double z_est  = values[42];
            const double phi_est   = values[43];
            const double theta_est = values[44];
            const double psi_est   = values[45];

            // Total position error [m]
            const double dx = x_est - xn;
            const double dy = y_est - yn;
            const double dz = z_est - zn;
            const double e_pos = std::sqrt(dx*dx + dy*dy + dz*dz);

            // Total velocity error [m/s]
            const double du = u_est - u;
            const double dv = v_est - v;
            const double dw = w_est - w;
            const double e_vel = std::sqrt(du*du + dv*dv + dw*dw);

            // Orientation error (SO(3) geodesic angle) [deg]
            const Eigen::Matrix3d R_true = RnbFromEuler_END(phi, theta, psi);
            const Eigen::Matrix3d R_est  = RnbFromEuler_END(phi_est, theta_est, psi_est);
            const Eigen::Matrix3d R_err  = R_est * R_true.transpose();
            const double cosang = clamp((R_err.trace() - 1.0) * 0.5, -1.0, 1.0);
            const double ang_deg = rad2deg(std::acos(cosang));

            // Orientation rate error: ||omega_est - omega|| in deg/s
            const double dp = p_est - p;
            const double dq = q_est - q;
            const double dr = r_est - r;
            const double domega = std::sqrt(dp*dp + dq*dq + dr*dr); // rad/s
            const double domega_degps = rad2deg(domega);            // deg/s

            time.push_back(t);
            err_pos_total.push_back(e_pos);
            err_vel_total.push_back(e_vel);
            err_ori_deg.push_back(ang_deg);
            err_omega_degps.push_back(domega_degps);
        }
    }
    file.close();

    if (time.empty()) {
        std::cerr << "Error: No valid data found in " << filepath << std::endl;
        return;
    }

    // Optional: waypoint change markers
    std::vector<double> wpt_change_times = loadWaypointChangeTimes();

    plt::figure_size(2480, 620);

    // Vertical lines for waypoints (span a generous range)
    double ymax = 0.0;
    if (!err_pos_total.empty())      ymax = std::max(ymax, *std::max_element(err_pos_total.begin(), err_pos_total.end()));
    if (!err_vel_total.empty())      ymax = std::max(ymax, *std::max_element(err_vel_total.begin(), err_vel_total.end()));
    if (!err_ori_deg.empty())        ymax = std::max(ymax, *std::max_element(err_ori_deg.begin(), err_ori_deg.end()));
    if (!err_omega_degps.empty())    ymax = std::max(ymax, *std::max_element(err_omega_degps.begin(), err_omega_degps.end()));
    double line_top = std::max(1.0, ymax) * 1.1 + 5.0;

    for (double t : wpt_change_times) {
        std::vector<double> x_line = {t, t};
        std::vector<double> y_line = {0.0, line_top};
        plt::plot(x_line, y_line, "k-");
    }

    // Four series on one set of axes (mixed units; label clearly)
    plt::named_plot("Orientation rate error ||Δω|| [deg/s]",  time, err_omega_degps, "g-");
    plt::named_plot("Orientation error [deg]",    time, err_ori_deg,     "b-");
    plt::named_plot("||p_est - p|| [m]",          time, err_pos_total,   "m-");
    plt::named_plot("||v_est - v|| [m/s]",        time, err_vel_total,   "c-");

    plt::xlabel("Time [s]");
    plt::ylabel("Errors");
    plt::title("State Estimate Errors");
    plt::grid(true);
    plt::legend();
    plt::show();
}


void plotIMUAccelBiasCompare()
{
    namespace fs = std::filesystem;
    fs::path filepath = fs::path(getRepositoryPath()) / "data" / "simdata.csv";
    if (!fs::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    // Column indices (0-based) in simdata.csv
    constexpr int T_COL = 0;

    // Estimated biases (you store these):
    constexpr int BA_EST_X = 59, BA_EST_Y = 60, BA_EST_Z = 61;

    // True biases (only if you actually store them; otherwise set these to -1 and we’ll just plot estimates)
    constexpr int BA_TRUE_X = 65, BA_TRUE_Y = 66, BA_TRUE_Z = 67;

    std::vector<double> t;
    std::vector<double> bax_true, bay_true, baz_true;
    std::vector<double> bax_est,  bay_est,  baz_est;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v;
        v.reserve(80);

        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { v.push_back(std::stod(cell)); }
            catch (...) { /* skip */ }
        }

        const int need_cols = std::max({T_COL, BA_EST_Z, BA_TRUE_Z}) + 1;
        if ((int)v.size() < need_cols) continue;

        const double tt = v[T_COL];

        const double ex = v[BA_EST_X], ey = v[BA_EST_Y], ez = v[BA_EST_Z];

        // If true cols exist, use them; else set to NaN and skip plotting them later
        const bool have_true = (BA_TRUE_X >= 0 && BA_TRUE_Y >= 0 && BA_TRUE_Z >= 0);
        const double tx = have_true ? v[BA_TRUE_X] : std::numeric_limits<double>::quiet_NaN();
        const double ty = have_true ? v[BA_TRUE_Y] : std::numeric_limits<double>::quiet_NaN();
        const double tz = have_true ? v[BA_TRUE_Z] : std::numeric_limits<double>::quiet_NaN();

        // Skip non-finite rows (matplotlib-cpp sometimes behaves poorly with NaNs/Infs in some builds)
        auto finite = [](double a){ return std::isfinite(a); };
        if (!finite(tt) || !finite(ex) || !finite(ey) || !finite(ez)) continue;
        if (have_true && (!finite(tx) || !finite(ty) || !finite(tz))) continue;

        t.push_back(tt);
        bax_est.push_back(ex); bay_est.push_back(ey); baz_est.push_back(ez);

        if (have_true) {
            bax_true.push_back(tx); bay_true.push_back(ty); baz_true.push_back(tz);
        }
    }
    file.close();

    if (t.empty()) {
        std::cerr << "Error: No valid accel bias data found in " << filepath << std::endl;
        return;
    }

    // ---- simple downsample to avoid crashes ----
    auto downsample7 = [](size_t max_pts,
                          std::vector<double>& t,
                          std::vector<double>& a1, std::vector<double>& a2, std::vector<double>& a3,
                          std::vector<double>& b1, std::vector<double>& b2, std::vector<double>& b3)
    {
        const size_t N = t.size();
        if (N <= max_pts) return;
        const size_t step = (N + max_pts - 1) / max_pts;

        std::vector<double> t2;  t2.reserve((N + step - 1) / step);
        std::vector<double> a12; a12.reserve(t2.capacity());
        std::vector<double> a22; a22.reserve(t2.capacity());
        std::vector<double> a32; a32.reserve(t2.capacity());
        std::vector<double> b12; b12.reserve(t2.capacity());
        std::vector<double> b22; b22.reserve(t2.capacity());
        std::vector<double> b32; b32.reserve(t2.capacity());

        for (size_t i = 0; i < N; i += step) {
            t2.push_back(t[i]);
            a12.push_back(a1[i]); a22.push_back(a2[i]); a32.push_back(a3[i]);
            b12.push_back(b1[i]); b22.push_back(b2[i]); b32.push_back(b3[i]);
        }

        t.swap(t2);
        a1.swap(a12); a2.swap(a22); a3.swap(a32);
        b1.swap(b12); b2.swap(b22); b3.swap(b32);
    };

    const bool have_true = (bax_true.size() == t.size());
    if (have_true) {
        downsample7(20000, t, bax_true, bay_true, baz_true, bax_est, bay_est, baz_est);
    } else {
        // if true not available, downsample only estimated vectors
        auto downsample4 = [](size_t max_pts,
                              std::vector<double>& t,
                              std::vector<double>& a1, std::vector<double>& a2, std::vector<double>& a3)
        {
            const size_t N = t.size();
            if (N <= max_pts) return;
            const size_t step = (N + max_pts - 1) / max_pts;

            std::vector<double> t2;  t2.reserve((N + step - 1) / step);
            std::vector<double> a12; a12.reserve(t2.capacity());
            std::vector<double> a22; a22.reserve(t2.capacity());
            std::vector<double> a32; a32.reserve(t2.capacity());

            for (size_t i = 0; i < N; i += step) {
                t2.push_back(t[i]);
                a12.push_back(a1[i]); a22.push_back(a2[i]); a32.push_back(a3[i]);
            }

            t.swap(t2);
            a1.swap(a12); a2.swap(a22); a3.swap(a32);
        };
        downsample4(20000, t, bax_est, bay_est, baz_est);
    }

    // ---- Plot: simplest possible (3 figures, 2 lines each) ----
    plt::figure_size(1600, 500);
    if (have_true) {
        plt::named_plot("ba_x true", t, bax_true, "-");
        plt::named_plot("ba_x est",  t, bax_est,  "C1-");
    } else {
        plt::named_plot("ba_x est",  t, bax_est,  "C1-");
    }
    plt::xlabel("Time [s]");
    plt::ylabel("ba_x [m/s^2]");
    plt::title("Accel bias X");
    plt::grid(true);
    plt::legend();
    plt::show();

    plt::figure_size(1600, 500);
    if (have_true) {
        plt::named_plot("ba_y true", t, bay_true, "-");
        plt::named_plot("ba_y est",  t, bay_est,  "C1-");
    } else {
        plt::named_plot("ba_y est",  t, bay_est,  "C1-");
    }
    plt::xlabel("Time [s]");
    plt::ylabel("ba_y [m/s^2]");
    plt::title("Accel bias Y");
    plt::grid(true);
    plt::legend();
    plt::show();

    plt::figure_size(1600, 500);
    if (have_true) {
        plt::named_plot("ba_z true", t, baz_true, "-");
        plt::named_plot("ba_z est",  t, baz_est,  "C1-");
    } else {
        plt::named_plot("ba_z est",  t, baz_est,  "C1-");
    }
    plt::xlabel("Time [s]");
    plt::ylabel("ba_z [m/s^2]");
    plt::title("Accel bias Z");
    plt::grid(true);
    plt::legend();
    plt::show();
}


void plotIMUGyroBiasCompare()
{
    namespace fs = std::filesystem;
    fs::path filepath = fs::path(getRepositoryPath()) / "data" / "simdata.csv";
    if (!fs::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open " << filepath << std::endl;
        return;
    }

    constexpr int T_COL = 0;

    // Estimated biases (you store these):
    constexpr int BG_EST_X = 62, BG_EST_Y = 63, BG_EST_Z = 64;

    // True biases (only if you actually store them; otherwise set to -1 and we’ll just plot estimates)
    constexpr int BG_TRUE_X = 68, BG_TRUE_Y = 69, BG_TRUE_Z = 70;

    std::vector<double> t;
    std::vector<double> bgx_true, bgy_true, bgz_true;
    std::vector<double> bgx_est,  bgy_est,  bgz_est;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v;
        v.reserve(80);

        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { v.push_back(std::stod(cell)); }
            catch (...) { /* skip */ }
        }

        const int need_cols = std::max({T_COL, BG_EST_Z, BG_TRUE_Z}) + 1;
        if ((int)v.size() < need_cols) continue;

        const double tt = v[T_COL];
        const double ex = v[BG_EST_X], ey = v[BG_EST_Y], ez = v[BG_EST_Z];

        const bool have_true = (BG_TRUE_X >= 0 && BG_TRUE_Y >= 0 && BG_TRUE_Z >= 0);
        const double tx = have_true ? v[BG_TRUE_X] : std::numeric_limits<double>::quiet_NaN();
        const double ty = have_true ? v[BG_TRUE_Y] : std::numeric_limits<double>::quiet_NaN();
        const double tz = have_true ? v[BG_TRUE_Z] : std::numeric_limits<double>::quiet_NaN();

        auto finite = [](double a){ return std::isfinite(a); };
        if (!finite(tt) || !finite(ex) || !finite(ey) || !finite(ez)) continue;
        if (have_true && (!finite(tx) || !finite(ty) || !finite(tz))) continue;

        t.push_back(tt);
        bgx_est.push_back(ex); bgy_est.push_back(ey); bgz_est.push_back(ez);

        if (have_true) {
            bgx_true.push_back(tx); bgy_true.push_back(ty); bgz_true.push_back(tz);
        }
    }
    file.close();

    if (t.empty()) {
        std::cerr << "Error: No valid gyro bias data found in " << filepath << std::endl;
        return;
    }

    // Downsample
    auto downsample7 = [](size_t max_pts,
                          std::vector<double>& t,
                          std::vector<double>& a1, std::vector<double>& a2, std::vector<double>& a3,
                          std::vector<double>& b1, std::vector<double>& b2, std::vector<double>& b3)
    {
        const size_t N = t.size();
        if (N <= max_pts) return;
        const size_t step = (N + max_pts - 1) / max_pts;

        std::vector<double> t2;  t2.reserve((N + step - 1) / step);
        std::vector<double> a12; a12.reserve(t2.capacity());
        std::vector<double> a22; a22.reserve(t2.capacity());
        std::vector<double> a32; a32.reserve(t2.capacity());
        std::vector<double> b12; b12.reserve(t2.capacity());
        std::vector<double> b22; b22.reserve(t2.capacity());
        std::vector<double> b32; b32.reserve(t2.capacity());

        for (size_t i = 0; i < N; i += step) {
            t2.push_back(t[i]);
            a12.push_back(a1[i]); a22.push_back(a2[i]); a32.push_back(a3[i]);
            b12.push_back(b1[i]); b22.push_back(b2[i]); b32.push_back(b3[i]);
        }

        t.swap(t2);
        a1.swap(a12); a2.swap(a22); a3.swap(a32);
        b1.swap(b12); b2.swap(b22); b3.swap(b32);
    };

    const bool have_true = (bgx_true.size() == t.size());
    if (have_true) {
        downsample7(20000, t, bgx_true, bgy_true, bgz_true, bgx_est, bgy_est, bgz_est);
    } else {
        auto downsample4 = [](size_t max_pts,
                              std::vector<double>& t,
                              std::vector<double>& a1, std::vector<double>& a2, std::vector<double>& a3)
        {
            const size_t N = t.size();
            if (N <= max_pts) return;
            const size_t step = (N + max_pts - 1) / max_pts;

            std::vector<double> t2;  t2.reserve((N + step - 1) / step);
            std::vector<double> a12; a12.reserve(t2.capacity());
            std::vector<double> a22; a22.reserve(t2.capacity());
            std::vector<double> a32; a32.reserve(t2.capacity());

            for (size_t i = 0; i < N; i += step) {
                t2.push_back(t[i]);
                a12.push_back(a1[i]); a22.push_back(a2[i]); a32.push_back(a3[i]);
            }

            t.swap(t2);
            a1.swap(a12); a2.swap(a22); a3.swap(a32);
        };
        downsample4(20000, t, bgx_est, bgy_est, bgz_est);
    }

    // Plot (3 figures)
    plt::figure_size(1600, 500);
    if (have_true) {
        plt::named_plot("bgyro_x true", t, bgx_true, "-");
        plt::named_plot("bgyro_x est",  t, bgx_est,  "C1-");
    } else {
        plt::named_plot("bgyro_x est",  t, bgx_est,  "C1-");
    }
    plt::xlabel("Time [s]");
    plt::ylabel("bgyro_x [rad/s]");
    plt::title("Gyro bias X");
    plt::grid(true);
    plt::legend();
    plt::show();

    plt::figure_size(1600, 500);
    if (have_true) {
        plt::named_plot("bgyro_y true", t, bgy_true, "-");
        plt::named_plot("bgyro_y est",  t, bgy_est,  "C1-");
    } else {
        plt::named_plot("bgyro_y est",  t, bgy_est,  "C1-");
    }
    plt::xlabel("Time [s]");
    plt::ylabel("bgyro_y [rad/s]");
    plt::title("Gyro bias Y");
    plt::grid(true);
    plt::legend();
    plt::show();

    plt::figure_size(1600, 500);
    if (have_true) {
        plt::named_plot("bgyro_z true", t, bgz_true, "-");
        plt::named_plot("bgyro_z est",  t, bgz_est,  "C1-");
    } else {
        plt::named_plot("bgyro_z est",  t, bgz_est,  "C1-");
    }
    plt::xlabel("Time [s]");
    plt::ylabel("bgyro_z [rad/s]");
    plt::title("Gyro bias Z");
    plt::grid(true);
    plt::legend();
    plt::show();
}

void plotOceanCurrent()
{
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

    // Columns (0-based): t=0, V_c=71, beta_c=72
    std::vector<double> t, Vc, betac;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> v; v.reserve(80);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { v.push_back(std::stod(cell)); }
            catch (const std::invalid_argument&) { /* skip */ }
        }
        if (v.size() >= 73) {               // need cols 0, 71, 72
            t.push_back(v[0]);
            Vc.push_back(10*v[71]);
            betac.push_back(v[72]);
        }
    }
    file.close();

    if (t.empty()) {
        std::cerr << "Error: No valid ocean current data found in " << filepath << std::endl;
        return;
    }

    // Use same decimation pattern as plotIMUGyro (4-vector overload)
    std::vector<double> dummy = betac;
    decimate_in_place(t, Vc, betac, dummy, 20000);

    // Optional: waypoint change markers (same as plotIMUGyro)
    std::vector<double> wpt_change_times;
    try { wpt_change_times = loadWaypointChangeTimes(); } catch (...) {}

    plt::figure_size(2400, 600);

    for (double tt : wpt_change_times) {
        std::vector<double> xl = {tt, tt};
        std::vector<double> yl = {-1000, 1000};
        plt::plot(xl, yl, "k--");
    }

    plt::named_plot("V_c [10*m/s]",    t, Vc,    "b-");
    plt::named_plot("beta_c [rad]", t, betac, "r-");
    plt::xlabel("Time [s]");
    plt::ylabel("Ocean current");
    plt::title("Ocean Current Speed and Direction");
    plt::grid(true);
    plt::legend();
    plt::show();
}


RealTimePlotter::RealTimePlotter() {
    plt::figure();   
    plt::xlabel("x(t) [m]");
    plt::ylabel("y(t) [m]");
    plt::title("Live Plot: Vessel Status");
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

void RealTimePlotter::draw_vessel(double x, double y, double theta, const std::string& style /* = "r-" */) {
    double L = 5.0;  // Hull length
    double W = 1;  // Half the width of each hull
    double gap = 1; // Space between hulls
    double bridge_L = 2.0; // Bridge length (shorter than hull)

    // Define a single connected shape (hulls + bridge)
    std::vector<double> x_shape = {
        -L/2, L/2.5, L/1.5,  bridge_L/2,  bridge_L/2,  L/1.5, L/2.5, -L/2, -L/2, -bridge_L/2, -bridge_L/2, -L/2, -L/2
    };
    
    std::vector<double> y_shape = {
        -W - gap/2, -W - gap/2, -gap/2, -gap/2, gap/2, gap/2, W + gap/2, W + gap/2, gap/2, gap/2, -gap/2, -gap/2, -W - gap/2
    };

    std::vector<double> x_rotated, y_rotated;
    x_rotated.reserve(x_shape.size());
    y_rotated.reserve(y_shape.size());
    for (size_t i = 0; i < x_shape.size(); ++i) {
        double x_new = x + x_shape[i] * sin(theta) + y_shape[i] * cos(theta);
        double y_new = y + x_shape[i] * cos(theta) - y_shape[i] * sin(theta);
        x_rotated.push_back(x_new);
        y_rotated.push_back(y_new);
    }

    plt::plot(x_rotated, y_rotated, style);
}

void RealTimePlotter::updatePlot(double x, double y, double psi, double arrowLength,
                                 std::vector<double> gx, std::vector<double> gy)
{
    // Reuse new overload; "estimate" == "real" to preserve old behavior
    updatePlot(x, y, psi, x, y, psi, arrowLength, gx, gy);
}

void RealTimePlotter::updatePlot(double x, double y, double psi_value,
                                 double x_est, double y_est, double psi_est,
                                 double arrowLength,
                                 const std::vector<double>& guidance_x,
                                 const std::vector<double>& guidance_y)
{
    // store histories
    m_x.push_back(x);           m_y.push_back(y);           m_psi.push_back(psi_value);
    m_x_est.push_back(x_est);   m_y_est.push_back(y_est);   m_psi_est.push_back(psi_est);

    // clear and redraw
    plt::cla();

    // --- draw order: ESTIMATED (red) first, then TRUE (green) to overlay ---
    // vessels
    draw_vessel(x_est, y_est, psi_est,   "r-"); // estimated: red
    draw_vessel(x,     y,     psi_value, "g-"); // true:      green

    // path (unchanged)
    if (!m_path_x.empty() && !m_path_y.empty()) {
        plt::plot(m_path_x, m_path_y, "k-");
    }

    // trails
    plt::plot(m_x_est, m_y_est, "r-"); // estimated trail: red dashed
    plt::plot(m_x,     m_y,     "g-");  // true trail:      green solid

    // guidance target(s): blue
    if (!guidance_x.empty()) {
        if (guidance_x.size() == 1) {
            plt::plot(guidance_x, guidance_y, "bo"); // blue point
        } else {
            plt::plot(guidance_x, guidance_y, "b-"); // blue curve
        }
    }

    // current positions (estimated first, then true so green is on top)
    plt::plot(std::vector<double>{x_est}, std::vector<double>{y_est}, "ro");
    plt::plot(std::vector<double>{x},     std::vector<double>{y},     "go");

    plt::xlabel("x(t) [m]");
    plt::ylabel("y(t) [m]");
    plt::title("Live Plot: True (green), Estimated (red), Guidance (blue)");
    plt::axis("equal");
    plt::grid(true);

    // plt::legend(); // enable if your matplotlib-cpp build supports labels

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

// Add this function to load waypoint change times
std::vector<double> loadWaypointChangeTimes() {
    std::vector<double> times;
    std::filesystem::path filepath = std::filesystem::path(getRepositoryPath()) / "data" / "wpt_change_times.csv";
    
    if (!std::filesystem::exists(filepath)) {
        return times; // Return empty vector if file doesn't exist
    }
    
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            try {
                times.push_back(std::stod(line));
            } catch (const std::invalid_argument&) {
                // Skip invalid entries
            }
        }
        file.close();
    }
    
    return times;
}
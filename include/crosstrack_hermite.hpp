#ifndef CROSSTRACK_HERMITE_LOS_HPP
#define CROSSTRACK_HERMITE_LOS_HPP

#include <vector>
#include <cmath>
#include <algorithm>

class CrosstrackHermiteLOS {
public:
    struct PathData {
        std::vector<double> w_path;
        std::vector<double> x_path;
        std::vector<double> y_path;
        std::vector<double> dx_path;
        std::vector<double> dy_path;
        std::vector<double> pi_h;
    };

    static double computeLOSAngle(
        const PathData& path,
        double x, double y, double h, double Delta_h,
        const std::vector<double>& pp_x,
        const std::vector<double>& pp_y,
        int& idx_start,
        int N_horizon,
        double gamma_h = -1);

private:
    static double persistentBetaHat;
};

#endif // CROSSTRACK_HERMITE_LOS_HPP
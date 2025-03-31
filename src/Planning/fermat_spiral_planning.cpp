#include "Planning/fermat_spiral_planning.hpp"
#include "Utilities/calculations.hpp"
#include "Utilities/plotting.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

FermatSpiralPath::FermatSpiralPath(double kappa_max) : kappa_max_(kappa_max){
    //theta_kappa_max = 0.2699549107922; 
    theta_kappa_max = std::sqrt(std::sqrt(7)/2 - 5/4);
    std::cout << "theta_kappa_max: " << theta_kappa_max << std::endl;
}

void FermatSpiralPath::updateWaypoints(const Waypoints& waypoints) {
    if (waypoints.size() < 2)
        throw std::runtime_error("At least two waypoints are required.");
    waypoints_ = waypoints;
}

FermatSpiralPath::FSParameters FermatSpiralPath::computeFSParameters(const Vector2D &A,
                                                                     const Vector2D &B,
                                                                     const Vector2D &C) const
{
    FSParameters params;

    Vector2D v_in  = (B - A).normalized();
    Vector2D v_out = (C - B).normalized();

    double dotVal = v_in.dot(v_out);
    params.delta_chi = std::acos(dotVal);

    params.rho = - sign(v_in.y * v_out.x - v_in.x * v_out.y);

    params.chi0    = std::atan2(v_in.y,  v_in.x);
    params.chi_end = std::atan2(v_out.y, v_out.x);

    params.theta_end = computeThetaEnd(params.delta_chi/2);
    double local_theta_kappa_max = std::min(params.theta_end, theta_kappa_max);
    
    params.k = computeScalingConstant(local_theta_kappa_max, kappa_max_);
    double l = computeDistanceForCorner(params.k, params.theta_end, params.delta_chi);
    
    params.wheel_over = B - v_in * l; 
    params.pull_out = B + v_out * l;
    params.x0 = params.wheel_over.x;
    params.y0 = params.wheel_over.y;
    params.x_end = params.pull_out.x;
    params.y_end = params.pull_out.y;

    params.u_max = std::sqrt(params.theta_end);

    if (std::isnan(params.chi0) || std::isnan(params.x0) || std::isnan(params.y0) ||
    std::isnan(params.chi_end) || std::isnan(params.x_end) || std::isnan(params.y_end) ||
    std::isnan(params.k) || std::isnan(params.rho)) {
    std::cerr << "Error: Invalid FSParameters detected!" << std::endl;
    std::abort();
    }

    return params;
}

// Convert to using u for mirrored
Vector2D FermatSpiralPath::computeSpiralPoint(double theta, double base_angle, double lambda,
                                              double k, double rho, double x, double y, double theta_max) const {

    double u = std::sqrt(theta);

    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        return Vector2D(x + k * u * std::cos(common),
                        y + k * u * std::sin(common));
    } else {
        double common = rho * u * u - base_angle;
        return Vector2D(x - k * u * std::cos(common),
                        y + k * u * std::sin(common));
    }
}

Vector2D FermatSpiralPath::computeSpiralDerivative(double theta, double base_angle, double lambda,
                                                     double k, double rho, double theta_max) const {
    double u = std::sqrt(theta);

    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        double dx_du = std::cos(common) - 2.0 * rho * u * u * std::sin(common);
        double dy_du = std::sin(common) + 2.0 * rho * u * u * std::cos(common);
        return (k) * Vector2D(dx_du, dy_du);
    } else {
        double common = rho * u * u - base_angle;
        double dx_du_prime = - std::cos(common) + 2.0 * rho * u * u * std::sin(common);
        double dy_du_prime =   std::sin(common) + 2.0 * rho * u * u * std::cos(common);
        return (k) * Vector2D(dx_du_prime, dy_du_prime);
    }
}


Vector2D FermatSpiralPath::computeSpiralSecondDerivative(double theta, double base_angle, double lambda,
                                                           double k, double rho, double theta_max) const {
    double u = std::sqrt(theta);
    
    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        double d2x_du2 = - 6*rho*u*std::sin(common) - 4*u*u*u*std::cos(common);
        double d2y_du2 =   6*rho*u*std::cos(common) - 4*u*u*u*std::sin(common);
        return (k) * Vector2D(d2x_du2, d2y_du2);
    } else {
        double common = rho * u * u - base_angle;
        double d2x_duprime2 = 6*rho*u*std::sin(common) + 4*u*u*u*std::cos(common);
        double d2y_duprime2 = 6*rho*u*std::cos(common) - 4*u*u*u*std::sin(common);
        return (k) * Vector2D(d2x_duprime2, d2y_duprime2);
    }
}

double FermatSpiralPath::f(double theta, double delta_chi) const {
    return theta + std::atan(2.0 * theta) - delta_chi;
}

double FermatSpiralPath::fprime(double theta) const {
    return 1.0 + 2.0 / (1.0 + 4.0 * theta * theta);
}

double FermatSpiralPath::fsecond(double theta) const {
    return -16.0 * theta / std::pow(1.0 + 4.0 * theta * theta, 2);
}

double FermatSpiralPath::computeThetaEnd(double delta_chi) const {
    double theta = delta_chi;  
    const double tol = 0.001;
    const int maxIter = 10;
    for (int i = 0; i < maxIter; ++i) {
        double f_val = f(theta, delta_chi);
        double fp = fprime(theta);
        double fpp = fsecond(theta);
        double denom = 2.0 * fp * fp - f_val * fpp;
        if (std::abs(denom) < 0.0000001)
            break;
        double delta = 2.0 * f_val * fp / denom;
        theta -= delta;
        if (std::abs(delta) < tol)
            break;
    }
    return theta;
}

double FermatSpiralPath::computeScalingConstant(double theta_kappa_max, double kappa_max) const {
    return (1.0 / kappa_max) * 2 * std::sqrt(theta_kappa_max) * 
           ((3.0 + 4.0 * theta_kappa_max * theta_kappa_max) /
           std::pow(1.0 + 4.0 * theta_kappa_max * theta_kappa_max, 1.5));
}

double FermatSpiralPath::computeDistanceForCorner(double k, double theta_end, double delta_chi) const {
    double l1 = k * std::sqrt(theta_end) * std::cos(theta_end);
    double h  = k * std::sqrt(theta_end) * std::sin(theta_end);
    double alpha = (M_PI - delta_chi) / 2.0;
    double l2 = h / std::tan(alpha);
    return l1 + l2;
}


PathPoint FermatSpiralPath::projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel) {
    PathPoint pp;
    Vector2D AB = B - A;
    double t = ((vessel - A).dot(AB)) / (AB.dot(AB));
    t = std::max(0.0, std::min(1.0, t));
    pp.pos = A + AB * t;
    pp.dpos = AB.normalized();
    pp.ddpos = Vector2D(0.0, 0.0);
    return pp;
}

// Function to calculate f(th)
double FermatSpiralPath::computeF(double th, const Vector2D& vessel, double base_angle, double lamda, 
    double k, double rho, double xCenter, double yCenter, double theta_end) {
Vector2D pos = computeSpiralPoint(th, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
Vector2D d1 = computeSpiralDerivative(th, base_angle, lamda, k, rho, theta_end);

if (std::isnan(pos.x) || std::isnan(pos.y)) {
std::cout << "Error: computeSpiralPoint returned NaN!" << std::endl;
std::abort();
}
if (std::isnan(d1.x) || std::isnan(d1.y)) {
std::cout << "Error: computeSpiralPointDerivative returned NaN!" << std::endl;
std::abort();
}

double dx = vessel.x - pos.x;
double dy = vessel.y - pos.y;
return d1.y * dy + d1.x * dx;
}

// Function to calculate fprime(th)
double FermatSpiralPath::computeFPrime(double th, const Vector2D& vessel, double base_angle, double lamda, 
         double k, double rho, double xCenter, double yCenter, double theta_end) {
Vector2D pos = computeSpiralPoint(th, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
Vector2D d1 = computeSpiralDerivative(th, base_angle, lamda, k, rho, theta_end);
Vector2D d2 = computeSpiralSecondDerivative(th, base_angle, lamda, k, rho, theta_end);

if (std::isnan(d2.x) || std::isnan(d2.y)) {
std::cout << "Error: computeSpiralSecondDerivative returned NaN!" << std::endl;
std::abort();
}

double dx = vessel.x - pos.x;
double dy = vessel.y - pos.y;
return d2.y * dy + d2.x * dx - d1.x * d1.x - d1.y * d1.y;
}

PathPoint FermatSpiralPath::projectOntoSpiral(Vector2D vessel, FSParameters params, double lamda) {
    double thetaGuess = params.theta_end;
    double maxIterations = 2000;
    double tolerance = 1e-16;

    double base_angle = (lamda > 0) ? params.chi0 : (params.chi_end);
    double xCenter    = (lamda > 0) ? params.x0  : params.x_end;
    double yCenter    = (lamda > 0) ? params.y0  : params.y_end;

    double k = params.k;
    double rho = params.rho;
    double theta_end = params.theta_end;

    std::cout << "Debug Info: " << std::endl;
    std::cout << "  thetaGuess: " << thetaGuess << std::endl;
    std::cout << "  params.theta_end: " << params.theta_end << std::endl;
    std::cout << "  base_angle: " << base_angle << std::endl;
    std::cout << "  xCenter: " << xCenter << std::endl;
    std::cout << "  yCenter: " << yCenter << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << "  rho: " << rho << std::endl;
    std::cout << "  lamda: " << lamda << std::endl;
    std::cout << "  vessel.x: " << vessel.x << " vessel.y: " << vessel.y << std::endl;
    
    double theta = thetaGuess;
    for (int i = 0; i < maxIterations; ++i)
    {
        double val = computeF(theta, vessel, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
        double der = computeFPrime(theta, vessel, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);

        if (std::fabs(der) < 1e-4) {
            std::cout << "Warning: fprime too low, applying damping" << std::endl;
            der = (der < 0) ? -1e-4 : 1e-4; // Apply damping to prevent division issues
        }

        double step = val / der;
        if (std::isnan(step)) {
            std::cout << "Error: step is NaN!" << std::endl;
            break;
        }

        if (step < tolerance) {
            break;
        }

        // Costum function for better convergence insted of using step directly
        theta -= (step > 0 ? 1 : (step < 0 ? -1 : 0)) * step * step;


        const double theta_start = 0;
        if (theta < theta_start) {
            theta = theta_start;
        }
        else if (theta > theta_end) {
            theta = theta_end;
        }

    }

    PathPoint closest_point;
    closest_point.pos = computeSpiralPoint(theta, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
    closest_point.dpos = computeSpiralDerivative(theta, base_angle, lamda, k, rho, theta_end);
    closest_point.ddpos = computeSpiralSecondDerivative(theta, base_angle, lamda, k, rho, theta_end);

    return closest_point;
}


double FermatSpiralPath::alongTrackError(Vector2D vessel, PathPoint path_point) {
    //std::cout << "dpos: (" << path_point.dpos.x << ", " << path_point.dpos.y << ")" << std::endl;
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double x_e = (vessel.x - path_point.pos.x)*std::cos(lamda) 
                +(vessel.y - path_point.pos.y)*std::sin(lamda);
    return x_e;
}

double FermatSpiralPath::crossTrackError(Vector2D vessel, PathPoint path_point) {
    //std::cout << "dpos: (" << path_point.dpos.x << ", " << path_point.dpos.y << ")" << std::endl;
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double y_e = -(vessel.x - path_point.pos.x)*std::sin(lamda) 
                 +(vessel.y - path_point.pos.y)*std::cos(lamda);
    return y_e;
}


PathTrackingInfo FermatSpiralPath::getClosestPoint(const Vector2D vessel_position, int &wpt_index) {
    
    Vector2D vessel_pos = vessel_position;

    if (wpt_index == 0 || wpt_index == 1){ wpt_index = 2;}
    Vector2D wpt_prev2  = waypoints_[wpt_index-2];
    Vector2D wpt_prev   = waypoints_[wpt_index-1];
    Vector2D wpt        = waypoints_[wpt_index];
    Vector2D wpt_next   = waypoints_[wpt_index+1];

    FSParameters params;

    PathPoint line_prev_point;
    double x_e_line_prev;
    double y_e_line_prev;
    double line_prev_error;

    PathPoint FS_point;
    double x_e_FS;
    double y_e_FS;
    double FS_error;

    PathPoint FS_mirrored_point;
    double x_e_FS_mirrored;
    double y_e_FS_mirrored;
    double FS_mirrored_error;

    PathPoint line_point;
    double x_e_line;
    double y_e_line;
    double line_error;

    PathTrackingInfo tracking_info;

    // Current Segments
    
    params = computeFSParameters(wpt_prev2, wpt_prev, wpt);

    if (wpt_index == 2) { 
        line_prev_point = projectOntoLine(wpt_prev2, params.wheel_over, vessel_position);
    }
    else {
        line_prev_point = projectOntoLine(stored_pull_out, params.wheel_over, vessel_position);
    }
    x_e_line_prev = alongTrackError(vessel_position, line_prev_point);
    y_e_line_prev = crossTrackError(vessel_position, line_prev_point);
    line_prev_error = std::sqrt(x_e_line_prev*x_e_line_prev + y_e_line_prev*y_e_line_prev);

    FS_point = projectOntoSpiral(vessel_pos, params, +1);
    x_e_FS = alongTrackError(vessel_position, FS_point);
    y_e_FS = crossTrackError(vessel_position, FS_point);
    FS_error = std::sqrt(x_e_FS*x_e_FS + y_e_FS*y_e_FS);

    FS_mirrored_point = projectOntoSpiral(vessel_pos, params, -1);
    x_e_FS_mirrored = alongTrackError(vessel_position, FS_mirrored_point);
    y_e_FS_mirrored = crossTrackError(vessel_position, FS_mirrored_point);
    FS_mirrored_error = std::sqrt(x_e_FS_mirrored*x_e_FS_mirrored + y_e_FS_mirrored*y_e_FS_mirrored);


    Vector2D prev_pull_out = params.pull_out;
    params = computeFSParameters(wpt_prev, wpt, wpt_next);

    if (wpt_index < waypoints_.size()-1) { //There is more segments
        line_point = projectOntoLine(prev_pull_out, params.wheel_over, vessel_position);
    }
    else { //There is not more segments
        line_point = projectOntoLine(prev_pull_out, wpt_next, vessel_position);
    }
    x_e_line = alongTrackError(vessel_position, line_point); 
    y_e_line = crossTrackError(vessel_position, line_point);
    line_error = std::sqrt(x_e_line*x_e_line + y_e_line*y_e_line);


    if (line_prev_error < FS_error && line_prev_error < FS_mirrored_error && line_prev_error < line_error){
        tracking_info.point = line_prev_point;
        tracking_info.x_e = x_e_line_prev;
        tracking_info.y_e = y_e_line_prev;
    } 
    else if (FS_error < FS_mirrored_error && FS_error < line_error){
        tracking_info.point = FS_point;
        tracking_info.x_e = x_e_FS;
        tracking_info.y_e = y_e_FS;
    } 
    else if (FS_mirrored_error < line_error){
        tracking_info.point = FS_mirrored_point;
        tracking_info.x_e = x_e_FS_mirrored;
        tracking_info.y_e = y_e_FS_mirrored;
    } 
    else{
        tracking_info.point = line_point;
        tracking_info.x_e = x_e_line;
        tracking_info.y_e = y_e_line;
        if (wpt_index < waypoints_.size()-2){
            stored_pull_out = prev_pull_out;
            wpt_index++;
        }
    }
    return tracking_info;
}

Waypoints FermatSpiralPath::samplePath(double delta) 
{
    Waypoints path;
    Vector2D prev_end = waypoints_[0];
    path.push_back(prev_end);

    std::vector<Vector2D> vessels;
    std::vector<Vector2D> projections;

    for (size_t i = 1; i < waypoints_.size() - 1; ++i)
    {
        const Vector2D &A = waypoints_[i - 1];
        const Vector2D &B = waypoints_[i];
        const Vector2D &C = waypoints_[i + 1];

        FSParameters params = computeFSParameters(A, B, C);

        // 1) Straight line from the last end to the wheel‐over point
        Vector2D dir = (params.wheel_over - prev_end).normalized();
        double dist = (params.wheel_over - prev_end).norm();

        for (double d = 0.0; d < dist; d += 0.5) {
            path.push_back(prev_end + dir * d);
        }
        path.push_back(params.wheel_over);

        // 2) Forward FS: u ∈ [0..u_max]
        for (double theta = 0.0; theta <= params.theta_end; theta += delta)
        {
            Vector2D pt = computeSpiralPoint(
                theta,
                params.chi0,     
                +1.0,           
                params.k,
                params.rho,
                params.x0,
                params.y0,
                params.theta_end    
            );
            path.push_back(pt);

            vessels.push_back(pt + Vector2D(-1, 1));
            PathPoint projection = projectOntoSpiral(pt + Vector2D(-1, 1), params, +1);
            projections.push_back(projection.pos);
        }

        // 3) Mirrored FS: u ∈ [0.0 ,u_max]
        for (double theta = params.theta_end; theta >= 0.0; theta -= delta)
        {
            Vector2D pt = computeSpiralPoint(
                theta,
                params.chi_end,  
                -1.0,            
                params.k,
                params.rho,
                params.x_end,
                params.y_end,
                params.theta_end
            );
            path.push_back(pt);

            vessels.push_back(pt + Vector2D(-1, 1)); // Vessel position (simulating)
            PathPoint projection = projectOntoSpiral(pt + Vector2D(-1, 1), params, -1); 
            projections.push_back(projection.pos);
        }

        // 4) Update prev_end to the final "pull‐out" point
        prev_end = params.pull_out;
    }

    // Final straight line from last corner to the last waypoint
    if (!waypoints_.empty()) {
        Vector2D last_wp = waypoints_.back();
        Vector2D dir = (last_wp - prev_end).normalized();
        double dist = (last_wp - prev_end).norm();
        for (double d = 0.0; d < dist; d += 0.5) {
            path.push_back(prev_end + dir * d);
        }
        path.push_back(last_wp);
    }

    plot_points(vessels, projections);

    return path;
}


void FermatSpiralPath::printParameters() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Kappa max: " << kappa_max_ << "\n";
    std::cout << "Total waypoints: " << waypoints_.size() << "\n";
    for (size_t i = 0; i < std::min(waypoints_.size(), size_t(5)); ++i) {
        std::cout << "Waypoint " << i << ": (" << waypoints_[i].x << ", " << waypoints_[i].y << ")\n";
    }
}

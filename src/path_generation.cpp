#include "path_generation.hpp"
#include <matplotlibcpp.h>
#include <iostream>
#include <iomanip>

//------------------------------------------------------
// StraightLinePath Class Implementation
//------------------------------------------------------
StraightLinePath::StraightLinePath() {}

void StraightLinePath::updateWaypoints(const Waypoints& waypoints) {
    if (waypoints.size() < 2)
        throw std::runtime_error("At least two waypoints are required.");
    waypoints_ = waypoints;
}

double StraightLinePath::alongTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double x_e = (vessel.x - path_point.pos.x)*std::cos(lamda) 
                +(vessel.y - path_point.pos.y)*std::sin(lamda);
    return x_e;
}

double StraightLinePath::crossTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double y_e = -(vessel.x - path_point.pos.x)*std::sin(lamda) 
                 +(vessel.y - path_point.pos.y)*std::cos(lamda);
    return y_e;
}

PathPoint StraightLinePath::getClosestPoint(const Vector2D vessel_position, int& wpt_index) {
    
    Vector2D vessel_pos = vessel_position;
    Vector2D wpt_prev   = waypoints_[wpt_index-1];
    Vector2D wpt        = waypoints_[wpt_index];
    Vector2D wpt_next   = waypoints_[wpt_index+1];

    PathPoint line_prev_point;
    double x_e_line_prev;
    double y_e_line_prev;
    double line_prev_error;

    PathPoint line_point;
    double x_e_line;
    double y_e_line;
    double line_error;

    PathPoint closest_point;

    if (wpt_index > 1){

        line_prev_point = projectOntoLine(wpt_prev, wpt, vessel_position);
        x_e_line_prev = alongTrackError(vessel_position, line_prev_point);
        y_e_line_prev = crossTrackError(vessel_position, line_prev_point);
        line_prev_error = std::sqrt(x_e_line_prev*x_e_line_prev + y_e_line_prev*y_e_line_prev);
    }

    line_point = projectOntoLine(wpt, wpt_next, vessel_position);
    x_e_line = alongTrackError(vessel_position, line_point); 
    y_e_line = crossTrackError(vessel_position, line_point);
    line_error = std::sqrt(x_e_line*x_e_line + y_e_line*y_e_line);

    if (wpt_index == 1){
        closest_point = line_point;
        if (wpt_index < waypoints_.size()-1){
        wpt_index++;
        }
        return closest_point;
    }

    if (line_prev_error < line_error){
        closest_point = line_prev_point;    
    } 
    else {
        closest_point = line_point;
        if (wpt_index < waypoints_.size()-1){
            wpt_index++;
        }
    } 

    //std::cout << "closest_point: " << closest_point.pos.x << ", " << closest_point.pos.y << "\n" <<std::endl;
    //std::cout << "y_e_line: " << y_e_line << "\n" <<std::endl;
    return closest_point;
}


PathPoint StraightLinePath::projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel) {
    PathPoint pp;
    Vector2D AB = B - A;
    double t = ((vessel - A).dot(AB)) / (AB.dot(AB));
    t = std::max(0.0, std::min(1.0, t));
    pp.pos = A + AB * t;
    pp.dpos = AB.normalized();
    pp.ddpos = Vector2D(0.0, 0.0);
    return pp;
}


std::vector<Vector2D> StraightLinePath::samplePath(double delta) const {
    return waypoints_;
}

void StraightLinePath::printParameters() const {
    std::cout << "Total waypoints: " << waypoints_.size() << "\n";
    for (size_t i = 0; i < std::min(waypoints_.size(), size_t(5)); ++i) {
        std::cout << "Waypoint " << i << ": (" << waypoints_[i].x << ", " << waypoints_[i].y << ")\n";
    }
}


//------------------------------------------------------
// FermatSpiralPath Class Implementation
//------------------------------------------------------
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
    
    params.point1 = B - v_in * l; 
    params.point2 = B + v_out * l;
    params.x0 = params.point1.x;
    params.y0 = params.point1.y;
    params.x_end = params.point2.x;
    params.y_end = params.point2.y;

    params.u_max = std::sqrt(params.theta_end);
    params.u_mid = std::sqrt(params.theta_end/2);

    return params;
}

// Convert to using u for mirrored
Vector2D FermatSpiralPath::computeSpiralPoint(double theta, double base_angle, double lambda,
                                              double k, double rho, double x, double y, double theta_max) const {

    double u = std::sqrt(theta);
    double u_m = std::sqrt(theta_max - theta);
    if (lambda > 0) {
        return Vector2D(x + k * u * std::cos(rho * u * u + base_angle),
                        y + k * u * std::sin(rho * u * u + base_angle));
    } else {
        return Vector2D(x + k * u_m * std::cos(rho * (- u_m * u_m) + base_angle),
                        y + k * u_m * std::sin(rho * (- u_m * u_m) + base_angle));
    }
}

// Probably an error in this implementation
Vector2D FermatSpiralPath::computeSpiralDerivative(double theta, double base_angle, double lambda,
                                                     double k, double rho, double theta_max) const {
    double u = std::sqrt(theta);
    double u_m = std::sqrt(theta_max - theta);  

    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        double dx_du = std::cos(common) - 2.0 * rho * u * u * std::sin(common);
        double dy_du = std::sin(common) + 2.0 * rho * u * u * std::cos(common);
        return k * Vector2D(dx_du, dy_du);
    } else {
        double common = rho * (- u_m * u_m) + base_angle;
        double dx_du_prime = std::cos(common) + 2.0 * rho * u_m * u_m * std::sin(common);
        double dy_du_prime = std::sin(common) + 2.0 * rho * u_m * u_m * std::cos(common);
        return (-k/(2*u_m)) * Vector2D(dx_du_prime, dy_du_prime);
    }
}

Vector2D FermatSpiralPath::computeSpiralSecondDerivative(double theta, double base_angle, double lambda,
                                                           double k, double rho, double theta_max) const {
    double u = std::sqrt(theta);
    double u_m = std::sqrt(theta_max - theta);      

    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        double d2x_du2 = (4*u*u*u*u + 1) * std::cos(common) + 4*rho*u*u*std::sin(common);
        double d2y_du2 = (4*u*u*u*u + 1) * std::sin(common) - 4*rho*u*u*std::cos(common);
        return (-k/(4*std::pow(u*u, 3/2))) * Vector2D(d2x_du2, d2y_du2);
    } else {
        double common = rho * (- u_m * u_m) + base_angle;
        double d2x_duprime2 = (4*(u_m*u_m*u_m*u_m) + 1) * std::cos(common) - 4*rho*u_m*u_m*std::sin(common);
        double d2y_duprime2 = (4*(u_m*u_m*u_m*u_m) + 1) * std::sin(common) + 4*rho*u_m*u_m*std::cos(common);
        return (-k / (4*std::pow(u_m*u_m, 3/2))) * Vector2D(d2x_duprime2, d2y_duprime2);
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
    double theta = delta_chi / 2.0;  // initial guess
    const double tol = 1e-3;
    const int maxIter = 10;
    for (int i = 0; i < maxIter; ++i) {
        double f_val = f(theta, delta_chi);
        double fp = fprime(theta);
        double fpp = fsecond(theta);
        double denom = 2.0 * fp * fp - f_val * fpp;
        if (std::abs(denom) < 1e-8)
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

PathPoint FermatSpiralPath::projectOntoSpiral(Vector2D vessel, FSParameters params, double lamda) {
    double tolerance = 1e-6;
    double thetaGuess = params.theta_end / 2.0;
    double maxIterations = 10;

    double base_angle = (lamda > 0) ? params.chi0 : (params.chi_end);
    double xCenter    = (lamda > 0) ? params.x0  : params.x_end;
    double yCenter    = (lamda > 0) ? params.y0  : params.y_end;

    double k = params.k;
    double rho = params.rho;
    double theta_end = params.theta_end;

    auto f = [&](double th) {
        Vector2D pos = computeSpiralPoint(th, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
        Vector2D d1  = computeSpiralDerivative(th, base_angle, lamda, k, rho, theta_end);
        double dx = vessel.x - pos.x;
        double dy = vessel.y - pos.y;
        return d1.y * dy + d1.x * dx;
    };
    auto fprime = [&](double th) {
        Vector2D pos = computeSpiralPoint(th, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
        Vector2D d1  = computeSpiralDerivative(th, base_angle, lamda, k, rho, theta_end);
        Vector2D d2  = computeSpiralSecondDerivative(th, base_angle, lamda, k, rho, theta_end);

        double dx = vessel.x - pos.x;
        double dy = vessel.y - pos.y;
        return d2.y * dy + d2.x * dx - d2.x * d2.x - d2.y * d2.y;
    };

    double theta = thetaGuess;
    for (int i = 0; i < maxIterations; ++i)
    {
        double val = f(theta);
        double der = fprime(theta);

        // Avoid division by a near-zero derivative.
        if (std::fabs(der) < 1e-14)
            break;

        double step = val / der;
        theta -= step;

        // Clamp theta to the valid interval [thetaMin, thetaMax].
        const double epsilon = 1e-8;
        if (theta < epsilon)
            theta = epsilon;
        if (theta > theta_end)
            theta = theta_end;

        if (std::fabs(step) < tolerance)
            break;
    }

    PathPoint closest_point;
    closest_point.pos = computeSpiralPoint(theta, base_angle, lamda, k, rho, xCenter, yCenter, theta_end);
    closest_point.dpos = computeSpiralDerivative(theta, base_angle, lamda, k, rho, theta_end);
    closest_point.ddpos = computeSpiralSecondDerivative(theta, base_angle, lamda, k, rho, theta_end);

    return closest_point;
}


double FermatSpiralPath::alongTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double x_e = (vessel.x - path_point.pos.x)*std::cos(lamda) 
                +(vessel.y - path_point.pos.y)*std::sin(lamda);
    return x_e;
}

double FermatSpiralPath::crossTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double y_e = -(vessel.x - path_point.pos.x)*std::sin(lamda) 
                 +(vessel.y - path_point.pos.y)*std::cos(lamda);
    return y_e;
}


PathPoint FermatSpiralPath::getClosestPoint(const Vector2D vessel_position, int& wpt_index) {
    
    Vector2D vessel_pos = vessel_position;
    Vector2D wpt_prev   = waypoints_[wpt_index-1];
    Vector2D wpt        = waypoints_[wpt_index];
    Vector2D wpt_next   = waypoints_[wpt_index+1];

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

    PathPoint closest_point;

    if (wpt_index > 1){
        Vector2D wpt_prev2  = waypoints_[wpt_index-2];
        FSParameters params = computeFSParameters(wpt_prev2, wpt_prev, wpt);

        line_prev_point = projectOntoLine(point2_prev, params.point1, vessel_position);
        x_e_line_prev = alongTrackError(vessel_position, line_prev_point);
        y_e_line_prev = crossTrackError(vessel_position, line_prev_point);
        line_prev_error = std::sqrt(x_e_line_prev*x_e_line_prev + y_e_line_prev*y_e_line_prev);
        point2_prev = params.point2;

        FS_point = projectOntoSpiral(vessel_pos, params, +1);
        x_e_FS = alongTrackError(vessel_position, FS_point);
        y_e_FS = crossTrackError(vessel_position, FS_point);
        FS_error = std::sqrt(x_e_FS*x_e_FS + y_e_FS*y_e_FS);

        FS_mirrored_point = projectOntoSpiral(vessel_pos, params, -1);
        x_e_FS_mirrored = alongTrackError(vessel_position, FS_mirrored_point);
        y_e_FS_mirrored = crossTrackError(vessel_position, FS_mirrored_point);
        FS_mirrored_error = std::sqrt(x_e_FS_mirrored*x_e_FS_mirrored+ y_e_FS_mirrored*y_e_FS_mirrored);
    }

    FSParameters params = computeFSParameters(wpt_prev, wpt, wpt_next);
    Vector2D wheel_over = params.point1;
    Vector2D pull_out   = params.point2;

    if (wpt_index > 1){
        line_point = projectOntoLine(point2_prev, pull_out, vessel_position);
    } else{
        line_point = projectOntoLine(wpt_prev, wheel_over, vessel_position);
    }
    x_e_line = alongTrackError(vessel_position, line_point); 
    y_e_line = crossTrackError(vessel_position, line_point);
    line_error = std::sqrt(x_e_line*x_e_line + y_e_line*y_e_line);


    if (wpt_index == 1){
        closest_point = line_point;
        if (wpt_index < waypoints_.size()-1){
        wpt_index++;
        }
        return closest_point;
    }

    if (line_prev_error < FS_error && line_prev_error < FS_mirrored_error && line_prev_error < line_error){
        closest_point = line_prev_point; 
    } 
    else if (FS_mirrored_error < FS_error && FS_mirrored_error < line_error){
        closest_point = FS_mirrored_point;
    } 
    else if (FS_error < line_error){
        closest_point = FS_point;
    } 
    else{
        closest_point = line_point;
        if (wpt_index < waypoints_.size()-1){
        wpt_index++;
        }
    }
    //std::cout << "closest_point: " << closest_point.pos.x << ", " << closest_point.pos.y << "\n" <<std::endl;
    //std::cout << "y_e_line: " << y_e_line << "\n" <<std::endl;
    return closest_point;
}

Waypoints FermatSpiralPath::samplePath(double delta) const
{
    Waypoints path;
    Vector2D prev_end = waypoints_[0];
    path.push_back(prev_end);

    for (size_t i = 1; i < waypoints_.size() - 1; ++i)
    {
        const Vector2D &A = waypoints_[i - 1];
        const Vector2D &B = waypoints_[i];
        const Vector2D &C = waypoints_[i + 1];

        FSParameters params = computeFSParameters(A, B, C);

        // 1) Straight line from the last end to the wheel‐over point1
        Vector2D dir = (params.point1 - prev_end).normalized();
        double dist = (params.point1 - prev_end).norm();

        for (double d = 0.0; d < dist; d += 0.5) {
            path.push_back(prev_end + dir * d);
        }
        path.push_back(params.point1);

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
        }

        // 3) Mirrored FS: u ∈ [0.0 ,u_max]
        for (double theta = 0.0; theta <= params.theta_end; theta += delta)
        {
            Vector2D pt = computeSpiralPoint(
                theta,
                params.chi_end + M_PI,  
                -1.0,            
                params.k,
                params.rho,
                params.x_end,
                params.y_end,
                params.theta_end
            );
            path.push_back(pt);
        }

        // 4) Update prev_end to the final “pull‐out” point2
        prev_end = params.point2;
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

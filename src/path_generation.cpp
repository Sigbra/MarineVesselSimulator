#include "path_generation.hpp"
#include <matplotlibcpp.h>
#include <iostream>
#include <iomanip>

//------------------------------------------------------
// StraightLinePath Class Implementation
//------------------------------------------------------
StraightLinePath::StraightLinePath() {
    // Initialize waypoints to (0,0); they will be updated later.
    wpt_prev_ = Vector2D(0.0, 0.0);
    wpt_      = Vector2D(0.0, 0.0);
}

void StraightLinePath::updateWaypoints(const Vector2D &wpt_prev, const Vector2D &wpt) {
    // Update the waypoints.
    wpt_prev_ = wpt_prev;
    wpt_      = wpt;
}

Vector2D StraightLinePath::getPoint(double u) const {
    return wpt_prev_ + (wpt_ - wpt_prev_).normalized() * u;
}

Vector2D StraightLinePath::getDerivative(double u) const {
    return (wpt_ - wpt_prev_).normalized();
}

Vector2D StraightLinePath::getSecondDerivative(double u) const {
    return Vector2D(0.0, 0.0);
}

PathPoint StraightLinePath::getClosestPoint(const Vector2D &vessel, double normal_angle) const {
    PathPoint pp;
    // Compute the vector from the previous waypoint to the current one.
    Vector2D d = wpt_ - wpt_prev_;
    double d_squared = d.dot(d);
    double u = 0.0;
    
    if(d_squared > 0.0) {
        // Projection factor: how far along the line (as a fraction) the projection is.
        u = (vessel - wpt_prev_).dot(d) / d_squared;
        // Clamp u to the interval [0, 1] so the result lies on the segment.
        if(u < 0.0) {
            u = 0.0;
        } else if(u > 1.0) {
            u = 1.0;
        }
    }
    
    pp.pos = getPoint(u);
    pp.dpos = getDerivative(u);
    pp.ddpos = getSecondDerivative(u);
    return pp;
}

std::vector<Vector2D> StraightLinePath::samplePath(double delta_u) const {
    std::vector<Vector2D> points;
    for (double u = 0.0; u <= 1.0; u += delta_u)
        points.push_back(getPoint(u));
    return points;
}

void StraightLinePath::printParameters() const {
    std::cout << "Straight line path parameters:\n";
    std::cout << "wpt_prev: (" << wpt_prev_.x << ", " << wpt_prev_.y << ")\n";
    std::cout << "wpt: (" << wpt_.x << ", " << wpt_.y << ")\n";
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
    double u_max = std::sqrt(theta_max);
    if (lambda > 0) {
        return Vector2D(x + k * u * std::cos(rho * u * u + base_angle),
                        y + k * u * std::sin(rho * u * u + base_angle));
    } else {
        return Vector2D(x + k * std::sqrt(theta_max - theta) * std::cos(rho * (theta - theta_max) + base_angle),
                        y + k * std::sqrt(theta_max - theta) * std::sin(rho * (theta - theta_max) + base_angle));
    }
}

// Probably an error in this implementation
Vector2D FermatSpiralPath::computeSpiralDerivative(double theta, double base_angle, double lambda,
                                                     double k, double rho, double theta_max) const {
    double u = std::sqrt(theta);
    double u_m = std::sqrt(theta - theta_max);
    double u_m2 = std::sqrt(theta_max - theta);    

    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        double dx_du = std::cos(common) - 2.0 * rho * u * u * std::sin(common);
        double dy_du = std::sin(common) + 2.0 * rho * u * u * std::cos(common);
        return (k/(2*u)) * Vector2D(dx_du, dy_du);
    } else {
        double common_m = rho * u_m * u_m + base_angle;
        double common_m2 = rho * u_m2 * u_m2 + base_angle;
        double dx_du_prime = std::cos(common_m) - 2.0 * rho * u_m2 * u_m2 * std::sin(common_m);
        double dy_du_prime = std::sin(common_m) + 2.0 * rho * u_m2 * u_m2 * std::cos(common_m);
        return (-k/(2*u_m2)) * Vector2D(dx_du_prime, dy_du_prime);
    }
}

Vector2D FermatSpiralPath::computeSpiralSecondDerivative(double theta, double base_angle, double lambda,
                                                           double k, double rho, double theta_max) const {
    double u = std::sqrt(theta);
    double u_m = std::sqrt(theta - theta_max);
    double u_m2 = std::sqrt(theta_max - theta);       

    if (lambda > 0) {
        double common = rho * u * u + base_angle;
        double d2x_du2 = (4*u*u*u*u + 1) * std::cos(common) + 4*rho*u*u*std::sin(common);
        double d2y_du2 = (4*u*u*u*u + 1) * std::sin(common) - 4*rho*u*u*std::cos(common);
        return (-k/(4*u*u*u)) * Vector2D(d2x_du2, d2y_du2);
    } else {
        double common = rho * u_m * u_m + base_angle;
        double d2x_duprime2 = (4*(u_m*u_m*u_m*u_m) + 1) * std::cos(common) - 4*rho*u_m2*u_m2*std::sin(common);
        double d2y_duprime2 = (4*(u_m*u_m*u_m*u_m) + 1) * std::sin(common) + 4*rho*u_m2*u_m2*std::cos(common);
        return (-k / (4*u_m2*u_m2*u_m2)) * Vector2D(d2x_duprime2, d2y_duprime2);
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

double FermatSpiralPath::crossTrackError(Vector2D vessel, PathPoint path_point) {
    double lamda = std::atan2(path_point.dpos.y, path_point.dpos.x);
    double y_e = -(vessel.x - path_point.pos.x)*std::sin(lamda) 
                 +(vessel.y - path_point.pos.y)*std::cos(lamda);
    return y_e;
}

PathPoint FermatSpiralPath::getClosestPoint(const Vector2D vessel_position, int index) {

    Vector2D vessel_pos = vessel_position;
    Vector2D wpt_prev   = waypoints_[index-1];
    Vector2D wpt        = waypoints_[index];
    Vector2D wpt_next   = waypoints_[index+1];

    FSParameters params = computeFSParameters(wpt_prev, wpt, wpt_next);
    Vector2D wheel_over = params.point1;
    Vector2D pull_out   = params.point2;

    PathPoint line_segment_point;
    if (index == 1){
        line_segment_point = projectOntoLine(wpt_prev, wheel_over, vessel_position);
    }
    else {
        line_segment_point = projectOntoLine(point2_prev, pull_out, vessel_position);
    }
    point2_prev = params.point2;
    double y_e_line = crossTrackError(vessel_position, line_segment_point);

    PathPoint FS_point = projectOntoSpiral(vessel_pos, params, +1);
    double y_e_FS = crossTrackError(vessel_position, FS_point);

    PathPoint FS_mirrored_point = projectOntoSpiral(vessel_pos, params, -1);;
    double y_e_FS_mirrored = crossTrackError(vessel_position, FS_mirrored_point);

    // Return the point from the segment with smallest y_e
    PathPoint closest_point;
    if (y_e_line < y_e_FS && y_e_line < y_e_FS_mirrored){ 
        closest_point = line_segment_point;
    } 
    else if (y_e_FS < y_e_FS_mirrored){
        closest_point = FS_point;
    } 
    else {
        closest_point = FS_mirrored_point;
    }
    std::cout << "closest_point: " << closest_point.pos.x << ", " << closest_point.pos.y << std::endl;
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

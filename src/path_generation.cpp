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
FermatSpiralPath::FermatSpiralPath(double kappa_max) : kappa_max_(kappa_max){}

void FermatSpiralPath::updateWaypoints(const Waypoints& waypoints) {
    if (waypoints.size() < 2)
        throw std::runtime_error("At least two waypoints are required.");
    waypoints_ = waypoints;
}

FermatSpiralPath::FSParameters FermatSpiralPath::computeFSParameters(const Vector2D &A, 
                                                                      const Vector2D &B, 
                                                                      const Vector2D &C) const {
    FSParameters params;
    Vector2D v_in  = (B - A).normalized();
    Vector2D v_out = (C - B).normalized();
    double dotVal = std::max(-1.0, std::min(1.0, v_in.dot(v_out)));
    params.delta_chi = std::acos(dotVal);
    params.rho = - sign(v_in.y * v_out.x - v_in.x * v_out.y);
    params.chi0 = std::atan2(v_in.y, v_in.x);
    params.chi_end = std::atan2(v_out.y, v_out.x);
    // Compute theta_end using Halley's method:
    params.theta_end = computeThetaEnd(params.delta_chi);
    // Now the parameter θ itself spans [0, theta_end].
    params.u_max = params.theta_end;  // now u_max is theta_end
    const double theta_max_intrinsic = 0.27;
    double theta_kappa_max = (params.theta_end < theta_max_intrinsic) ? params.theta_end : theta_max_intrinsic;
    params.k = computeScalingConstant(theta_kappa_max, kappa_max_);
    
    // Compute the wheel-over point p0:
    double l1 = params.k * std::sqrt(params.theta_end) * std::cos(params.theta_end);
    Vector2D p0 = B - v_in * l1;
    params.x0 = p0.x;
    params.y0 = p0.y;
    
    // Compute the pull-out point p_end (using the forward FS equation at theta = theta_end).
    Vector2D pend = computeSpiralPoint(params.theta_end, params.chi0, +1, params.k, params.rho, params.x0, params.y0, 0.0);
    params.x_end = pend.x;
    params.y_end = pend.y;
    
    // Compute the meeting parameter θ_mid so that the forward FS covers half the turn.
    double halfTurn = (params.chi_end - params.chi0) / 2.0;
    params.u_mid = std::sqrt(std::fabs(halfTurn));  // note: with ρ = ±1, this is √(|χ_end-χ0|/2)
    
    return params;
}

// Forward: use lambda > 0; mirrored: lambda < 0.
// When lambda > 0, theta_end is not used (pass 0).
Vector2D FermatSpiralPath::computeSpiralPoint(double theta, double base_angle, double lambda,
                                                double k, double rho, double x, double y, double theta_end) const {
    if (lambda > 0) {
        // Forward FS:
        // p_FS(θ) = [ x + k*sqrt(θ) cos(ρθ + base_angle);
        //             y + k*sqrt(θ) sin(ρθ + base_angle) ]
        return Vector2D(x + k * std::sqrt(theta) * std::cos(rho * theta + base_angle),
                        y + k * std::sqrt(theta) * std::sin(rho * theta + base_angle));
    } else {
        // Mirrored FS:
        // Let τ = theta_end - theta.
        double tau = theta_end - theta;
        // p_FS_M(θ) = [ x + k*sqrt(τ) cos(ρ(θ - theta_end) + base_angle);
        //               y + k*sqrt(τ) sin(ρ(θ - theta_end) + base_angle) ]
        return Vector2D(x + k * std::sqrt(tau) * std::cos(rho * (theta - theta_end) + base_angle),
                        y + k * std::sqrt(tau) * std::sin(rho * (theta - theta_end) + base_angle));
    }
}


Vector2D FermatSpiralPath::computeSpiralDerivative(double theta, double base_angle, double lambda,
                                                     double k, double rho, double theta_end) const {
    if (lambda > 0) {
        // Forward FS derivative:
        // p_FS_dot(θ) = k/(2√θ)*[ cos(ρθ+base_angle) - 2ρθ sin(ρθ+base_angle);
        //                         sin(ρθ+base_angle) + 2ρθ cos(ρθ+base_angle) ]
        return Vector2D( k/(2.0 * std::sqrt(theta)) *
                         ( std::cos(rho*theta + base_angle) - 2.0 * rho * theta * std::sin(rho*theta + base_angle) ),
                         k/(2.0 * std::sqrt(theta)) *
                         ( std::sin(rho*theta + base_angle) + 2.0 * rho * theta * std::cos(rho*theta + base_angle) ));
    } else {
        // Mirrored FS derivative:
        double tau = theta_end - theta;
        // p_FS_M_dot(θ) = -k/(2√τ)*[ cos(ρ(θ-θ_end)+base_angle) + 2ρτ sin(ρ(θ-θ_end)+base_angle);
        //                           sin(ρ(θ-θ_end)+base_angle) - 2ρτ cos(ρ(θ-θ_end)+base_angle) ]
        return Vector2D( -k/(2.0 * std::sqrt(tau)) *
                         ( std::cos(rho*(theta - theta_end) + base_angle) + 2.0 * rho * tau * std::sin(rho*(theta - theta_end) + base_angle) ),
                         -k/(2.0 * std::sqrt(tau)) *
                         ( std::sin(rho*(theta - theta_end) + base_angle) - 2.0 * rho * tau * std::cos(rho*(theta - theta_end) + base_angle) ));
    }
}

Vector2D FermatSpiralPath::computeSpiralSecondDerivative(double theta, double base_angle, double lambda,
                                                           double k, double rho, double theta_end) const {
    if (lambda > 0) {
        // Forward FS second derivative:
        // p_FS_ddot(θ) = -k/(4θ^(3/2))*[ (4θ^2+1)*cos(ρθ+base_angle) + 4ρθ sin(ρθ+base_angle);
        //                               (4θ^2+1)*sin(ρθ+base_angle) - 4ρθ cos(ρθ+base_angle) ]
        return Vector2D( -k/(4.0 * std::pow(theta, 3.0/2.0)) *
                         ( (4.0*theta*theta + 1.0) * std::cos(rho*theta + base_angle) + 4.0 * rho * theta * std::sin(rho*theta + base_angle) ),
                         -k/(4.0 * std::pow(theta, 3.0/2.0)) *
                         ( (4.0*theta*theta + 1.0) * std::sin(rho*theta + base_angle) - 4.0 * rho * theta * std::cos(rho*theta + base_angle) ));
    } else {
        // Mirrored FS second derivative:
        double tau = theta_end - theta;
        // p_FS_M_ddot(θ) = -k/(4τ^(3/2))*[ (4τ^2+1)*cos(ρ(θ-θ_end)+base_angle) - 4ρτ sin(ρ(θ-θ_end)+base_angle);
        //                                (4τ^2+1)*sin(ρ(θ-θ_end)+base_angle) + 4ρτ cos(ρ(θ-θ_end)+base_angle) ]
        return Vector2D( -k/(4.0 * std::pow(tau, 3.0/2.0)) *
                         ( (4.0*tau*tau + 1.0) * std::cos(rho*(theta - theta_end) + base_angle) - 4.0 * rho * tau * std::sin(rho*(theta - theta_end) + base_angle) ),
                         -k/(4.0 * std::pow(tau, 3.0/2.0)) *
                         ( (4.0*tau*tau + 1.0) * std::sin(rho*(theta - theta_end) + base_angle) + 4.0 * rho * tau * std::cos(rho*(theta - theta_end) + base_angle) ));
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

// Compute theta_end using Halley's method (eq. (48) in Sec. 4.2)
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

// Newton–Raphson for finding the closest point on an FS segment.
// Revised getClosestSpiralPoint.
// We now constrain the search to theta ∈ [theta_lower, theta_upper].
PathPoint FermatSpiralPath::getClosestSpiralPoint(const FSParameters &params, bool mirrored,
                                                  const Vector2D &vessel, double normal_angle,
                                                  double theta_lower, double theta_upper) const {
    double lambda = mirrored ? -1.0 : +1.0;
    double base_angle = mirrored ? params.chi_end : params.chi0;
    double x0 = mirrored ? params.x_end : params.x0;
    double y0 = mirrored ? params.y_end : params.y0;
    // Initial guess: the midpoint of the interval.
    double theta_val = (theta_lower + theta_upper) / 2.0;
    const double tol = 1e-4;
    const int maxIter = 20;
    for (int iter = 0; iter < maxIter; ++iter) {
        Vector2D diff = computeSpiralPoint(theta_val, base_angle, lambda, params.k, params.rho, x0, y0, params.theta_end) - vessel;
        double f_val = std::atan2(diff.y, diff.x) - normal_angle;
        while (f_val > M_PI)  f_val -= 2 * M_PI;
        while (f_val < -M_PI) f_val += 2 * M_PI;
        Vector2D dP = computeSpiralDerivative(theta_val, base_angle, lambda, params.k, params.rho, params.theta_end);
        double f_der = (dP.x * diff.y - dP.y * diff.x) / (diff.x * diff.x + diff.y * diff.y);
        double dtheta = -f_val / f_der;
        theta_val += dtheta;
        if (theta_val < theta_lower) theta_val = theta_lower;
        if (theta_val > theta_upper) theta_val = theta_upper;
        if (std::abs(dtheta) < tol)
            break;
    }
    PathPoint pp;
    pp.pos   = computeSpiralPoint(theta_val, base_angle, lambda, params.k, params.rho, x0, y0, params.theta_end);
    pp.dpos  = computeSpiralDerivative(theta_val, base_angle, lambda, params.k, params.rho, params.theta_end);
    pp.ddpos = computeSpiralSecondDerivative(theta_val, base_angle, lambda, params.k, params.rho, params.theta_end);
    return pp;
}

PathPoint FermatSpiralPath::getCompletePathPoint(const Vector2D &vessel) {
    std::vector<PathPoint> candidates;
    // Candidate from straight-line segments.
    for (size_t i = 1; i < waypoints_.size(); ++i) {
        candidates.push_back(projectOntoLine(waypoints_[i-1], waypoints_[i], vessel));
    }
    // For interior corners, consider FS transitions.
    if (waypoints_.size() >= 3) {
        for (size_t i = 1; i < waypoints_.size() - 1; ++i) {
            FSParameters params = computeFSParameters(waypoints_[i-1], waypoints_[i], waypoints_[i+1]);
            // Forward candidate: search in [0, θ_mid].
            double theta_mid = params.u_mid; // u_mid stored from computeFSParameters
            Vector2D forward_start = computeSpiralPoint(0.0, params.chi0, +1, params.k, params.rho, params.x0, params.y0, 0.0);
            double normal_angle_forward = std::atan2(vessel.y - forward_start.y, vessel.x - forward_start.x);
            PathPoint pp_forward = getClosestSpiralPoint(params, false, vessel, normal_angle_forward, 0.0, theta_mid);
            candidates.push_back(pp_forward);
            
            // Mirrored candidate: search in [θ_mid, θ_end].
            Vector2D mirrored_start = computeSpiralPoint(params.theta_end, params.chi_end, -1, params.k, params.rho, params.x_end, params.y_end, params.theta_end);
            double normal_angle_mirrored = std::atan2(vessel.y - mirrored_start.y, vessel.x - mirrored_start.x);
            PathPoint pp_mirrored = getClosestSpiralPoint(params, true, vessel, normal_angle_mirrored, theta_mid, params.theta_end);
            candidates.push_back(pp_mirrored);
        }
    }
    if (candidates.empty())
        throw std::runtime_error("No segments available for projection.");
    PathPoint best = candidates[0];
    double dmin = (best.pos - vessel).norm();
    for (const auto &pp : candidates) {
        double d = (pp.pos - vessel).norm();
        if (d < dmin) {
            dmin = d;
            best = pp;
        }
    }
    return best;
}

Waypoints FermatSpiralPath::samplePath(double delta_u) const {
    Waypoints path;
    size_t N = waypoints_.size();
    if (N < 2)
        return path;
    
    // Lambda to sample a straight line between two points.
    auto sampleLine = [&](const Vector2D &start, const Vector2D &end, int numSamples) {
        for (int j = 0; j <= numSamples; ++j) {
            double t = double(j) / numSamples;
            Vector2D pt = start + t * (end - start);
            path.push_back(pt);
        }
    };

    // If exactly two waypoints, sample a straight line.
    if (N == 2) {
        sampleLine(waypoints_.front(), waypoints_.back(), 20);
        return path;
    }
    
    // --- First corner (at waypoint[1]) ---
    FSParameters params = computeFSParameters(waypoints_[0], waypoints_[1], waypoints_[2]);
    double theta_mid = params.u_mid;  // meeting parameter
    // (A) Straight line from waypoint[0] to FS entry point (p0).
    Vector2D p0(params.x0, params.y0);
    sampleLine(waypoints_[0], p0, 10);
    // (B) Forward FS segment from θ=0 to θ=θ_mid.
    for (double theta = 0.0; theta <= theta_mid + 1e-9; theta += delta_u) {
        Vector2D pt = computeSpiralPoint(theta, params.chi0, +1, params.k, params.rho, params.x0, params.y0, 0.0);
        path.push_back(pt);
    }
    // (C) Mirrored FS segment from θ=θ_mid to θ=θ_end.
    for (double theta = theta_mid; theta <= params.theta_end + 1e-9; theta += delta_u) {
        Vector2D pt = computeSpiralPoint(theta, params.chi_end, -1, params.k, params.rho, params.x_end, params.y_end, params.theta_end);
        path.push_back(pt);
    }
    // (D) Straight line from FS exit (at θ=θ_end) to waypoint[1].
    Vector2D p_exit = computeSpiralPoint(params.theta_end, params.chi_end, -1, params.k, params.rho, params.x_end, params.y_end, params.theta_end);
    sampleLine(p_exit, waypoints_[1], 10);
    
    Vector2D prev_exit = p_exit;
    
    // --- Interior corners (i = 2 to N-2) ---
    for (size_t i = 2; i < N - 1; ++i) {
        FSParameters curr_params = computeFSParameters(waypoints_[i-1], waypoints_[i], waypoints_[i+1]);
        double curr_theta_mid = curr_params.u_mid;
        Vector2D curr_p0(curr_params.x0, curr_params.y0);
        // (A) Straight line from previous FS exit to current FS entry.
        sampleLine(prev_exit, curr_p0, 10);
        // (B) Forward FS segment for current corner: θ from 0 to θ_mid.
        for (double theta = 0.0; theta <= curr_theta_mid + 1e-9; theta += delta_u) {
            Vector2D pt = computeSpiralPoint(theta, curr_params.chi0, +1, curr_params.k, curr_params.rho, curr_params.x0, curr_params.y0, 0.0);
            path.push_back(pt);
        }
        // (C) Mirrored FS segment for current corner: θ from θ_mid to θ_end.
        for (double theta = curr_theta_mid; theta <= curr_params.theta_end + 1e-9; theta += delta_u) {
            Vector2D pt = computeSpiralPoint(theta, curr_params.chi_end, -1, curr_params.k, curr_params.rho, curr_params.x_end, curr_params.y_end, curr_params.theta_end);
            path.push_back(pt);
        }
        // (D) Straight line from current FS exit to waypoint[i].
        Vector2D curr_exit = computeSpiralPoint(curr_params.theta_end, curr_params.chi_end, -1, curr_params.k, curr_params.rho, curr_params.x_end, curr_params.y_end, curr_params.theta_end);
        sampleLine(curr_exit, waypoints_[i], 10);
        prev_exit = curr_exit;
    }
    
    // --- Final segment: straight line from last interior FS exit to final waypoint.
    sampleLine(prev_exit, waypoints_.back(), 20);
    
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
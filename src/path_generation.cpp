#include "path_generation.hpp"
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
FermatSpiralPath::FermatSpiralPath(double kappa_max)
    : kappa_max_(kappa_max)
{
    // Initialize waypoints to (0,0); they will be updated later.
    wpt_prev_ = Vector2D(0.0, 0.0);
    wpt_      = Vector2D(0.0, 0.0);
    wpt_next_ = Vector2D(0.0, 0.0);
    
    // Initialize other parameters.
    delta_chi_ = 0.0;
    rho_       = 1.0;
    chi0_      = 0.0;
    chi_end_   = 0.0;
    theta_end_ = 0.0;
    u_max_     = 0.0;
    k_         = 0.0;
    x0_ = y0_  = 0.0;
    x_end_ = y_end_ = 0.0;
}

void FermatSpiralPath::updateWaypoints(const Vector2D &wpt_prev, const Vector2D &wpt, const Vector2D &wpt_next) {
    // Update the waypoints.
    wpt_prev_ = wpt_prev;
    wpt_      = wpt;
    wpt_next_ = wpt_next;
    
    // Recompute normalized directions.
    Vector2D v_in  = (wpt_ - wpt_prev_).normalized();
    Vector2D v_out = (wpt_next_ - wpt_).normalized();
    
    // Compute course change magnitude and turning direction.
    double dotVal = v_in.dot(v_out);
    if (dotVal > 1.0) dotVal = 1.0;
    if (dotVal < -1.0) dotVal = -1.0;
    delta_chi_ = std::acos(dotVal);
    rho_ = - sign(v_in.y * v_out.x - v_in.x * v_out.y);
    
    // Set initial and final course angles.
    chi0_    = std::atan2(v_in.y, v_in.x);
    chi_end_ = std::atan2(v_out.y, v_out.x);
    
    // Domain determination: Solve θ + arctan(2θ) = |Δχ| for θ_end.
    theta_end_ = computeThetaEnd(delta_chi_);
    u_max_ = std::sqrt(theta_end_);
    
    // Scaling determination: Use the smaller of θ_end and an intrinsic max (≈0.27 rad).
    const double theta_max_intrinsic = 0.27;
    double theta_kappa_max = (theta_end_ < theta_max_intrinsic) ? theta_end_ : theta_max_intrinsic;
    k_ = computeScalingConstant(theta_kappa_max, kappa_max_);
    
    // Length calculation: Back off the waypoint along v_in by l1.
    double l1 = k_ * std::sqrt(theta_end_) * std::cos(theta_end_);
    Vector2D p0 = wpt_ - v_in * l1;
    x0_ = p0.x;
    y0_ = p0.y;

    Vector2D pend = getPoint(u_max_);
    x_end_ = pend.x;
    y_end_ = pend.y;
}

Vector2D FermatSpiralPath::getPoint(double u) const {
    double angle = rho_ * u * u + chi0_;
    return Vector2D(x0_ + k_ * u * std::cos(angle),
                    y0_ + k_ * u * std::sin(angle));
}

Vector2D FermatSpiralPath::getDerivative(double u) const {
    double angle = rho_ * u * u + chi0_;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    return Vector2D(k_ * (cosA - 2.0 * rho_ * u * u * sinA),
                    k_ * (sinA + 2.0 * rho_ * u * u * cosA));
}

Vector2D FermatSpiralPath::getSecondDerivative(double u) const {
    double angle = rho_ * u * u + chi0_;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    return Vector2D(k_ * (-6.0 * rho_ * u * sinA - 4.0 * rho_ * rho_ * u * u * u * cosA),
                    k_ * ( 6.0 * rho_ * u * cosA - 4.0 * rho_ * rho_ * u * u * u * sinA));
}

PathPoint FermatSpiralPath::getClosestPoint(const Vector2D &vessel, double normal_angle) const {
    // Find u such that the direction of (p(u) - vessel) equals normal_angle.
    double u = u_max_ / 2.0;  // initial guess
    const double tol = 1e-4;
    const int maxIter = 20;
    for (int iter = 0; iter < maxIter; ++iter) {
        Vector2D diff = getPoint(u) - vessel;
        double f_val = std::atan2(diff.y, diff.x) - normal_angle;
        // Wrap f_val to [-pi, pi]
        while (f_val > M_PI)  f_val -= 2 * M_PI;
        while (f_val < -M_PI) f_val += 2 * M_PI;
        Vector2D dP = getDerivative(u);
        double f_der = (dP.x * diff.y - dP.y * diff.x) / (diff.x * diff.x + diff.y * diff.y);
        double du = -f_val / f_der;
        u += du;
        if (std::abs(du) < tol)
            break;
    }
    PathPoint pp;
    pp.pos   = getPoint(u);
    pp.dpos  = getDerivative(u);
    pp.ddpos = getSecondDerivative(u);
    return pp;
}

std::vector<Vector2D> FermatSpiralPath::samplePath(double delta_u) const {
    std::vector<Vector2D> points;
    for (double u = 0.0; u <= u_max_; u += delta_u)
        points.push_back(getPoint(u));
    // Ensure final point is included.
    if (points.empty() || (points.back() - getPoint(u_max_)).norm() > 1e-6)
        points.push_back(getPoint(u_max_));
    return points;
}

Vector2D FermatSpiralPath::getMirroredPoint(double u) const {
    double angle = chi_end_ - rho_ * u * u;
    return Vector2D(x_end_ + k_ * u * std::cos(angle),
                    y_end_ + k_ * u * std::sin(angle));
}

Vector2D FermatSpiralPath::getMirroredDerivative(double u) const {
    double angle = chi_end_ - rho_ * u * u;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    // Differentiate: d/du [k*u*cos(chi_end - rho*u^2)] = k[cos(angle) + 2*rho*u^2*sin(angle)]
    double dx_du = k_ * (cosA + 2.0 * rho_ * u * u * sinA);
    // d/du [k*u*sin(chi_end - rho*u^2)] = k[sin(angle) - 2*rho*u^2*cos(angle)]
    double dy_du = k_ * (sinA - 2.0 * rho_ * u * u * cosA);
    return Vector2D(dx_du, dy_du);
}

Vector2D FermatSpiralPath::getMirroredSecondDerivative(double u) const {
    double angle = chi_end_ - rho_ * u * u;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    // For x: derivative of k[cos(angle) + 2ρ*u^2*sin(angle)]
    // A'(u) = -2ρ*u, so:
    // d²x/du² = k[ - sin(angle)*(-2ρ*u) + 4ρ*u*sin(angle) + 2ρ*u^2*cos(angle)*(-2ρ*u) ]
    //         = k[ 2ρ*u*sin(angle) + 4ρ*u*sin(angle) - 4ρ^2*u^3*cos(angle) ]
    //         = k[ 6ρ*u*sin(angle) - 4ρ^2*u^3*cos(angle) ]
    double d2x_du2 = k_ * (6.0 * rho_ * u * sinA - 4.0 * rho_ * rho_ * u * u * u * cosA);
    // For y: derivative of k[sin(angle) - 2ρ*u^2*cos(angle)]
    // d²y/du² = k[ cos(angle)*(-2ρ*u) - 4ρ*u*cos(angle) + 2ρ*u^2*sin(angle)*(-2ρ*u) ]
    //         = k[ -2ρ*u*cos(angle) - 4ρ*u*cos(angle) - 4ρ^2*u^3*sin(angle) ]
    //         = k[ -6ρ*u*cos(angle) - 4ρ^2*u^3*sin(angle) ]
    double d2y_du2 = k_ * (-6.0 * rho_ * u * cosA - 4.0 * rho_ * rho_ * u * u * u * sinA);
    return Vector2D(d2x_du2, d2y_du2);
}

PathPoint FermatSpiralPath::getMirroredClosestPoint(const Vector2D &vessel, double normal_angle) const {
    // Find u such that the direction of (p(u) - vessel) equals normal_angle.
    double u = u_max_ / 2.0;  // initial guess
    const double tol = 1e-4;
    const int maxIter = 20;
    for (int iter = 0; iter < maxIter; ++iter) {
        Vector2D diff = getMirroredPoint(u) - vessel;
        double f_val = std::atan2(diff.y, diff.x) - normal_angle;
        // Wrap f_val to [-pi, pi]
        while (f_val > M_PI)  f_val -= 2 * M_PI;
        while (f_val < -M_PI) f_val += 2 * M_PI;
        Vector2D dP = getMirroredDerivative(u);
        double f_der = (dP.x * diff.y - dP.y * diff.x) / (diff.x * diff.x + diff.y * diff.y);
        double du = -f_val / f_der;
        u += du;
        if (std::abs(du) < tol)
            break;
    }
    PathPoint pp;
    pp.pos   = getMirroredPoint(u);
    pp.dpos  = getMirroredDerivative(u);
    pp.ddpos = getMirroredSecondDerivative(u);
    return pp;

}

void FermatSpiralPath::printParameters() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "delta_chi = " << delta_chi_ << " rad, rho = " << rho_ << "\n";
    std::cout << "chi0 = " << chi0_ << " rad, chi_end = " << chi_end_ << " rad\n";
    std::cout << "theta_end = " << theta_end_ << " rad, u_max = " << u_max_ << "\n";
    std::cout << "Scaling constant k = " << k_ << "\n";
}

//---------------------------
// Helper Function Definitions
//---------------------------
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

// Compute scaling constant k (eq. (51) in Sec. 4.3):
// k = (1/κ_max) * sqrt(theta_kappa_max) * ((3 + 4*theta_kappa_max^2) / (1 + 4*theta_kappa_max^2)^(3/2))
double FermatSpiralPath::computeScalingConstant(double theta_kappa_max, double kappa_max) const {
    return (1.0 / kappa_max) * 2 * std::sqrt(theta_kappa_max) * 
           ((3.0 + 4.0 * theta_kappa_max * theta_kappa_max) /
           std::pow(1.0 + 4.0 * theta_kappa_max * theta_kappa_max, 1.5));
}

// Helper: Projects a point onto a line segment [A, B] and returns a PathPoint.
// For a straight line, the tangent (first derivative) is constant (B-A normalized)
// and the second derivative is zero.
PathPoint FermatSpiralPath::projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel) {
    PathPoint pp;
    Vector2D AB = B - A;
    // Compute parameter t for the projection (without extrapolation).
    double t = ((vessel - A).dot(AB)) / (AB.dot(AB));
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    pp.pos = A + AB * t;
    pp.dpos = AB.normalized(); // constant tangent along the line.
    pp.ddpos = Vector2D(0.0, 0.0);
    return pp;
}

// Function to compute the “active” point on the complete path
// (FS segments and straight-line segments) along with its derivatives,
// given the vehicle’s position.
// It assumes that the complete path is composed of:
//   Segment 1: Straight line from wpt_prev to p₀ (start of FS entering).
//   Segment 2: FS entering segment (from p₀ to p_end).
//   Segment 3: FS mirrored segment (from p_end to p_exit).
//   Segment 4: Straight line from p_exit to wpt_next.
PathPoint FermatSpiralPath::getCompletePathPoint(const Vector2D &vessel)
{
// We assume that updateWaypoints(wpt_prev, wpt, wpt_next) has been called already.

// Retrieve key points from the FS smoother:
// p0: the start point of the entering FS segment.
Vector2D p0 = getPoint(0.0);
// p_end: the end of the entering FS segment.
// (We assume the class provides an accessor for u_max, e.g., getUMax().)
double u_max = getUMax();
Vector2D p_end = getPoint(u_max);
// p_exit: the end (pull–out) point of the mirrored (exiting) FS segment.
Vector2D p_exit = getMirroredPoint(u_max); // Assuming symmetry.

// Compute the projection for each segment:

// Segment 1: Straight line from wpt_prev to p0.
PathPoint pp1 = projectOntoLine(wpt_prev_, p0, vessel);

// Segment 2: Entering FS segment.
// (We pass an initial guess for the normal angle, e.g. the angle from p0 to vessel.)
double normal_angle_FS = std::atan2(vessel.y - p0.y, vessel.x - p0.x);
PathPoint pp2 = getClosestPoint(vessel, normal_angle_FS);

// Segment 3: Mirrored (exiting) FS segment.
// (We use an initial guess based on p_end.)
double normal_angle_MIR = std::atan2(vessel.y - p_end.y, vessel.x - p_end.x);
PathPoint pp3 = getMirroredClosestPoint(vessel, normal_angle_MIR);

// Segment 4: Straight line from p_exit to wpt_next.
PathPoint pp4 = projectOntoLine(p_exit, wpt_next_, vessel);

// Compute the Euclidean distance from the vessel to each projected point.
double d1 = (pp1.pos - vessel).norm();
double d2 = (pp2.pos - vessel).norm();
double d3 = (pp3.pos - vessel).norm();
double d4 = (pp4.pos - vessel).norm();

// Choose the segment whose projection is closest to the vehicle.
PathPoint best = pp1;
double dmin = d1;
if (d2 < dmin) { dmin = d2; best = pp2; }
if (d3 < dmin) { dmin = d3; best = pp3; }
if (d4 < dmin) { dmin = d4; best = pp4; }

return best;
}

#ifndef FERMAT_SPIRAL_PATH_HPP
#define FERMAT_SPIRAL_PATH_HPP

#include <cmath>
#include <vector>

//-----------------------------
// 2D Vector Struct and Helpers
//-----------------------------
struct Vector2D {
    double x, y;
    Vector2D(double _x = 0.0, double _y = 0.0) : x(_x), y(_y) {}

    Vector2D operator+(const Vector2D &rhs) const;
    Vector2D operator-(const Vector2D &rhs) const;
    Vector2D operator*(double s) const;
    double dot(const Vector2D &rhs) const;
    double norm() const;
    Vector2D normalized() const;
};

inline Vector2D Vector2D::operator+(const Vector2D &rhs) const {
    return Vector2D(x + rhs.x, y + rhs.y);
}

inline Vector2D Vector2D::operator-(const Vector2D &rhs) const {
    return Vector2D(x - rhs.x, y - rhs.y);
}

inline Vector2D Vector2D::operator*(double s) const {
    return Vector2D(x * s, y * s);
}

inline double Vector2D::dot(const Vector2D &rhs) const {
    return x * rhs.x + y * rhs.y;
}

inline double Vector2D::norm() const {
    return std::sqrt(x * x + y * y);
}

inline Vector2D Vector2D::normalized() const {
    double n = norm();
    return (n > 1e-8) ? Vector2D(x / n, y / n) : Vector2D(0, 0);
}

// Sign helper function.
inline int sign(double x) {
    return (x >= 0.0) ? 1 : -1;
}

//----------------------------------------------
// List of waypoints as a vector of Point2D
//----------------------------------------------
using Waypoints = std::vector<Vector2D>;


//----------------------------------------------
// Structure to Return a Path Point and Derivatives
//----------------------------------------------
struct PathPoint {
    Vector2D pos;    // Position p(u)
    Vector2D dpos;   // First derivative p'(u)
    Vector2D ddpos;  // Second derivative p''(u)
};

//----------------------------------------------------------
// StraightLinePath Class Declaration
//----------------------------------------------------------
class StraightLinePath {
public:
    StraightLinePath();
    void updateWaypoints(const Vector2D &wpt_prev, const Vector2D &wpt);
    Vector2D getPoint(double u) const;
    Vector2D getDerivative(double u) const;
    Vector2D getSecondDerivative(double u) const;
    PathPoint getClosestPoint(const Vector2D &vessel, double normal_angle) const;
    std::vector<Vector2D> samplePath(double delta_u) const;
    void printParameters() const;

private:
    Vector2D wpt_prev_;
    Vector2D wpt_;
};

//----------------------------------------------------------
// FermatSpiralPath Class Declaration
// Based on the paper "Continuous-Curvature Path Generation Using Fermat's Spiral"
// by Anastasios M. Lekkas, Andreas R. Dahl, Morten Breivik, Thor I. Fossen.
// Modeling, Identification and Control, Vol. 34, No. 4, 2013, pp. 183–198, ISSN 1890–1328
//----------------------------------------------------------
class FermatSpiralPath {
public:
    // Constructor:
    //  - kappa_max: the curvature constraint (rad/m).
    FermatSpiralPath(double kappa_max);

    void updateWaypoints(const Waypoints& waypoints);

    PathPoint getCompletePathPoint(const Vector2D &vessel);

    // Samples the path segments from u = 0 to u = u_max with step delta_u.
    Waypoints samplePath(double delta_u) const;

    // For debugging: print key parameters.
    void printParameters() const;

private:
    // Input waypoints.
    Waypoints waypoints_;

    // Maximum curvature.
    double kappa_max_;
    double theta_kappa_max;

    // Structure to store FS parameters
    struct FSParameters {
        // - |Δχ|
        double delta_chi; 
        // - Turning direction (+1 for anticlockwise, -1 for clockwise)
        double rho;    
        // - Initial course angle for forward FS.
        double chi0;
        // - Final course angle for forward FS (mirrored segment uses this)
        double chi_end;
        // - Solution of θ + arctan(2θ) = |Δχ|
        double theta_end; 
        // - u_max = sqrt(theta_end)
        double u_max;    
        //Meeting point of two FS segments
        double u_mid;
        //Midpoint position of the FS segment
        Vector2D midpoint_pos;
        // - Scaling constant.
        double k;         
        // - Starting point and end point of the spiral.
        Vector2D point1, point2;
        // - Previous waypoint.
        double x0, y0;
        // - Next waypoint.
        double x_end, y_end; 
    };


    // Helper: compute FS parameters from three waypoints (A, B, C).
    FSParameters computeFSParameters(const Vector2D &A, const Vector2D &B, const Vector2D &C) const;

    // FSParameters computeFSParametersCorner(const Vector2D &point1,
    //                                         const Vector2D &B,
    //                                         const Vector2D &point2) const;

    // Compute the FS point.
    // When lambda > 0, we assume a forward FS segment and theta_end is ignored.
    // When lambda < 0, we assume a mirrored FS segment and use theta_end to define τ = theta_end - theta.
    Vector2D computeSpiralPoint(double u, double base_angle, double lambda,
                                double k, double rho, double x, double y, double u_max) const;

    // Compute the derivative of the FS point with respect to theta.
    Vector2D computeSpiralDerivative(double u, double base_angle, double lambda,
                                     double k, double rho, double u_max) const;

    // Compute the second derivative of the FS point with respect to theta.
    Vector2D computeSpiralSecondDerivative(double u, double base_angle, double lambda,
                                           double k, double rho, double u_max) const;

    PathPoint projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel);

    // Helper functions.
    double f(double theta, double delta_chi) const;
    double fprime(double theta) const;
    double fsecond(double theta) const;
    double computeThetaEnd(double delta_chi) const;
    double computeThetaMid(FSParameters params) const;
    double computeScalingConstant(double theta_kappa_max, double kappa_max) const;
    double computeDistanceForCorner(double k, double theta_end, double delta_chi) const;

    // Newton–Raphson routine to find the closest point on a spiral segment.
    // The search is limited to theta in [theta_lower, theta_upper].
    PathPoint getClosestSpiralPoint(const FSParameters &params, bool mirrored,
                                    const Vector2D &vessel, double normal_angle,
                                    double u_lower, double u_upper) const;
};

// Add this non-member function after the Vector2D class definition
inline Vector2D operator*(double s, const Vector2D& v) {
    return v * s;  // Reuse the existing Vector2D * double operator
}

double SpiralCurvature(double theta, double kappa_max);

double SpiralCurvatureDerivative(double theta, double kappa_max);

// Newton-Raphson method to find theta_kappa_max for given kappa_max
double compute_local_theta_kappa_max(double kappa_max, double initial_guess = 0.3, double tol = 1e-6, int max_iter = 20);

#endif // FERMAT_SPIRAL_PATH_HPP

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
// Structure to Return a Path Point and Derivatives
//----------------------------------------------
struct PathPoint {
    Vector2D pos;    // Position p(u)
    Vector2D dpos;   // First derivative p'(u)
    Vector2D ddpos;  // Second derivative p''(u)
};

//----------------------------------------------------------
// Path Generation Class Declaration
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

    void updateWaypoints(const Vector2D &wpt_prev, const Vector2D &wpt, const Vector2D &wpt_next);

    // Returns the point on the spiral for a given parameter u ∈ [0, u_max].
    Vector2D getPoint(double u) const;

    // Returns the first derivative with respect to u.
    Vector2D getDerivative(double u) const;

    // Returns the second derivative with respect to u.
    Vector2D getSecondDerivative(double u) const;

    PathPoint getCompletePathPoint(const Vector2D &vessel);

    // Samples the path from u = 0 to u = u_max with step delta_u.
    std::vector<Vector2D> samplePath(double delta_u) const;

    // --- Mirrored FS segment functions (for the exiting curve) ---
    // The mirrored curve (eq. (20)) is reparameterized (letting u = sqrt(theta_end - theta)):
    // p_mir(u) = [ x_end + k*u*cos(χ_end - ρ*u^2), y_end + k*u*sin(χ_end - ρ*u^2) ]
    // where (x_end, y_end) = p( u_max ) is the end of the entering curve.
    Vector2D getMirroredPoint(double u) const;
    Vector2D getMirroredDerivative(double u) const;
    Vector2D getMirroredSecondDerivative(double u) const;
    double getUMax() const { return u_max_; }


    // For debugging: print key parameters.
    void printParameters() const;

private:
    // Input waypoints.
    Vector2D wpt_prev_, wpt_, wpt_next_;
    double kappa_max_;

    // Computed course change and angles.
    double delta_chi_; // |Δχ|
    double rho_;       // turning direction (+1 for anticlockwise, -1 for clockwise)
    double chi0_;
    double chi_end_;

    // Domain and scaling parameters.
    double theta_end_; // θ_end (from solving θ + arctan(2θ) = |Δχ|)
    double u_max_;     // u_max = sqrt(theta_end_)
    double k_;         // Scaling constant (see eq. (51))

    // Starting point of the spiral.
    double x0_, y0_;

    // End point of the entering FS segment (used for the mirrored/exiting segment).
    double x_end_, y_end_; 

    // Helper functions.
    double f(double theta, double delta_chi) const;
    double fprime(double theta) const;
    double fsecond(double theta) const;
    double computeThetaEnd(double delta_chi) const;
    double computeScalingConstant(double theta_kappa_max, double kappa_max) const;

    // Given a vessel's position and a desired normal angle (radians),
    // returns the point on the path (with derivatives) closest to the vessel.
    PathPoint getClosestPoint(const Vector2D &vessel, double normal_angle) const;

    PathPoint getMirroredClosestPoint(const Vector2D &vessel, double normal_angle) const;

    PathPoint projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel);
};

// Add this non-member function after the Vector2D class definition
inline Vector2D operator*(double s, const Vector2D& v) {
    return v * s;  // Reuse the existing Vector2D * double operator
}

#endif // FERMAT_SPIRAL_PATH_HPP

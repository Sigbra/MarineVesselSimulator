#ifndef CALCULATIONS_UTILITIES_HPP
#define CALCULATIONS_UTILITIES_HPP

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

//----------------------------------------------
// 2D Vector Class
//----------------------------------------------
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

inline int sign(double x) {
    return (x >= 0.0) ? 1 : -1;
}

//----------------------------------------------
// List of waypoints as a vector of Vector2D
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

//----------------------------------------------
// Structure to Return Path Following Information
//----------------------------------------------
struct PathTrackingInfo {
    PathPoint point;    // Closest point on the path
    double x_e;         // Along-track error
    double y_e;         // Cross-track error
};

//----------------------------------------------
// Functions
//----------------------------------------------
double ssa(double angle);

double deg2rad(double degrees);

double rad2deg(double radians);

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// Skew-symmetric matrix: [a]_x
Mat3  skew3(const Vec3& a);

// Custom ZYX (yaw from North toward East) BODY→END DCM
Mat3  RnbFromEuler(double phi, double theta, double psi);

// Custom quaternion→DCM (BODY→END) matching the Euler above
Mat3  RnbFromQuatCustom(const Eigen::Quaterniond& q_in);

// Marine heading ψ from R_nb (angle from North toward East)
double yawFromRnb(const Mat3& Rnb);

// Heading from quaternion (uses custom converter)
double yawFromQuatEND(const Eigen::Quaterniond& q_in);

Eigen::Quaterniond quatFromEulerEND(double phi, double theta, double psi); // ZYX


Waypoints addIntermediateWaypoints(const Waypoints& input, double space);

#endif 
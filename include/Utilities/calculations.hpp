#ifndef CALCULATIONS_UTILITIES_HPP
#define CALCULATIONS_UTILITIES_HPP

#include <cmath>
#include <vector>
#include <Eigen/Dense>

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
// Functions
//----------------------------------------------
double ssa(double angle);

double deg2rad(double degrees);

double rad2deg(double radians);

Waypoints addIntermediateWaypoints(const Waypoints& input, double space);

#endif 
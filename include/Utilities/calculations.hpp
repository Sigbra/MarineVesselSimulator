#ifndef CALCULATIONS_UTILITIES_HPP
#define CALCULATIONS_UTILITIES_HPP

#include <cmath>
#include <vector>
#include <Eigen/Core>
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
    double   dot(const Vector2D &rhs) const;
    double   norm() const;
    Vector2D normalized() const;
};

inline Vector2D Vector2D::operator+(const Vector2D &rhs) const { return Vector2D(x + rhs.x, y + rhs.y); }
inline Vector2D Vector2D::operator-(const Vector2D &rhs) const { return Vector2D(x - rhs.x, y - rhs.y); }
inline Vector2D Vector2D::operator*(double s) const            { return Vector2D(x * s, y * s); }
inline double   Vector2D::dot(const Vector2D &rhs) const       { return x * rhs.x + y * rhs.y; }
inline double   Vector2D::norm() const                         { return std::sqrt(x * x + y * y); }
inline Vector2D Vector2D::normalized() const {
    double n = norm();
    return (n > 1e-8) ? Vector2D(x / n, y / n) : Vector2D(0, 0);
}

inline int sign(double x) { return (x >= 0.0) ? 1 : -1; }

//----------------------------------------------
// Waypoints & path helpers
//----------------------------------------------
using Waypoints = std::vector<Vector2D>;

struct PathPoint {
    Vector2D pos;    // p(u)
    Vector2D dpos;   // p'(u)
    Vector2D ddpos;  // p''(u)
};

struct PathTrackingInfo {
    PathPoint point; // closest point on path
    double x_e;      // along-track error
    double y_e;      // cross-track error
};

//----------------------------------------------
// Basic math / helpers
//----------------------------------------------
double ssa(double angle);         // wrap to (-pi, pi]
double deg2rad(double degrees);
double rad2deg(double radians);

//----------------------------------------------
// Linear algebra aliases
//----------------------------------------------
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// Skew-symmetric matrix: [a]_x
Mat3 skew3(const Vec3& a);

//----------------------------------------------
// Quaternion (no Eigen)
//----------------------------------------------
struct Quat {
    double w, x, y, z;
    Quat() : w(1.0), x(0.0), y(0.0), z(0.0) {}
    Quat(double W, double X, double Y, double Z) : w(W), x(X), y(Y), z(Z) {}
};

//----------------------------------------------
// Frames & rotations (END convention)
// BODY: x forward, y starboard, z down
// NAV (END): x East, y North, z Down
//----------------------------------------------

// ZYX Euler (φ,θ,ψ) → R_nb (BODY→NAV/END)
Mat3 RnbFromEuler(double phi, double theta, double psi);

// Quaternion (w,x,y,z) → R_nb (BODY→NAV/END), consistent with END mapping
Mat3 RnbFromQuatCustom(const Quat& q_in);

// Marine heading ψ from R_nb (angle from North→East)
// Equivalent to ψ = atan2(R(0,0), R(1,0)) under END convention.
double yawFromRnb(const Mat3& Rnb);

// Same heading extractor, named explicitly for END
double HeadingFromRnb_END(const Mat3& R_nb);

// Heading from quaternion (END)
double yawFromQuatEND(const Quat& q_in);

// Same, named explicitly for END
double HeadingFromQuat_END(const Quat& q_nb);

// Build quaternion from END ZYX Euler
Quat quatFromEulerEND(double phi, double theta, double psi);

//----------------------------------------------
// Utilities
//----------------------------------------------
Waypoints addIntermediateWaypoints(const Waypoints& input, double space);

#endif // CALCULATIONS_UTILITIES_HPP

// #ifndef CALCULATIONS_UTILITIES_HPP
// #define CALCULATIONS_UTILITIES_HPP

// #include <cmath>
// #include <vector>
// #include <Eigen/Dense>
// #include <Eigen/Core>
// #include <Eigen/Geometry>

// //----------------------------------------------
// // 2D Vector Class
// //----------------------------------------------
// struct Vector2D {
//     double x, y;
//     Vector2D(double _x = 0.0, double _y = 0.0) : x(_x), y(_y) {}

//     Vector2D operator+(const Vector2D &rhs) const;
//     Vector2D operator-(const Vector2D &rhs) const;
//     Vector2D operator*(double s) const;
//     double   dot(const Vector2D &rhs) const;
//     double   norm() const;
//     Vector2D normalized() const;
// };

// inline Vector2D Vector2D::operator+(const Vector2D &rhs) const {
//     return Vector2D(x + rhs.x, y + rhs.y);
// }
// inline Vector2D Vector2D::operator-(const Vector2D &rhs) const {
//     return Vector2D(x - rhs.x, y - rhs.y);
// }
// inline Vector2D Vector2D::operator*(double s) const {
//     return Vector2D(x * s, y * s);
// }
// inline double Vector2D::dot(const Vector2D &rhs) const {
//     return x * rhs.x + y * rhs.y;
// }
// inline double Vector2D::norm() const {
//     return std::sqrt(x * x + y * y);
// }
// inline Vector2D Vector2D::normalized() const {
//     double n = norm();
//     return (n > 1e-8) ? Vector2D(x / n, y / n) : Vector2D(0, 0);
// }

// inline int sign(double x) {
//     return (x >= 0.0) ? 1 : -1;
// }

// //----------------------------------------------
// // Waypoints & path helpers
// //----------------------------------------------
// using Waypoints = std::vector<Vector2D>;

// struct PathPoint {
//     Vector2D pos;    // p(u)
//     Vector2D dpos;   // p'(u)
//     Vector2D ddpos;  // p''(u)
// };

// struct PathTrackingInfo {
//     PathPoint point; // closest point on path
//     double x_e;      // along-track error
//     double y_e;      // cross-track error
// };

// //----------------------------------------------
// // Basic math / helpers
// //----------------------------------------------
// double ssa(double angle);         // wrap to (-pi, pi]
// double deg2rad(double degrees);
// double rad2deg(double radians);

// //----------------------------------------------
// // Linear algebra aliases
// //----------------------------------------------
// using Vec3 = Eigen::Vector3d;
// using Mat3 = Eigen::Matrix3d;

// // Skew-symmetric matrix: [a]_x
// Mat3 skew3(const Vec3& a);

// //----------------------------------------------
// // Frames & rotations (END convention)
// // BODY: x forward, y starboard, z down
// // NAV (END): x East, y North, z Down
// //----------------------------------------------

// // ZYX Euler (φ,θ,ψ) → R_nb (BODY→NAV/END)
// Mat3 RnbFromEuler(double phi, double theta, double psi);

// // Quaternion (w,x,y,z) → R_nb (BODY→NAV/END), consistent with END mapping
// Mat3 RnbFromQuatCustom(const Eigen::Quaterniond& q_in);

// // Marine heading ψ from R_nb (angle from North→East)
// // Equivalent to ψ = atan2(R(0,0), R(1,0)) under END convention.
// double yawFromRnb(const Mat3& Rnb);

// // Same heading extractor, named explicitly for END
// double HeadingFromRnb_END(const Mat3& R_nb);

// // Heading from quaternion (END)
// double yawFromQuatEND(const Eigen::Quaterniond& q_in);

// // Same, named explicitly for END
// double HeadingFromQuat_END(const Eigen::Quaterniond& q_nb);

// // Build quaternion from END ZYX Euler
// Eigen::Quaterniond quatFromEulerEND(double phi, double theta, double psi);

// //----------------------------------------------
// // Utilities
// //----------------------------------------------
// Waypoints addIntermediateWaypoints(const Waypoints& input, double space);

// #endif // CALCULATIONS_UTILITIES_HPP

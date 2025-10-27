#include "Utilities/calculations.hpp"
#include <cmath>
#include <iostream>

double ssa(double angle)
{
    if (std::abs(angle) > 2*M_PI) {
        std::cerr << "Warning: Unusually large angle value detected: " << angle << std::endl;
        angle = std::fmod(angle, 2 * M_PI); 
    }
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;
    return angle;
}

double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}

Mat3 skew3(const Vec3& a) {
  Mat3 S;
  S <<    0,   -a.z(),  a.y(),
        a.z(),    0,   -a.x(),
       -a.y(),  a.x(),    0;
  return S;
}

Mat3 RnbFromEuler(double phi, double theta, double psi) {
  const double cphi = std::cos(phi), sphi = std::sin(phi);
  const double cth  = std::cos(theta), sth = std::sin(theta);
  const double cpsi = std::cos(psi),   spsi = std::sin(psi);

  Mat3 R;
  // Row 1: East
  R(0,0) =  spsi * cth;
  R(0,1) =  cpsi * cphi + sphi * sth * spsi;
  R(0,2) = -cpsi * sphi + sth * spsi * cphi;

  // Row 2: North
  R(1,0) =  cpsi * cth;
  R(1,1) = -spsi * cphi + cpsi * sth * sphi;
  R(1,2) =  spsi * sphi + cpsi * cphi * sth;

  // Row 3: Down
  R(2,0) = -sth;
  R(2,1) =  cth * sphi;
  R(2,2) =  cth * cphi;

  return R;
}

Mat3 RnbFromQuatCustom(const Eigen::Quaterniond& q_in) {
  const Eigen::Quaterniond q = q_in.normalized();
  const double w = q.w(), x = q.x(), y = q.y(), z = q.z();
  const double xx = x*x, yy = y*y, zz = z*z;
  const double wx = w*x, wy = w*y, wz = w*z;
  const double xy = x*y, xz = x*z, yz = y*z;

  Mat3 R;
  // Row 1: East  (standard Y row)
  R(0,0) = 2.0*(xy + wz);
  R(0,1) = 1.0 - 2.0*(xx + zz);
  R(0,2) = 2.0*(yz - wx);

  // Row 2: North (standard X row)
  R(1,0) = 1.0 - 2.0*(yy + zz);
  R(1,1) = 2.0*(xy - wz);
  R(1,2) = 2.0*(xz + wy);

  // Row 3: Down  (standard Z row)
  R(2,0) = 2.0*(xz - wy);
  R(2,1) = 2.0*(yz + wx);
  R(2,2) = 1.0 - 2.0*(xx + yy);

  return R;
}

double yawFromRnb(const Mat3& R) {
  // ψ = atan2(E•x_B, N•x_B) = atan2(R(0,0), R(1,0))
  return std::atan2(R(0,0), R(1,0));
}

double yawFromQuatEND(const Eigen::Quaterniond& q_in) {
  return yawFromRnb(RnbFromQuatCustom(q_in));
}

Eigen::Quaterniond quatFromEulerEND(double phi, double theta, double psi) {
  const double c1 = std::cos(phi*0.5),   s1 = std::sin(phi*0.5);    // roll
  const double c2 = std::cos(theta*0.5), s2 = std::sin(theta*0.5);  // pitch
  const double c3 = std::cos(psi*0.5),   s3 = std::sin(psi*0.5);    // yaw
  // standard ZYX quaternion (w,x,y,z)
  const double w = c1*c2*c3 + s1*s2*s3;
  const double x = s1*c2*c3 - c1*s2*s3;
  const double y = c1*s2*c3 + s1*c2*s3;
  const double z = c1*c2*s3 - s1*s2*c3;
  return Eigen::Quaterniond(w,x,y,z).normalized();
}

Waypoints addIntermediateWaypoints(const Waypoints& input, double space) {
    Waypoints output;
    
    // Check if there are any waypoints to process
    if (input.empty())
        return output;
    
    // Always include the first waypoint
    output.push_back(input[0]);
    
    // Process each pair of consecutive waypoints
    for (size_t i = 1; i < input.size(); ++i) {
        double x1 = input[i-1].x;
        double y1 = input[i-1].y;
        double x2 = input[i].x;
        double y2 = input[i].y;
        
        // Compute the Euclidean distance between the two waypoints
        double dist = std::hypot(x2 - x1, y2 - y1);
        
        // If the distance exceeds the spacing, add intermediate waypoints
        if (dist > space) {
            // Determine the number of segments required
            int num_segments = static_cast<int>(std::ceil(dist / space));
            // Insert intermediate waypoints along the line
            for (int seg = 1; seg < num_segments; ++seg) {
                double t = static_cast<double>(seg) / num_segments;
                double new_x = x1 + t * (x2 - x1);
                double new_y = y1 + t * (y2 - y1);
                output.push_back({new_x, new_y});
            }
        }
        
        // Add the original waypoint
        output.push_back(input[i]);
    }
    
    return output;
}
#include "Utilities/calculations.hpp"
#include <cmath>
#include <iostream>

// ---------- Your existing non-quat utilities (unchanged) ----------
double ssa(double angle)
{
    if (std::abs(angle) > 2*M_PI) {
        //std::cerr << "Warning: Unusually large angle value detected: " << angle << std::endl;
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
  S <<     0,   -a.z(),  a.y(),
        a.z(),      0,  -a.x(),
       -a.y(),   a.x(),     0;
  return S;
}

// ---------- Minimal quaternion (no Eigen) ----------

// Robust normalize; falls back to identity if norm is (near) zero.
static inline Quat q_normalized_safe(Quat q){
  const double n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
  if (n2 > 0.0) {
    const double invn = 1.0 / std::sqrt(n2);
    q.w *= invn; q.x *= invn; q.y *= invn; q.z *= invn;
  } else {
    q = Quat(); // identity
  }
  return q;
}

// Deterministic sign canonicalization (same rule you had: tie-break on z when w≈0)
static inline Quat q_canonical(Quat q){
  q = q_normalized_safe(q);
  if (q.w < 0.0 || (std::abs(q.w) <= 1e-12 && q.z < 0.0)) {
    q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z;
  }
  return q;
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

// ---------- Quaternion ↔ rotation using your END mapping ----------
// This matches your earlier R (Rzyx with END rows: East, North, Down) and is sign-agnostic.
Mat3 RnbFromQuatCustom(const Quat& q_in) {
  const Quat q = q_canonical(q_in); // normalized + deterministic sign
  const double w = q.w, x = q.x, y = q.y, z = q.z;

  const double xx = x*x, yy = y*y, zz = z*z;
  const double wx = w*x, wy = w*y, wz = w*z;
  const double xy = x*y, xz = x*z, yz = y*z;

  Mat3 R;
  // Row 1: East
  R(0,0) = 2.0*(xy + wz);
  R(0,1) = 1.0 - 2.0*(xx + zz);
  R(0,2) = 2.0*(yz - wx);

  // Row 2: North
  R(1,0) = 1.0 - 2.0*(yy + zz);
  R(1,1) = 2.0*(xy - wz);
  R(1,2) = 2.0*(xz + wy);

  // Row 3: Down
  R(2,0) = 2.0*(xz - wy);
  R(2,1) = 2.0*(yz + wx);
  R(2,2) = 1.0 - 2.0*(xx + yy);

  return R;
}

// Single, branch-free heading extractor for END convention.
// ψ = atan2(R(0,0), R(1,0))
double HeadingFromRnb_END(const Mat3& R_nb)
{
  return std::atan2(R_nb(0,0), R_nb(1,0));
}

// “Heading from quat” helper delegates to the two above.
double HeadingFromQuat_END(const Quat& q_nb)
{
  const Mat3 R_nb = RnbFromQuatCustom(q_canonical(q_nb));
  return HeadingFromRnb_END(R_nb);
}

// Same as your yawFromRnb (kept for API parity)
double yawFromRnb(const Mat3& R) {
  // ψ = atan2(E•x_B, N•x_B) = atan2(R(0,0), R(1,0))
  return std::atan2(R(0,0), R(1,0));
}

double yawFromQuatEND(const Quat& q_in) {
  return yawFromRnb(RnbFromQuatCustom(q_in));
}

// ZYX (roll=φ, pitch=θ, yaw=ψ) → quaternion (w,x,y,z) under your END/Rzyx convention.
Quat quatFromEulerEND(double phi, double theta, double psi) {
  const double c1 = std::cos(0.5*phi),   s1 = std::sin(0.5*phi);    // roll
  const double c2 = std::cos(0.5*theta), s2 = std::sin(0.5*theta);  // pitch
  const double c3 = std::cos(0.5*psi),   s3 = std::sin(0.5*psi);    // yaw

  // Standard Hamilton (w,x,y,z) for ZYX composition
  const double w = c1*c2*c3 + s1*s2*s3;
  const double x = s1*c2*c3 - c1*s2*s3;
  const double y = c1*s2*c3 + s1*c2*s3;
  const double z = c1*c2*s3 - s1*s2*c3;

  return q_normalized_safe(Quat(w,x,y,z));
}

// ---------- Waypoint densification (unchanged) ----------
Waypoints addIntermediateWaypoints(const Waypoints& input, double space) {
    Waypoints output;

    if (input.empty())
        return output;

    output.push_back(input[0]);

    for (size_t i = 1; i < input.size(); ++i) {
        double x1 = input[i-1].x;
        double y1 = input[i-1].y;
        double x2 = input[i].x;
        double y2 = input[i].y;

        double dist = std::hypot(x2 - x1, y2 - y1);

        if (dist > space) {
            int num_segments = static_cast<int>(std::ceil(dist / space));
            for (int seg = 1; seg < num_segments; ++seg) {
                double t = static_cast<double>(seg) / num_segments;
                double new_x = x1 + t * (x2 - x1);
                double new_y = y1 + t * (y2 - y1);
                output.push_back({new_x, new_y});
            }
        }

        output.push_back(input[i]);
    }

    return output;
}
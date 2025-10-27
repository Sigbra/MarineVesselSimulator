#ifndef EKF13_HPP
#define EKF13_HPP

#pragma once
#include <Eigen/Dense>
#include <optional>
#include <functional>
#include "Models/model_utilities.hpp"  // Rzyx(phi,theta,psi), Tzyx(phi,theta), ssa()

// -----------------------------------------------------------------------------
// EKF13 (planar) — Frames & conventions
// Global: END  (x=East, y=North, z=Down)
// Body:   x forward, y starboard, z down
// Yaw ψ:  0 at North, positive clockwise (about +Down)
// States: x = [pE, pN, vE, vN, phi, theta, psi, bgx, bgy, bgz, bax, bay, baz]^T
// Inputs: u = IMU (body z-down): [ax, ay, az, wx, wy, wz]
// Meas:   y = [pE, pN, psi]  (same yaw convention)
// -----------------------------------------------------------------------------
class EKF13 {
public:
  using Mat = Eigen::MatrixXd;
  using Vec = Eigen::VectorXd;

  static constexpr int NX = 13; // [pE,pN,vE,vN, phi,theta,psi, bgx,bgy,bgz, bax,bay,baz]
  static constexpr int NU = 6;  // [ax,ay,az, wx,wy,wz] (body z-down)
  static constexpr int NY = 3;  // [pE,pN,psi]
  static constexpr int NW = 12; // process/IMU noise channels

  struct Input {
    double ax, ay, az; // accelerometer (m/s^2), body z-down
    double wx, wy, wz; // gyro (rad/s),       body z-down
  };
  struct GnssMeas {
    double px, py, psi; // END: px=East (m), py=North (m), psi: 0@North +CW
  };

  // dt_fast_sec: IMU period, dt_slow_sec: GNSS period, g: gravity magnitude
  EKF13(double dt_fast_sec = 0.01, double dt_slow_sec = 1.0, double g = 9.81);

  // Configuration
  void setState(const Eigen::Matrix<double,NX,1>& x0);
  void setCovariance(const Eigen::Matrix<double,NX,NX>& P0);
  void setProcessNoise(const Eigen::Matrix<double,NW,NW>& Qd);
  void setMeasNoise(const Eigen::Matrix<double,NY,NY>& Rd);
  void setGravity(double g);

  // Access
  const Eigen::Matrix<double,NX,1>&        state()      const;
  const Eigen::Matrix<double,NX,NX>&       covariance() const;

  // Export in EKF18's 12-state layout; fields not estimated here are set to 0:
  // [u v w p q r  x y z  phi theta psi]^T   (END positions, body rates/vels zero)
  Eigen::VectorXd getState12() const;

  // One filter step at the fast rate; pass GNSS at slow ticks (optional)
  void step(const Input& u, const std::optional<GnssMeas>& gnss);

private:
  // Continuous-time dynamics xdot = f(x,u,w) in END/body z-down/yaw 0@N +CW
  Eigen::Matrix<double,NX,1> f(const Eigen::Matrix<double,NX,1>& x,
                               const Input& u,
                               const Eigen::Matrix<double,NW,1>& w) const;

  // Centered-difference numerical Jacobian utility
  Eigen::MatrixXd numericalJacobian(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& Ffun,
    const Eigen::VectorXd& x,
    double eps) const;

  // EKF internals
  void predictor(const Input& u);
  void corrector(const GnssMeas& y);

private:
  // Timing / constants
  double h_;   // fast sample time (s)
  int    Z_;   // slow/fast ratio (e.g., 1.0/h for 1 Hz GNSS)
  int    k_;   // fast step counter
  double g_;   // gravity magnitude (m/s^2), +g along +D in END

  // Filter state/covariance
  Eigen::Matrix<double,NX,1>  x_;
  Eigen::Matrix<double,NX,NX> P_;

  // Noise (discrete)
  Eigen::Matrix<double,NW,NW> Qd_; // process noise (IMU + bias RWs)
  Eigen::Matrix<double,NY,NY> Rd_; // measurement noise [pE,pN,psi]
};

#endif // EKF13_HPP

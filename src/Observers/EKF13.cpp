#include "Observers/EKF13.hpp"
#include <cmath>
#include <Eigen/Dense>
#include "Models/model_utilities.hpp"  
#include "Utilities/calculations.hpp"

// --------------------------- ctor / config ---------------------------
EKF13::EKF13(double dt_fast_sec, double dt_slow_sec, double g)
: h_(dt_fast_sec),
  Z_(static_cast<int>(std::lround(dt_slow_sec / dt_fast_sec))),
  k_(0),
  g_(g)
{
  x_.setZero();
  P_.setIdentity(NX, NX);
  Qd_.setIdentity(NW, NW);
  Rd_.setIdentity(NY, NY);

  // State covariance (tune as needed)
  P_.diagonal() <<
    5,5,                         // px, py  (m^2)
    1,1,                         // vx, vy  (m^2/s^2)
    std::pow(5.0*M_PI/180.0,2),  // phi
    std::pow(5.0*M_PI/180.0,2),  // theta
    std::pow(5.0*M_PI/180.0,2),  // psi  (0 at North, +CW)
    std::pow(0.5*M_PI/180.0,2),  // bgx
    std::pow(0.5*M_PI/180.0,2),  // bgy
    std::pow(0.5*M_PI/180.0,2),  // bgz
    0.01, 0.01, 0.01;            // bax, bay, baz  ((m/s^2)^2)

  // Process noise (IMU white + bias random walks)
  Qd_.setZero();
  // accel white noise ((m/s^2)^2)
  Qd_(0,0) = Qd_(1,1) = Qd_(2,2) = std::pow(0.15,2);
  // gyro white noise ((rad/s)^2)
  Qd_(3,3) = Qd_(4,4) = Qd_(5,5) = std::pow(2.0*M_PI/180.0,2);
  // gyro bias RW ((rad/s)^2)
  Qd_(6,6) = Qd_(7,7) = Qd_(8,8) = std::pow(0.01*M_PI/180.0,2);
  // accel bias RW ((m/s^2)^2)
  Qd_(9,9)   = std::pow(0.01,2);
  Qd_(10,10) = std::pow(0.01,2);
  Qd_(11,11) = std::pow(0.01,2);

  // Measurement noise
  Rd_.setZero();
  Rd_(0,0) = Rd_(1,1) = std::pow(0.7,2);            // GNSS pos (m^2)
  Rd_(2,2) = std::pow(1.0*M_PI/180.0,2);            // heading (rad^2), 0@North +CW
}

void EKF13::setState(const Eigen::Matrix<double,NX,1>& x0) { x_ = x0; }
void EKF13::setCovariance(const Eigen::Matrix<double,NX,NX>& P0) { P_ = P0; }
void EKF13::setProcessNoise(const Eigen::Matrix<double,NW,NW>& Qd) { Qd_ = Qd; }
void EKF13::setMeasNoise(const Eigen::Matrix<double,NY,NY>& Rd) { Rd_ = Rd; }
void EKF13::setGravity(double g) { g_ = g; }

const Eigen::Matrix<double,EKF13::NX,1>& EKF13::state() const { return x_; }
const Eigen::Matrix<double,EKF13::NX,EKF13::NX>& EKF13::covariance() const { return P_; }

// ------------------------------ step --------------------------------
void EKF13::step(const Input& u, const std::optional<GnssMeas>& gnss) {
  // Corrector (multi-rate): only every Z_ steps, if a measurement is present
  const bool is_gnss_step = (k_ % Z_ == 0);
  if (is_gnss_step && gnss.has_value()) {
    corrector(gnss.value());
  }
  // Predictor (always)
  predictor(u);
  ++k_;
}

// ----------------------------- dynamics ------------------------------
// END global (x=East, y=North, z=Down), body x fwd, y starboard, z down.
// Yaw convention: psi = 0 at North, positive clockwise (about +Down).
Eigen::Matrix<double,EKF13::NX,1>
EKF13::f(const Eigen::Matrix<double,NX,1>& x,
         const Input& u,
         const Eigen::Matrix<double,NW,1>& w) const
{
  // State unpack
  const double px=x(0), py=x(1), vx=x(2), vy=x(3);
  const double phi=x(4), th=x(5), psi=x(6);     // psi is 0@North, +CW
  const double bgx=x(7), bgy=x(8), bgz=x(9);
  const double bax=x(10),bay=x(11),baz=x(12);

  // IMU (body, z-down) + noise
  const Eigen::Vector3d a_b(u.ax + w(0), u.ay + w(1), u.az + w(2));
  const Eigen::Vector3d w_b(u.wx + w(3), u.wy + w(4), u.wz + w(5));

  // Bias-corrected signals (still body z-down)
  const Eigen::Vector3d a_corr = a_b - Eigen::Vector3d(bax,bay,baz);
  const Eigen::Vector3d w_corr = w_b - Eigen::Vector3d(bgx,bgy,bgz);

  // Attitude kinematics (ZYX) with your convention
  const Eigen::Vector3d euler_dot = Tzyx(phi, th) * w_corr;

  // Body -> END using your Rzyx (consistent with psi convention)
  const Eigen::Matrix3d R_end = Rzyx(phi, th, psi);  // body -> END

  // Translational dynamics in END
  Eigen::Vector3d vdot_end = R_end * a_corr;

  // Gravity in END: +g along +D
  vdot_end += Eigen::Vector3d(0.0, 0.0, +g_);

  Eigen::Matrix<double,NX,1> xd; xd.setZero();
  xd(0) = vx;              // ṗ_E
  xd(1) = vy;              // ṗ_N
  xd(2) = vdot_end(0);     // v̇_E
  xd(3) = vdot_end(1);     // v̇_N
  xd(4) = euler_dot(0);    // φ̇
  xd(5) = euler_dot(1);    // θ̇
  xd(6) = euler_dot(2);    // ψ̇  (matches 0@North +CW convention)

  // Bias random walks (driven by w(6..11))
  xd(7)  = w(6);
  xd(8)  = w(7);
  xd(9)  = w(8);
  xd(10) = w(9);
  xd(11) = w(10);
  xd(12) = w(11);
  return xd;
}

// ------------------------ numerical Jacobian ------------------------
Eigen::MatrixXd
EKF13::numericalJacobian(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& Ffun,
  const Eigen::VectorXd& x,
  double eps) const
{
  const int n = static_cast<int>(x.size());
  Eigen::VectorXd f0 = Ffun(x);
  Eigen::MatrixXd J(f0.size(), n);
  for (int i=0;i<n;++i) {
    Eigen::VectorXd xp = x; xp(i) += eps;
    Eigen::VectorXd xm = x; xm(i) -= eps;
    Eigen::VectorXd fp = Ffun(xp);
    Eigen::VectorXd fm = Ffun(xm);
    J.col(i) = (fp - fm) / (2.0*eps);
  }
  return J;
}

// ------------------------------- predict ----------------------------
void EKF13::predictor(const Input& u) {
  // State propagate (Euler)
  Eigen::Matrix<double,NW,1> w_zero; 
  w_zero.setZero();
  const Eigen::Matrix<double,NX,1> xd = f(x_, u, w_zero);
  x_ = x_ + h_ * xd;

  // Build Ad = I + h * (∂f/∂x), Ed = h * (∂f/∂w)
  auto fx = [&](const Eigen::VectorXd& xv) -> Eigen::VectorXd {
    Eigen::Matrix<double,NX,1> xloc = xv;
    return f(xloc, u, Eigen::Matrix<double,NW,1>::Zero());
  };
  auto fw = [&](const Eigen::VectorXd& wv) -> Eigen::VectorXd {
    Eigen::Matrix<double,NW,1> wloc = wv;
    return f(x_, u, wloc);
  };

  const Eigen::Matrix<double,NX,NX> A_d =
      Eigen::Matrix<double,NX,NX>::Identity()
    + h_ * numericalJacobian(fx, x_.cast<double>(), 1e-6).cast<double>();

  const Eigen::Matrix<double,NX,NW> E_d =
      h_ * numericalJacobian(fw, Eigen::VectorXd::Zero(NW), 1e-6).cast<double>();

  // Covariance predict
  P_ = A_d * P_ * A_d.transpose() + E_d * Qd_ * E_d.transpose();
}


// ------------------------------- correct ----------------------------
void EKF13::corrector(const GnssMeas& y) {
  // Measurement model h(x) = [p_E, p_N, psi] with psi in your convention (0@North, +CW)
  Eigen::Matrix<double,NY,1> yhat; yhat << x_(0), x_(1), x_(6);

  Eigen::Matrix<double,NY,1> z;
  z << y.px, y.py, y.psi;

  // Innovation (use smallest-signed-angle for yaw residual only)
  Eigen::Matrix<double,NY,1> nu = z - yhat;
  nu(2) = ssa(nu(2));  // no wrap elsewhere

  Eigen::Matrix<double,NY,NX> C_d; C_d.setZero();
  C_d(0,0) = 1.0; // p_E
  C_d(1,1) = 1.0; // p_N
  C_d(2,6) = 1.0; // psi

  const Eigen::Matrix<double,NY,NY> S = C_d * P_ * C_d.transpose() + Rd_;
  const Eigen::Matrix<double,NX,NY> K = P_ * C_d.transpose() * S.inverse();

  x_ = x_ + K * nu;

  // Joseph-form covariance update (numerically robust)
  const Eigen::Matrix<double,NX,NX> I  = Eigen::Matrix<double,NX,NX>::Identity();
  const Eigen::Matrix<double,NX,NX> IKC = (I - K * C_d);
  P_ = IKC * P_ * IKC.transpose() + K * Rd_ * K.transpose();
}

// ------------------------------- export -----------------------------
Eigen::VectorXd EKF13::getState12() const {
  // Return EKF18-compatible 12-vector:
  // [u v w p q r  x y z  phi theta psi]^T
  // EKF13 does not estimate u,v,w or p,q,r or z -> set to zero.
  Eigen::VectorXd z(12);
  z.setZero();

  // positions (END)
  z(6)  = x_(0);   // x = East
  z(7)  = x_(1);   // y = North
  z(8)  = 0.0;     // z = Down (planar model)

  // attitudes (psi uses 0@North, +CW)
  z(9)  = x_(4);   // phi
  z(10) = x_(5);   // theta
  z(11) = x_(6);   // psi

  return z;
}



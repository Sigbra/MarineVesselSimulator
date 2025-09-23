#include "Observers/kalmanObserver.hpp"
#include <Eigen/Dense>
#include "Models/model_utilities.hpp"
#include <cmath>

// ---------- Helpers ----------
static inline double wrap_to_pi(double a){
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a <= -M_PI) a += 2.0*M_PI;
    return a;
}

// ----------------- Constructor -----------------
KalmanFilter15::KalmanFilter15(double dt)
: dt_(dt)
{
    x_ = Eigen::VectorXd::Zero(15);
    P_ = Eigen::MatrixXd::Identity(15,15) * 0.1;

    // State transition Jacobian (will be rebuilt each predict)
    F_ = Eigen::MatrixXd::Identity(15,15);

    // Process noise (block-wise, scaled by dt)
    Q_ = Eigen::MatrixXd::Zero(15,15);
    const double q_vel  = 5e-3;   // m^2/s^3   (vel process)
    const double q_pos  = 1e-4;   // m^2/s     (pos process)
    const double q_ang  = 5e-5;   // rad^2/s   (angles)
    const double q_bias = 1e-8;   // (gyro bias)

    Q_.block<3,3>(0,0)    = q_vel  * dt_ * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(6,6)    = q_pos  * dt_ * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(9,9)    = q_ang  * dt_ * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(12,12)  = q_bias * dt_ * Eigen::Matrix3d::Identity();

    // Measurement matrix (GNSS: xN,yN,zN + heading)
    H_ = Eigen::MatrixXd::Zero(4,15);
    H_.block<3,3>(0,6) = Eigen::Matrix3d::Identity();  // positions
    H_(3,11) = 1.0;                                    // heading (psi)

    // Measurement noise (match your simulator settings as close as possible)
    // If you use raw_GNSS sigma_pos = 0.02, set R_pos = 0.02^2
    const double sigma_pos = 0.02;  // m
    const double sigma_psi = 0.01;  // rad (tune based on antenna baseline)
    R_ = Eigen::MatrixXd::Identity(4,4);
    R_.block<3,3>(0,0) *= sigma_pos * sigma_pos;
    R_(3,3) = sigma_psi * sigma_psi;
}

// ----------------- Helper: Euler rates -----------------
Eigen::Vector3d KalmanFilter15::eulerRateFromBodyRates(const Eigen::Vector3d &bodyRates,
                                                       const Eigen::Vector3d &angles)
{
    const double phi = angles(0), theta = angles(1);
    const double sphi = std::sin(phi), cphi = std::cos(phi);
    const double tth  = std::tan(theta), cth = std::cos(theta);

    Eigen::Matrix3d T;
    T << 1.0, sphi*tth,   cphi*tth,
         0.0, cphi,      -sphi,
         0.0, sphi/cth,   cphi/cth;

    return T * bodyRates;
}

// ----------------- Predict -----------------
void KalmanFilter15::predict(const Eigen::Vector3d &imu_accel,
                             const Eigen::Vector3d &imu_gyro)
{
    // ----- angles (integrate using bias-corrected gyro) -----
    const Eigen::Vector3d bgyro = x_.segment<3>(12);
    const Eigen::Vector3d gyro_corrected = imu_gyro - bgyro;

    // Euler rates
    Eigen::Vector3d euler_dot = eulerRateFromBodyRates(gyro_corrected, x_.segment<3>(9));

    // Integrate & wrap
    x_(9)  = wrap_to_pi(x_(9)  + euler_dot(0) * dt_);
    x_(10) = wrap_to_pi(x_(10) + euler_dot(1) * dt_);
    x_(11) = wrap_to_pi(x_(11) + euler_dot(2) * dt_);

    // ----- rotation with UPDATED angles -----
    const double phi   = x_(9);
    const double theta = x_(10);
    const double psi   = x_(11);

    const double cphi = std::cos(phi),  sphi = std::sin(phi);
    const double cth  = std::cos(theta), sth  = std::sin(theta);
    const double cpsi = std::cos(psi),  spsi = std::sin(psi);

    Eigen::Matrix3d R_b2n;
    R_b2n <<  cth*cpsi,  sphi*sth*cpsi - cphi*spsi,  cphi*sth*cpsi + sphi*spsi,
              cth*spsi,  sphi*sth*spsi + cphi*cpsi,  cphi*sth*spsi - sphi*cpsi,
             -sth,       sphi*cth,                    cphi*cth;

    // ----- kinematics for position & velocity -----
    const Eigen::Vector3d v_body = x_.segment<3>(0);

    // position update in NED
    x_.segment<3>(6) += R_b2n * v_body * dt_;

    // velocity update in BODY (imu_accel is already gravity-compensated upstream)
    x_.segment<3>(0) += imu_accel * dt_;

    // ----- build state Jacobian F (key fix) -----
    F_.setIdentity();

    // (1) pos depends on body velocity: ∂pos/∂v = R_b2n * dt
    F_.block<3,3>(6,0) = R_b2n * dt_;

    // (2) pos depends on angles via R(angles)*v : ∂pos/∂angles
    const double eps = 1e-6;
    const Eigen::Matrix3d Rphi = Rzyx(phi + eps, theta,      psi);
    const Eigen::Matrix3d Rth  = Rzyx(phi,       theta+eps,  psi);
    const Eigen::Matrix3d Rpsi = Rzyx(phi,       theta,      psi+eps);

    const Eigen::Vector3d dpos_dphi   = ((Rphi - R_b2n) / eps) * v_body * dt_;
    const Eigen::Vector3d dpos_dtheta = ((Rth  - R_b2n) / eps) * v_body * dt_;
    const Eigen::Vector3d dpos_dpsi   = ((Rpsi - R_b2n) / eps) * v_body * dt_;

    F_(6,9)  = dpos_dphi(0);   F_(6,10) = dpos_dtheta(0);   F_(6,11) = dpos_dpsi(0);
    F_(7,9)  = dpos_dphi(1);   F_(7,10) = dpos_dtheta(1);   F_(7,11) = dpos_dpsi(1);
    F_(8,9)  = dpos_dphi(2);   F_(8,10) = dpos_dtheta(2);   F_(8,11) = dpos_dpsi(2);

    // (3) angles depend on gyro biases: ∂angles/∂bgyro = -T * dt
    // T(φ,θ) mapping body rates -> euler rates
    Eigen::Matrix3d T;
    {
        const double tth = std::tan(theta);
        T << 1.0, sphi*tth,   cphi*tth,
             0.0, cphi,      -sphi,
             0.0, sphi/cth,   cphi/cth;
    }
    F_.block<3,3>(9,12) = -T * dt_;

    // ----- covariance propagation -----
    P_ = F_ * P_ * F_.transpose() + Q_;
    P_ = 0.5 * (P_ + P_.transpose());  // keep symmetry
}


// ----------------- Update (pos + heading) -----------------
void KalmanFilter15::update(const Eigen::Vector3d &gnss_pos,
                            double gnss_heading)
{
    Eigen::VectorXd z(4);
    z.segment<3>(0) = gnss_pos;
    z(3) = gnss_heading;

    // Innovation
    Eigen::VectorXd y = z - H_ * x_;
    y(3) = wrap_to_pi(y(3)); // wrap heading residual

    // Kalman gain
    const Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    const Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    // Update
    x_ += K * y;
    x_(9)  = wrap_to_pi(x_(9));
    x_(10) = wrap_to_pi(x_(10));
    x_(11) = wrap_to_pi(x_(11));

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(15,15);
    P_ = (I - K * H_) * P_;
    P_ = 0.5 * (P_ + P_.transpose());
}

// ----------------- Update (pos only) -----------------
void KalmanFilter15::update_pos_only(const Eigen::Vector3d &gnss_pos)
{
    Eigen::Matrix<double,3,15> Hpos = Eigen::Matrix<double,3,15>::Zero();
    Hpos.block<3,3>(0,6) = Eigen::Matrix3d::Identity();

    // Use the same position variance as R_(0,0)
    Eigen::Matrix3d Rpos = Eigen::Matrix3d::Identity() * R_(0,0);

    const Eigen::Vector3d z = gnss_pos;
    const Eigen::Vector3d y = z - Hpos * x_;
    const Eigen::Matrix3d S = Hpos * P_ * Hpos.transpose() + Rpos;
    const Eigen::Matrix<double,15,3> K = P_ * Hpos.transpose() * S.inverse();

    x_ += K * y;

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(15,15);
    P_ = (I - K * Hpos) * P_;
    P_ = 0.5 * (P_ + P_.transpose());
}

// ----------------- Combined Observer -----------------
Eigen::VectorXd KalmanFilter15::Observer(const Eigen::Vector3d &gnss_pos1,
                                         const Eigen::Vector3d &gnss_pos2,
                                         double gnss_heading,
                                         const Eigen::Vector3d &imu_accel,
                                         const Eigen::Vector3d &imu_gyro)
{
    // 1) Predict using IMU
    predict(imu_accel, imu_gyro);

    // 2) One fused update with pos1 + heading
    update(gnss_pos1, gnss_heading);

    // 3) Second antenna as an additional POSITION-ONLY update
    update_pos_only(gnss_pos2);

    // Return full 15-state estimate
    return x_;
}

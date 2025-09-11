#include "Observers/kalmanObserver.hpp"
#include <cmath>

// ----------------- Constructor -----------------
KalmanFilter15::KalmanFilter15(double dt) : dt_(dt) {
    x_ = Eigen::VectorXd::Zero(15);
    P_ = Eigen::MatrixXd::Identity(15,15) * 0.1;

    F_ = Eigen::MatrixXd::Identity(15,15);
    Q_ = Eigen::MatrixXd::Identity(15,15) * 1e-4;

    // Measurement matrix (GNSS: xN,yN,zN + heading)
    H_ = Eigen::MatrixXd::Zero(4,15);
    H_.block<3,3>(0,6) = Eigen::Matrix3d::Identity();  // positions
    H_(3,11) = 1;                                     // heading

    // Measurement noise
    R_ = Eigen::MatrixXd::Identity(4,4);
    R_.block<3,3>(0,0) *= 0.05*0.05;  // GNSS pos variance
    R_(3,3) = 0.01*0.01;             // GNSS heading variance
}

// ----------------- Helper: Euler rates -----------------
Eigen::Vector3d KalmanFilter15::eulerRateFromBodyRates(const Eigen::Vector3d &bodyRates,
                                                       const Eigen::Vector3d &angles) {
    double phi = angles(0), theta = angles(1);
    Eigen::Matrix3d T;
    T << 1, std::sin(phi)*std::tan(theta), std::cos(phi)*std::tan(theta),
         0, std::cos(phi), -std::sin(phi),
         0, std::sin(phi)/std::cos(theta), std::cos(phi)/std::cos(theta);
    return T * bodyRates;
}

// ----------------- Predict -----------------
void KalmanFilter15::predict(const Eigen::Vector3d &imu_accel,
                             const Eigen::Vector3d &imu_gyro) {
    // Extract angles
    double phi   = x_(9);
    double theta = x_(10);
    double psi   = x_(11);

    // Correct gyro measurements by subtracting estimated biases
    Eigen::Vector3d gyro_corrected = imu_gyro - x_.segment<3>(12); // b_p,b_q,b_r

    // Compute Euler angle rates
    Eigen::Vector3d euler_dot = eulerRateFromBodyRates(gyro_corrected, x_.segment<3>(9));

    // Update angles
    x_(9) += euler_dot(0) * dt_;
    x_(10) += euler_dot(1) * dt_;
    x_(11) += euler_dot(2) * dt_;

    // Rotation matrix body -> NED
    double cphi = std::cos(phi), sphi = std::sin(phi);
    double cth  = std::cos(theta), sth  = std::sin(theta);
    double cpsi = std::cos(psi), spsi = std::sin(psi);

    Eigen::Matrix3d R_b2n;
    R_b2n << cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi,
             cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi,
             -sth,     sphi*cth,                  cphi*cth;

    // Integrate positions using body-frame velocities
    x_.segment<3>(6) += R_b2n * x_.segment<3>(0) * dt_;

    // Update linear velocities in body frame (can integrate accel if desired)
    x_.segment<3>(0) += imu_accel * dt_;

    // Gyro biases: random walk (small process noise)
    // Here we just keep them constant in predict; Q_ should have small diag for biases

    // Propagate covariance
    P_ = F_ * P_ * F_.transpose() + Q_;
    P_ = 0.5*(P_ + P_.transpose()); // ensure symmetry
}

// ----------------- Update -----------------
void KalmanFilter15::update(const Eigen::Vector3d &gnss_pos,
                            double gnss_heading) {
    Eigen::VectorXd z(4);
    z.segment<3>(0) = gnss_pos;
    z(3) = gnss_heading;

    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    x_ += K * y;
    P_ = (Eigen::MatrixXd::Identity(15,15) - K * H_) * P_;
    P_ = 0.5*(P_ + P_.transpose());
}


// ----------------- Combined Observer -----------------
Eigen::VectorXd KalmanFilter15::Observer(const Eigen::Vector3d &gnss_pos1,
                                                const Eigen::Vector3d &gnss_pos2,
                                                double gnss_heading,
                                                const Eigen::Vector3d &imu_accel,
                                                const Eigen::Vector3d &imu_gyro)
{
    // 1️⃣ Predict step using IMU
    predict(imu_accel, imu_gyro);

    // 2️⃣ Update step using first GNSS
    update(gnss_pos1, gnss_heading);

    // 3️⃣ Update step using second GNSS
    update(gnss_pos2, gnss_heading);

    // 4️⃣ Return full 15-state estimate
    return x_;
}

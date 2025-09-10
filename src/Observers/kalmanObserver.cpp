#include "Observers/kalmanObserver.hpp"

KalmanFilter12::KalmanFilter12(double dt) : dt_(dt)
{
    x_ = Eigen::VectorXd::Zero(12);
    P_ = Eigen::MatrixXd::Identity(12,12) * 0.1;

    F_ = Eigen::MatrixXd::Identity(12,12);
    Q_ = Eigen::MatrixXd::Identity(12,12) * 1e-4;

    // Measurement matrix
    H_ = Eigen::MatrixXd::Zero(10,12);
    H_(0,6)=1; H_(1,7)=1; H_(2,8)=1; // GNSS positions
    H_(3,11)=1; // GNSS heading
    H_(4,0)=1; H_(5,1)=1; H_(6,2)=1; // IMU accel
    H_(7,3)=1; H_(8,4)=1; H_(9,5)=1; // IMU gyro

    R_ = Eigen::MatrixXd::Identity(10,10);
    R_.block<3,3>(0,0) *= 0.05*0.05;  // GNSS pos variance
    R_(3,3) = 0.01*0.01;               // GNSS heading variance
    R_.block<3,3>(4,4) *= 0.02*0.02;  // IMU accel variance
    R_.block<3,3>(7,7) *= 0.001*0.001; // IMU gyro variance
}

// ----------------- Predict -----------------
void KalmanFilter12::predict(const Eigen::Vector3d &imu_accel,
                             const Eigen::Vector3d &imu_gyro)
{
    x_(0) += imu_accel(0) * dt_;
    x_(1) += imu_accel(1) * dt_;
    x_(2) += imu_accel(2) * dt_;

    x_(3) += imu_gyro(0) * dt_;
    x_(4) += imu_gyro(1) * dt_;
    x_(5) += imu_gyro(2) * dt_;

    x_(6) += x_(0) * dt_;
    x_(7) += x_(1) * dt_;
    x_(8) += x_(2) * dt_;

    x_(9)  += x_(3) * dt_;
    x_(10) += x_(4) * dt_;
    x_(11) += x_(5) * dt_;

    P_ = F_ * P_ * F_.transpose() + Q_;
}

// ----------------- Update -----------------
void KalmanFilter12::update(const Eigen::Vector3d &gnss_pos,
                            double gnss_heading)
{
    Eigen::VectorXd z(10);
    z.segment<3>(0) = gnss_pos;
    z(3) = gnss_heading;
    z.segment<3>(4) = x_.segment<3>(0);
    z.segment<3>(7) = x_.segment<3>(3);

    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    x_ += K * y;
    P_ = (Eigen::MatrixXd::Identity(12,12) - K * H_) * P_;
}

// ----------------- Combined Observer -----------------
Eigen::VectorXd KalmanFilter12::Observer(const Eigen::Vector3d &gnss_pos1,
                                         const Eigen::Vector3d &gnss_pos2,
                                         double gnss_heading,
                                         const Eigen::Vector3d &imu_accel,
                                         const Eigen::Vector3d &imu_gyro)
{
    predict(imu_accel, imu_gyro);
    update(gnss_pos1, gnss_heading);
    update(gnss_pos2, gnss_heading);
    return x_;
}

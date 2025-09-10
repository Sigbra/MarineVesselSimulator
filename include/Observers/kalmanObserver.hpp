#ifndef KALMANFILTER12_HPP
#define KALMANFILTER12_HPP

#include <Eigen/Dense>
#include <iostream>

struct KFState {
    Eigen::VectorXd x;   // state estimate (12x1)
    Eigen::MatrixXd P;   // state covariance (12x12)
};

class KalmanFilter12 {
public:
    KalmanFilter12(double dt);

    // ----------------- Existing methods -----------------
    void predict(const Eigen::Vector3d &imu_accel,
                 const Eigen::Vector3d &imu_gyro);

    void update(const Eigen::Vector3d &gnss_pos,
                double gnss_heading);

    KFState getState() const { return {x_, P_}; }

    // ----------------- New combined method -----------------
Eigen::VectorXd Observer(const Eigen::Vector3d &gnss_pos1,
                         const Eigen::Vector3d &gnss_pos2,
                         double gnss_heading,
                         const Eigen::Vector3d &imu_accel,
                         const Eigen::Vector3d &imu_gyro);

private:
    double dt_;
    Eigen::VectorXd x_;    // 12x1 state vector
    Eigen::MatrixXd P_;    // 12x12 covariance
    Eigen::MatrixXd F_;    // 12x12 state transition
    Eigen::MatrixXd Q_;    // 12x12 process noise
    Eigen::MatrixXd H_;    // measurement matrix
    Eigen::MatrixXd R_;    // measurement noise
};

#endif // KALMANFILTER12_HPP


#ifndef KALMANFILTER15_HPP
#define KALMANFILTER15_HPP

#include <Eigen/Dense>
#include <iostream>

struct KFState15 {
    Eigen::VectorXd x;   // 15x1 state estimate
    Eigen::MatrixXd P;   // 15x15 covariance
};

class KalmanFilter15 {
public:
    KalmanFilter15(double dt);

    // Predict using IMU (accelerometer + gyro)
    void predict(const Eigen::Vector3d &imu_accel,
                 const Eigen::Vector3d &imu_gyro);

    // Update using GNSS position and heading
    void update(const Eigen::Vector3d &gnss_pos,
                double gnss_heading);

    // Position-only update (used for second antenna in Observer)
    void update_pos_only(const Eigen::Vector3d &gnss_pos);

    // One-shot convenience
    Eigen::VectorXd Observer(const Eigen::Vector3d &gnss_pos1,
                             const Eigen::Vector3d &gnss_pos2,
                             double gnss_heading,
                             const Eigen::Vector3d &imu_accel,
                             const Eigen::Vector3d &imu_gyro);

    KFState15 getState() const { return {x_, P_}; }

private:
    double dt_;
    Eigen::VectorXd x_;    // 15x1 state vector
    Eigen::MatrixXd P_;    // 15x15 covariance
    Eigen::MatrixXd F_;    // 15x15 state transition (Jacobian)
    Eigen::MatrixXd Q_;    // 15x15 process noise
    Eigen::MatrixXd H_;    // 4x15 measurement matrix (pos+heading)
    Eigen::MatrixXd R_;    // 4x4 measurement noise

    // Helper: convert body rates to Euler rates
    Eigen::Vector3d eulerRateFromBodyRates(const Eigen::Vector3d &bodyRates,
                                           const Eigen::Vector3d &angles);
};

#endif // KALMANFILTER15_HPP

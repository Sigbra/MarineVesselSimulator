#ifndef IMU_HPP
#define IMU_HPP

// Using "DeepVL: Dynamics and Inertial Measurements-based Deep Velocity Learning for Underwater Odometry"
// by Mohit Singh, and Kostas Alexis
// Date; 2025
// url={https://arxiv.org/abs/2502.07726}

#include <Eigen/Dense>
#include <random>

struct IMUData {
    Eigen::Vector3d accel;   // accelerometer measurements (body frame)
    Eigen::Vector3d gyro;    // gyroscope measurements (body frame)
};

/**
 * @brief Simulate raw IMU measurements following NTNU formulation
 * 
 * @param x           12x1 state vector:
 *                    x(0:2) = u,v,w (body velocities)
 *                    x(3:5) = p,q,r (body angular rates)
 *                    x(6:8) = xn,yn,zn (NED position)
 *                    x(9:11)= phi,theta,psi (attitude)
 * @param gen         Random number generator
 * @param ba          Accelerometer bias (3x1)
 * @param bgyro       Gyroscope bias (3x1)
 * @param sigma_acc   Accelerometer noise std (m/s^2)
 * @param sigma_gyro  Gyroscope noise std (rad/s)
 * @return IMUData    Noisy IMU measurements
 */
IMUData raw_IMU(const Eigen::VectorXd &x,
                std::mt19937 &gen,
                Eigen::Vector3d &ba,
                Eigen::Vector3d &bgyro,
                double sigma_acc = 0.01,
                double sigma_gyro = 0.001);

#endif // IMU_HPP

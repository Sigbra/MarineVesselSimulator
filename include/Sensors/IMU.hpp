#ifndef IMU_HPP
#define IMU_HPP

// Using "DeepVL: Dynamics and Inertial Measurements-based Deep Velocity Learning for Underwater Odometry"
// by Mohit Singh, and Kostas Alexis (2025)

#include <Eigen/Dense>
#include <random>

struct IMUData {
    Eigen::Vector3d accel;   // accelerometer measurements (body frame)
    Eigen::Vector3d gyro;    // gyroscope measurements (body frame)
};

/**
 * @brief Simulate raw IMU measurements.
 *        Accelerometer includes gravity (remove with GravityCompensation()).
 * 
 * @param x           12x1 state vector:
 *                    x(0:2) = u,v,w (body velocities)
 *                    x(3:5) = p,q,r (body angular rates)
 *                    x(6:8) = xn,yn,zn (NED position)
 *                    x(9:11)= phi,theta,psi (attitude)
 * @param gen         Random number generator
 * @param ba          Accelerometer bias (3x1), updated in-place (random walk)
 * @param bgyro       Gyroscope bias (3x1), updated in-place (random walk)
 * @param dt          Sample time [s] (used to approximate a_body from v_body)
 * @param sigma_acc   Accelerometer noise std (m/s^2)
 * @param sigma_gyro  Gyroscope noise std (rad/s)
 * @return IMUData    Noisy IMU measurements (accel includes gravity)
 */
IMUData raw_IMU(const Eigen::VectorXd &x,
                const Eigen::VectorXd &xdot,
                std::mt19937 &gen,
                Eigen::Vector3d &ba,
                Eigen::Vector3d &bgyro,
                double sigma_acc = 0.01,
                double sigma_gyro = 0.001);

<<<<<<< HEAD
=======
IMUData raw_IMU_v2(const Eigen::VectorXd &x,
                   const Eigen::VectorXd &x_dot,
                   std::mt19937 &gen,
                   Eigen::Vector3d &ba,          // accel bias (updated in-place)
                   Eigen::Vector3d &bgyro,       // gyro bias (updated in-place)
                   double dt,                    // <-- sample time [s]
                   double sigma_acc_nd,          // accel noise density [m/s^2 / sqrt(Hz)]
                   double sigma_gyro_nd);

>>>>>>> 35-pirnn-observer
// Remove gravity from raw accelerometer using attitude (φ,θ,ψ)
Eigen::Vector3d GravityCompensation(const Eigen::Vector3d &accel_raw,
                                    double phi, double theta, double psi);

#endif // IMU_HPP

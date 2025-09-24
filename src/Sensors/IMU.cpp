#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "Sensors/IMU.hpp"
#include "Models/model_utilities.hpp"

// Gravity compensation helper kept as-is (just clarified names)
Eigen::Vector3d GravityCompensation(const Eigen::Vector3d &accel_raw,
                                    double phi, double theta, double psi)
{
    // Gravity in NED
    const Eigen::Vector3d gNED(0.0, 0.0, 9.81);

    // Rotation matrices
    Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);
    Eigen::Matrix3d R_n2b = R_b2n.transpose();

    // Gravity in body frame
    Eigen::Vector3d gB = R_n2b * gNED;

    // Subtract gravity from accelerometer measurement
    return accel_raw + gB;
}



// IMU: produce raw sensor outputs (accelerometer includes gravity), with bias RW
// NOTE: requires dt to compute acceleration from velocities.
// If you prefer, you can pass true a_body from your model instead of differencing.
IMUData raw_IMU(const Eigen::VectorXd &x,
                const Eigen::VectorXd &x_dot,
                std::mt19937 &gen,
                Eigen::Vector3d &ba,          // accel bias (updated in-place)
                Eigen::Vector3d &bgyro,       // gyro bias (updated in-place)
                double sigma_acc,
                double sigma_gyro)
{
    IMUData imu;

    // --- Extract states ---
    const double u = x(0), v = x(1), w = x(2);
    const double p = x(3), q = x(4), r = x(5);
    const double phi   = x(9), theta = x(10), psi = x(11);

    // --- Rotation matrices ---
    const Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);  // body → NED
    const Eigen::Matrix3d R_n2b = R_b2n.transpose();      // NED → body

    // --- Body linear acceleration (from model) ---
    Eigen::Vector3d aB(x_dot(0), x_dot(1), x_dot(2));

    // --- Gravity in NED (z positive down) ---
    const Eigen::Vector3d gNED(0.0, 0.0, 9.81);

    // --- Noise generators ---
    std::normal_distribution<double> nd_acc(0.0, sigma_acc);
    std::normal_distribution<double> nd_gyro(0.0, sigma_gyro);

    Eigen::Vector3d na(nd_acc(gen), nd_acc(gen), nd_acc(gen));
    Eigen::Vector3d ng(nd_gyro(gen), nd_gyro(gen), nd_gyro(gen));

    // --- Accelerometer (specific force) ---
    // fB = aB - gB
    Eigen::Vector3d gB = R_n2b * gNED;
    imu.accel = aB - gB + ba + na;

    // --- Gyroscope ---
    imu.gyro = Eigen::Vector3d(p, q, r) + bgyro + ng;

    // --- Random walk on biases ---
    std::normal_distribution<double> nd_ba(0.0, 1e-6);
    std::normal_distribution<double> nd_bg(0.0, 1e-7);
    ba    += Eigen::Vector3d(nd_ba(gen), nd_ba(gen), nd_ba(gen));
    bgyro += Eigen::Vector3d(nd_bg(gen), nd_bg(gen), nd_bg(gen));

    return imu;
}



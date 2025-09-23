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
    const Eigen::Vector3d g_NED(0.0, 0.0, 9.81);

    // Rotation body -> NED
    const Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);

    // Gravity in body frame
    const Eigen::Vector3d g_body = R_b2n.transpose() * g_NED; // NED->body

    // Remove gravity to get linear acceleration in body frame
    return accel_raw - g_body;
}

// IMU: produce raw sensor outputs (accelerometer includes gravity), with bias RW
// NOTE: requires dt to compute acceleration from velocities.
// If you prefer, you can pass true a_body from your model instead of differencing.
IMUData raw_IMU(const Eigen::VectorXd &x,
                std::mt19937 &gen,
                Eigen::Vector3d &ba,          // accel bias (updated in-place)
                Eigen::Vector3d &bgyro,       // gyro bias (updated in-place)
                double dt,
                double sigma_acc  /* m/s^2 */,
                double sigma_gyro /* rad/s */)
{
    IMUData imu;

    // States
    const double u = x(0), v = x(1), w = x(2);
    const double p = x(3), q = x(4), r = x(5);
    const double phi   = x(9), theta = x(10), psi = x(11);

    // Rotation body<->NED
    const Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);
    const Eigen::Matrix3d R_n2b = R_b2n.transpose();

    // Body velocities
    const Eigen::Vector3d vB(u, v, w);

    // --- Approximate body linear acceleration ---
    // We difference body velocity and include omega x v term.
    // To keep the function self-contained, we use a static previous value.
    // Initialize on first call.
    static bool first = true;
    static Eigen::Vector3d vB_prev = Eigen::Vector3d::Zero();

    Eigen::Vector3d aB;
    if (first) {
        aB = Eigen::Vector3d::Zero();  // no info on first call
        first = false;
    } else {
        aB = (vB - vB_prev) / std::max(dt, 1e-9) + Eigen::Vector3d(p,q,r).cross(vB);
    }
    vB_prev = vB;

    // Noise
    std::normal_distribution<double> nd_acc(0.0, sigma_acc);
    std::normal_distribution<double> nd_gyro(0.0, sigma_gyro);

    const Eigen::Vector3d na(nd_acc(gen),   nd_acc(gen),   nd_acc(gen));
    const Eigen::Vector3d ng(nd_gyro(gen),  nd_gyro(gen),  nd_gyro(gen));

    // Gravity in NED
    const Eigen::Vector3d gNED(0.0, 0.0, 9.81);

    // Accelerometer measures: a_body + g_body + bias + noise
    imu.accel = aB + R_n2b * gNED + ba + na;

    // Gyro measures: body rates + bias + noise
    imu.gyro  = Eigen::Vector3d(p, q, r) + bgyro + ng;

    // Bias random walks (very small)
    std::normal_distribution<double> nd_ba(0.0, 1e-6);
    std::normal_distribution<double> nd_bg(0.0, 1e-7);
    ba    += Eigen::Vector3d(nd_ba(gen), nd_ba(gen), nd_ba(gen));
    bgyro += Eigen::Vector3d(nd_bg(gen), nd_bg(gen), nd_bg(gen));

    return imu;
}

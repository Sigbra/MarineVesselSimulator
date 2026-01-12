#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "Sensors/IMU.hpp"
#include "Models/model_utilities.hpp"

// raw IMU using w x v term for accel and taking dt into account
IMUData raw_IMU(const Eigen::VectorXd &x,
                   const Eigen::VectorXd &x_dot,
                   std::mt19937 &gen,
                   Eigen::Vector3d &ba,          // accel bias (updated in-place)
                   Eigen::Vector3d &bgyro,       // gyro bias (updated in-place)
                   double dt,                    // <-- sample time [s]
                   double sigma_acc_nd,          // accel noise density [m/s^2 / sqrt(Hz)]
                   double sigma_gyro_nd)         // gyro  noise density [rad/s  / sqrt(Hz)]
{
    IMUData imu;

    // ---- Extract states (your indexing) ----
    const double u = x(0), v = x(1), w = x(2);
    const double p = x(3), q = x(4), r = x(5);
    const double phi = x(9), theta = x(10), psi = x(11);

    // ---- Frames & gravity (Body→NED; z-down) ----
    const Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);  // Body → NED/END
    const Eigen::Matrix3d R_n2b = R_b2n.transpose();      // NED  → Body
    const Eigen::Vector3d gNED(0.0, 0.0, 9.81);
    const Eigen::Vector3d gB = R_n2b * gNED;              // gravity in body

    // ---- Inertial acceleration expressed in body: a^b = (dv/dt)_B + ω×v ----
    const Eigen::Vector3d vB(u, v, w);
    const Eigen::Vector3d wB(p, q, r);
    const Eigen::Vector3d aB = Eigen::Vector3d(x_dot(0), x_dot(1), x_dot(2)) + wB.cross(vB);

    // ---- White measurement noise from densities (per-sample σ = ND / sqrt(dt)) ----
    static std::normal_distribution<double> N01(0.0, 1.0);
    const double inv_sqrt_dt = (dt > 0.0) ? 1.0 / std::sqrt(dt) : 0.0;
    const Eigen::Vector3d na = sigma_acc_nd  * inv_sqrt_dt *
        Eigen::Vector3d(N01(gen), N01(gen), N01(gen));
    const Eigen::Vector3d ng = sigma_gyro_nd * inv_sqrt_dt *
        Eigen::Vector3d(N01(gen), N01(gen), N01(gen));

    // ---- Outputs ----
    // Accelerometer measures specific force: f = a - g
    imu.accel = aB - gB + ba + na;
    // Gyro measures angular rate
    imu.gyro  = wB + bgyro + ng;

    // ---- Bias random walks (intensities are units / sqrt(s)) ----
    constexpr double SIG_RW_BA = 1e-6; // 3e-6 m/s^2 / sqrt(s)  (~10–40 µg over ~1 h)
    constexpr double SIG_RW_BG = 1e-7; // 6e-7 rad/s  / sqrt(s)  (~5–10 °/h bias)
    const double sdt = (dt > 0.0) ? std::sqrt(dt) : 0.0;
    ba    += SIG_RW_BA * sdt * Eigen::Vector3d(N01(gen), N01(gen), N01(gen));
    bgyro += SIG_RW_BG * sdt * Eigen::Vector3d(N01(gen), N01(gen), N01(gen));

    return imu;
}
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


// raw IMU using w x v term for accel and taking dt into account
IMUData raw_IMU_v2(const Eigen::VectorXd &x,
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
    constexpr double SIG_RW_BA = 3e-6; // m/s^2 / sqrt(s)  (~10–40 µg over ~1 h)
    constexpr double SIG_RW_BG = 6e-7; // rad/s  / sqrt(s)  (~5–10 °/h bias)
    const double sdt = (dt > 0.0) ? std::sqrt(dt) : 0.0;
    ba    += SIG_RW_BA * sdt * Eigen::Vector3d(N01(gen), N01(gen), N01(gen));
    bgyro += SIG_RW_BG * sdt * Eigen::Vector3d(N01(gen), N01(gen), N01(gen));

    return imu;
}
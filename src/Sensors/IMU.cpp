#include "Sensors/IMU.hpp"
#include "Models/model_utilities.hpp" 

IMUData raw_IMU(const Eigen::VectorXd &x,
                std::mt19937 &gen,
                Eigen::Vector3d &ba,
                Eigen::Vector3d &bgyro,
                double sigma_acc,
                double sigma_gyro)
{
    IMUData imu;

    // --- Extract states ---
    double u = x(0), v = x(1), w = x(2);
    double p = x(3), q = x(4), r = x(5);
    double phi   = x(9), theta = x(10), psi = x(11);

    // --- Rotation matrix: world (NED) -> body ---
    Eigen::Matrix3d Rnb = Rzyx(phi, theta, psi);
    Eigen::Matrix3d Rwb = Rnb.transpose(); // q^-1

    // Gravity in world frame
    Eigen::Vector3d gW(0, 0, 9.81);

    // --- Accelerometer ---
    Eigen::Vector3d aB(u, v, w); // linear acceleration in body frame (can refine with derivatives)
    
    // Noise
    std::normal_distribution<double> nd_acc(0.0, sigma_acc);
    Eigen::Vector3d na(nd_acc(gen), nd_acc(gen), nd_acc(gen));

    // Compute raw measurement
    imu.accel = aB + ba + Rwb * gW + na;

    // --- Gyroscope ---
    std::normal_distribution<double> nd_gyro(0.0, sigma_gyro);
    Eigen::Vector3d ngyro(nd_gyro(gen), nd_gyro(gen), nd_gyro(gen));

    imu.gyro = Eigen::Vector3d(p, q, r) + bgyro + ngyro;

    // --- Random walk update of biases ---
    std::normal_distribution<double> nd_ba(0.0, 1e-6);
    std::normal_distribution<double> nd_bg(0.0, 1e-7);
    ba    += Eigen::Vector3d(nd_ba(gen), nd_ba(gen), nd_ba(gen));
    bgyro += Eigen::Vector3d(nd_bg(gen), nd_bg(gen), nd_bg(gen));

    return imu;
}

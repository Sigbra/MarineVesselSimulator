#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "Models/model_utilities.hpp"
#include "Utilities/calculations.hpp"
// raw_GNSS: simulate raw GNSS antenna position measurement (NED) given state x and lever-arm (body-frame)
Eigen::Vector3d raw_GNSS(const Eigen::VectorXd &x,
                         const Eigen::Vector3d &lever_arm_body,
                         std::mt19937 &gen,
                         double sigma_pos /* m */)
{
    // Extract pose
    const double xn = x(6);
    const double yn = x(7);
    const double zn = x(8);
    const double phi   = x(9);
    const double theta = x(10);
    const double psi   = x(11);

    // Body-origin in NED
    const Eigen::Vector3d p_body_origin_ned(xn, yn, zn);

    // Rotation body -> NED  (consistent across codebase)
    const Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);

    // Antenna true position in NED
    const Eigen::Vector3d ant_true_ned = p_body_origin_ned + R_b2n * lever_arm_body;

    // Add independent Gaussian noise on N,E,D
    std::normal_distribution<double> nd(0.0, sigma_pos);
    const Eigen::Vector3d noise(nd(gen), nd(gen), nd(gen));

    return ant_true_ned + noise;
}

// Compute body-origin position (NED) from a raw GNSS antenna measurement, using TRUE attitude (optimistic sim)
Eigen::Vector3d origin_from_raw_GNSS(const Eigen::VectorXd &x,
                                     const Eigen::Vector3d &ant_meas_ned,
                                     const Eigen::Vector3d &lever_arm_body)
{
    const double phi   = x(9);
    const double theta = x(10);
    const double psi   = x(11);

    const Eigen::Matrix3d R_b2n = Rzyx(phi, theta, psi);

    // Back-project antenna to origin in NED
    return ant_meas_ned - R_b2n * lever_arm_body;
}

double gnss_heading_from_two_antennas(const Eigen::Vector3d &ant_port_ned,
                                      const Eigen::Vector3d &ant_stbd_ned)
{
    const Eigen::Vector3d b = ant_stbd_ned - ant_port_ned;

    const double bE = b(0); 
    const double bN = b(1); 

    // Yaw measured in rad(NED): atan2(E,N) + 90deg
    const double psi_meas = std::atan2(bE, bN) - M_PI/2;
    return ssa(psi_meas);
}

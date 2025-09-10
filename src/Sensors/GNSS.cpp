#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <tuple>
#include "Models/model_utilities.hpp"

// raw_GNSS: simulate raw GNSS antenna position measurement (NED) given state x and lever-arm (body-frame)
Eigen::Vector3d raw_GNSS(const Eigen::VectorXd &x,
                         const Eigen::Vector3d &lever_arm_body,
                         std::mt19937 &gen,
                         double sigma_pos = 0.05 /* m, default 5cm */)
{
    double xn = x(6); 
    double yn = x(7); 
    double zn = x(8); 
    double phi = x(9);
    double theta = x(10);
    double psi = x(11);

    // vehicle body-origin in NED
    Eigen::Vector3d p_body_origin_ned(xn, yn, zn); 

    // rotation body -> NED
    Eigen::Matrix3d Rnb = Rzyx(phi, theta, psi);

    // antenna position in NED = body-origin position + Rnb * lever-arm_body
    Eigen::Vector3d ant_true_ned = p_body_origin_ned + Rnb * lever_arm_body;

    // measurement noise (zero-mean gaussian independent on N,E,D)
    std::normal_distribution<double> nd(0.0, sigma_pos);
    Eigen::Vector3d noise(nd(gen), nd(gen), nd(gen));

    Eigen::Vector3d ant_meas_ned = ant_true_ned + noise;
    return ant_meas_ned;
}

Eigen::Vector3d origin_from_raw_GNSS(const Eigen::VectorXd &x,
                                      const Eigen::Vector3d &ant_meas_ned,
                                      const Eigen::Vector3d &lever_arm_body)
{
    // extract attitude
    double phi   = x(9);
    double theta = x(10);
    double psi   = x(11);

    // rotation body -> NED
    Eigen::Matrix3d Rnb = Rzyx(phi, theta, psi);

    // back-project to noisy origin estimate
    Eigen::Vector3d origin_est_ned = ant_meas_ned - Rnb * lever_arm_body;

    return origin_est_ned;
}


// heading from two GNSS antenna position signals (both in NED, noisy)
// antennas are placed port (y>0) and starboard (y<0) at the stern
double gnss_heading_from_two_antennas(const Eigen::Vector3d &ant_port_ned,
                                      const Eigen::Vector3d &ant_stbd_ned)
{
    // Baseline: starboard - port  (this makes +y = port, -y = starboard consistent)
    Eigen::Vector3d b = ant_stbd_ned - ant_port_ned;

    double bN = b(0); // North component
    double bE = b(1); // East component

    double psi_meas = std::atan2(bE, bN) + M_PI/2;

    // Wrap result to (-pi, pi]
    while (psi_meas <= -M_PI) psi_meas += 2.0 * M_PI;
    while (psi_meas >   M_PI) psi_meas -= 2.0 * M_PI;

    return psi_meas;
}

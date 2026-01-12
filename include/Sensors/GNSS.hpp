#ifndef GNSS_HPP
#define GNSS_HPP

// Using "GNSS-antenna lever arm compensation in aided inertial navigation of UAVs"
// by Stovner, Bård Nagy; Johansen, Tor Arne (2019, IEEE)

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <tuple>
#include "Models/model_utilities.hpp"

// Simulate raw GNSS antenna position measurement (NED) given state x and lever-arm (body-frame)
Eigen::Vector3d raw_GNSS(const Eigen::VectorXd &x,
                         const Eigen::Vector3d &lever_arm_body,
                         std::mt19937 &gen,
                         double sigma_pos = 0.05 /* m, default 5cm */);

// Back-project to body-origin (NED) from antenna measurement using TRUE attitude (optimistic sim)
Eigen::Vector3d origin_from_raw_GNSS(const Eigen::VectorXd &x,
                                     const Eigen::Vector3d &ant_meas_ned,
                                     const Eigen::Vector3d &lever_arm_body);

// Heading from two GNSS antenna positions (both in NED, noisy)
double gnss_heading_from_two_antennas(const Eigen::Vector3d &ant_port_ned,
                                      const Eigen::Vector3d &ant_stbd_ned);

#endif // GNSS_HPP
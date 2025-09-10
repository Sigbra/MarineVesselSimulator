#ifndef GNSS_HPP
#define GNSS_HPP

// Using "GNSS-antenna lever arm compensation in aided inertial navigation of UAVs"
// by Stovner, Bård Nagy; Johansen, Tor Arne
// Date; 2019
// Publisher; Institute of Electrical and Electronics Engineers (IEEE)
// (Found on NTNU Open)

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <tuple>
#include "Models/model_utilities.hpp"

// raw_GNSS: simulate raw GNSS antenna position measurement (NED) given state x and lever-arm (body-frame)
Eigen::Vector3d raw_GNSS(const Eigen::VectorXd &x,
                         const Eigen::Vector3d &lever_arm_body,
                         std::mt19937 &gen,
                         double sigma_pos = 0.05 /* m, default 5cm */);

Eigen::Vector3d origin_from_raw_GNSS(const Eigen::VectorXd &x,
                                      const Eigen::Vector3d &ant_meas_ned,
                                      const Eigen::Vector3d &lever_arm_body);

double gnss_heading_from_two_antennas(const Eigen::Vector3d &ant_port_ned,
                                      const Eigen::Vector3d &ant_stbd_ned);

#endif


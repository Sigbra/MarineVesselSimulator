#ifndef ALOSSPI_HPP
#define ALOSSPI_HPP

#include <utility>
#include "utilities.hpp"  // Defines: struct Waypoints { std::vector<double> x; std::vector<double> y; };

/// \brief Computes the desired LOS heading and cross-track error using the adaptive LOS (ALOS) guidance law.
/// 
/// \param x         The craft's North (x) position [m].
/// \param y         The craft's East (y) position [m].
/// \param Delta_h   Look-ahead distance [m] (typically 5–20 m).
/// \param gamma_h   Adaptive gain constant (typically 0.001).
/// \param h         Sampling time [s].
/// \param R_switch  Switching radius [m] (must be positive and less than the minimum distance between waypoints).
/// \param wpt       The waypoints structure with fields wpt.x and wpt.y (arrays of waypoint positions in NED).
/// \return A std::pair<double, double> where first is the desired heading (psi_ref in rad) and second is the cross-track error (y_e in m).

std::pair<double, double> ALOSpsi(double x, double y,
                                  double Delta_h, double gamma_h, double h,
                                  double R_switch,
                                  const Waypoints &wpt);

#endif // ALOSSPI_HPP

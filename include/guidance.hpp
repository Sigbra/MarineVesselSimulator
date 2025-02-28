#ifndef GUIDANCE_HPP
#define GUIDANCE_HPP
#include <Eigen/Dense>
#include "utilities.hpp"

void dynamicPositioning(const Waypoints &wpt,
    const Eigen::VectorXd &x,
    int &currentWptIndex,
    double &tau_X,
    double &tau_Y,
    double &tau_N,
    double &r_d,
    double &psi_d,
    double &z_psi,
    double h);

#endif 
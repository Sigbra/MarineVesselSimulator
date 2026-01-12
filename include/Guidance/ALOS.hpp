#ifndef ALOS_HPP
#define ALOS_HPP

#include <vector>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <iostream>
#include "Utilities/calculations.hpp"

class ALOS {
public:
    ALOS(double Delta_h, double gamma_h, double h);

    std::tuple<double, double> update(double xn, double yn,
                                      double path_x, double path_y,
                                      double path_x_dot, double path_y_dot,
                                      double precomputed_y_e = std::numeric_limits<double>::quiet_NaN());

    void reset();

private:
    // Path and guidance parameters.
    double Delta_h_;
    double gamma_h_;
    double h_;

    // Internal state.    
    double beta_hat;   
};

#endif

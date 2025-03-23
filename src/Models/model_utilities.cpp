#include "Models/model_utilities.hpp"       
#include "Utilities/calculations.hpp"     
#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>

//-------------------------------------------------------------------
// Helper function: Skew-symmetric matrix (Smtrx)
//-------------------------------------------------------------------
Eigen::Matrix3d Smtrx(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S <<      0, -v(2),  v(1),
          v(2),      0, -v(0),
         -v(1),  v(0),      0;
    return S;
}

//-------------------------------------------------------------------
// Helper function: Transformation matrix Hmtrx
// Returns a 6x6 matrix to transform from the center of gravity (CG)
// to the coordinate origin (CO) using the relation:
//   H = [ I   -Smtrx(r) ]
//       [ 0      I      ]
// where r is a 3x1 vector.
//-------------------------------------------------------------------
Eigen::MatrixXd Hmtrx(const Eigen::Vector3d &r) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6,6);
    H.block(0,0,3,3) = Eigen::Matrix3d::Identity();
    H.block(0,3,3,3) = -Smtrx(r);
    H.block(3,3,3,3) = Eigen::Matrix3d::Identity();
    return H;
}

//-------------------------------------------------------------------
// Helper function: Tzyx
//-------------------------------------------------------------------
Eigen::Matrix3d Tzyx(double phi, double theta) {
    Eigen::Matrix3d T;

    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double t_theta = std::tan(theta);
    double c_theta = std::cos(theta);

    T << 1, s_phi * t_theta, c_phi * t_theta,
         0, c_phi, -s_phi,
         0, s_phi / c_theta, c_phi / c_theta;

    return T;
}

//-------------------------------------------------------------------
// Helper function: m2c (added-mass to Coriolis matrix)
// Stub, return a 6x6 zero matrix.
//-------------------------------------------------------------------
Eigen::MatrixXd m2c(const Eigen::MatrixXd &MA, const Eigen::VectorXd & /*nu_r*/) {
    return Eigen::MatrixXd::Zero(6,6);
}

//-------------------------------------------------------------------
// Helper function: addedMassSurge
// Stub: returns a value proportional to the mass.
//-------------------------------------------------------------------
double addedMassSurge(double m, double /*L*/, double /*rho*/) {
    return 0.1 * m; 
}

// Function: Hoerner
// Computes the 2D Hoerner cross-flow form coefficient based on beam (B) and draft (T).
// The coefficient is obtained via linear interpolation from digitized data.
// Inputs:
//    B - beam (m)
//    T - draft (m)
// Output:
//    CY_2D - 2D Hoerner cross-flow form coefficient
double Hoerner(double B, double T) {
    // Compute the ratio B/(2*T)
    double ratio = B / (2.0 * T);
    
    // Digitized data: each pair is {B/(2*T), C_D}
    const std::vector<std::pair<double, double>> CD_DATA = {
        {0.0108623, 1.96608},
        {0.176606,  1.96573},
        {0.353025,  1.89756},
        {0.451863,  1.78718},
        {0.472838,  1.58374},
        {0.492877,  1.27862},
        {0.493252,  1.21082},
        {0.558473,  1.08356},
        {0.646401,  0.998631},
        {0.833589,  0.87959},
        {0.988002,  0.828415},
        {1.30807,   0.759941},
        {1.63918,   0.691442},
        {1.85998,   0.657076},
        {2.31288,   0.630693},
        {2.59998,   0.596186},
        {3.00877,   0.586846},
        {3.45075,   0.585909},
        {3.7379,    0.559877},
        {4.00309,   0.559315}
    };
    
    // If the ratio is above the highest value in the table, return the last value
    if (ratio >= CD_DATA.back().first) {
        return CD_DATA.back().second;
    }
    
    // If the ratio is below the first data point, return the first value.
    if (ratio <= CD_DATA.front().first) {
        return CD_DATA.front().second;
    }
    
    // Otherwise, search for the interval in which the ratio falls and interpolate.
    for (size_t i = 0; i < CD_DATA.size() - 1; ++i) {
        double x1 = CD_DATA[i].first;
        double x2 = CD_DATA[i+1].first;
        if (ratio >= x1 && ratio <= x2) {
            double y1 = CD_DATA[i].second;
            double y2 = CD_DATA[i+1].second;
            // Linear interpolation
            double t = (ratio - x1) / (x2 - x1);
            return y1 + t * (y2 - y1);
        }
    }
    
    // Fallback: Should never reach here.
    throw std::runtime_error("Interpolation error in Hoerner function.");
}

//-------------------------------------------------------------------
// Helper function: crossFlowDrag
//-------------------------------------------------------------------
Eigen::VectorXd crossFlowDrag(double L, double B, double T, const Eigen::VectorXd& nu_r) {
    // Check that nu_r has the correct size
    if (nu_r.size() < 6) {
        throw std::runtime_error("nu_r must be a 6-element vector.");
    }

    const double rho = 1025.0;   // density of water (kg/m^3)
    const double dx = L / 20.0;    // divide the craft into 20 strips
    const double Cd_2D = Hoerner(B, T);  // 2-D drag coefficient from Hoerner's curve

    double Yh = 0.0, Zh = 0.0, Mh = 0.0, Nh = 0.0;

    // Loop from -L/2 to L/2 with step dx.
    // Use a for loop; note that floating point comparisons might not hit L/2 exactly.
    for (double xL = -L/2.0; xL <= L/2.0 + 1e-6; xL += dx) {
        // Extract relevant components from nu_r (C++ indices):
        // v_r (sway velocity) is nu_r[1] (MATLAB nu_r(2))
        // w_r (heave velocity) is nu_r[2] (MATLAB nu_r(3))
        // q (pitch rate)      is nu_r[4] (MATLAB nu_r(5))
        // r (yaw rate)        is nu_r[5] (MATLAB nu_r(6))
        double v_r = nu_r(1);
        double w_r = nu_r(2);
        double q   = nu_r(4);
        double r   = nu_r(5);

        // Effective velocities at the strip (including rotational effects)
        double effective_v = v_r + xL * r;
        double effective_w = w_r + xL * q;
        double U_h = std::abs(effective_v) * effective_v;
        double U_v = std::abs(effective_w) * effective_w;

        // Accumulate cross-flow drag forces and moments
        Yh -= 0.5 * rho * T * Cd_2D * U_h * dx;       // sway force
        Zh -= 0.5 * rho * T * Cd_2D * U_v * dx;         // heave force
        Mh -= 0.5 * rho * T * Cd_2D * xL * U_v * dx;    // pitch moment
        Nh -= 0.5 * rho * T * Cd_2D * xL * U_h * dx;    // yaw moment
    }

    // Assemble the 6-DOF cross-flow drag vector: [0, Yh, Zh, 0, Mh, Nh]^T
    Eigen::VectorXd tau_crossflow(6);
    tau_crossflow << 0.0, Yh, Zh, 0.0, Mh, Nh;

    return tau_crossflow;
}

//-------------------------------------------------------------------
// Helper function: eulerang
// Returns the 3x3 kinematic transformation matrix from body angular
// rates to Euler angle rates (using the ZYX Euler angle convention).
//-------------------------------------------------------------------
Eigen::MatrixXd eulerang(double phi, double theta, double psi) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, 6);
    Eigen::Matrix3d J1 = Rzyx(phi, theta, psi);
    Eigen::Matrix3d J2 = Tzyx(phi, theta);
    

    J.block<3,3>(0,0) = J1;
    J.block<3,3>(3,3) = J2;
    //std::cout << "eulerang, J: \n" << J << std::endl;

    return J;
}

//-------------------------------------------------------------------
// Helper function: Rzyx
// Returns the rotation matrix (3x3) from the body-fixed frame to the
// inertial frame using the ZYX Euler angle convention.
//-------------------------------------------------------------------
Eigen::Matrix3d Rzyx(double phi, double theta, double psi) {
    // Compute trigonometric values
    double cphi = std::cos(phi);
    double sphi = std::sin(phi);
    double cth  = std::cos(theta);
    double sth  = std::sin(theta);
    double cpsi = std::cos(psi);
    double spsi = std::sin(psi);

    // Define rotation matrix R
    Eigen::Matrix3d R;
    R <<  cpsi * cth,  -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth,
          spsi * cth,   cpsi * cphi + sphi * sth * spsi,  -cpsi * sphi + sth * spsi * cphi,
          -sth,         cth * sphi,                       cth * cphi;

    return R;
}

Eigen::Vector3d CO_Offset(double U) {

    //Length of craft;
    double L = 5;

    // Lower and upper bounds [m/s]
    double U_lower = 2; 
    double U_upper = 10;

    // Max and Min x values for CO [m]
    // - Midship
    double x_min = 0;
    // - Front of ship
    double x_max = 0.4*L;

    // CO
    double x = 0;
    double y = 0;
    double z = 0;

    // Finding offset x based on speed
    if (U < U_lower) {
        x = x_min;
    }
    else if (U > U_upper) {
        x = x_max;
    }
    else {
        x = x_min + (x_max - x_min) * (U-U_lower) / (U_upper-U_lower);
    }

    return Eigen::Vector3d(x, y, z);
}

// Calculates real propeller revs
// Assuming positive and negative Bollard and propeller revs are the same.
std::vector<double> nReal(std::vector<double> n_relative) {
    //n_relative: 0 to 1
    //n_real:   100 to -100
    double n1_real = 200 * n_relative[0] - 100; 
    double n2_real = 200 * n_relative[1] - 100; 
    return {n1_real, n2_real};
}

// Calculating thrusts based on relative propellar revs (n). Obs, wrong comments
//
//   Expecting n_r is scaled between [0, 1].
//   Thrust_pos = k_pos * (n*|n| - 0.5²) / (1-0.25), k_pos = Positive bollard pull = 200*g,
//   Thrust_neg = k_neg * (n*|n| - 0.5²) / (1-0.75), k_neg = Negative bollard pull = 200*g,
//
//   Same as direct Thrusts calculation;
//   Thrust_pos = k_pos * n_real*|n_real|, 
//   Thrust_neg = k_neg * n_real*|n_real|,
//   where 
//   n_real = nReal(n),
//   k_pos = Bollard_pull_pos / n_real_max, n_real_max = +100 revs,
//   k_neg = Bollard_pull_neg / n_real_min, n_real_min = -100 revs,
//
//   This formulation, as opposed to the direct one avoids nlpsol's solver issue at n = {0, 0}.
//
//   Assuming max positive and max negative propellar revs are the same.
Eigen::VectorXd ThrustsFromRealativeN(Eigen::VectorXd n_r) {
    double g = 9.81;
    double k_pos = 200*g;
    double k_neg = 200*g; 

    int n_r_size = n_r.size();

    Eigen::VectorXd Thrusts(n_r_size);
    Thrusts.setZero();

    for (int i = 0; i < n_r_size; ++i) {
        double n_i = n_r(i);
        if (n_i >= 0 && n_i <= 1) {
            Thrusts(i) = k_pos * n_i * fabs(n_i);
        }
        else if (n_i >= -1 && n_i < 0) {
            Thrusts(i) = k_neg * n_i * fabs(n_i);
        }
        else {
            std::cout << "Warning: n_r[" << i << "] outside expected interval [0, 1] with value: " << n_r(i) << std::endl;
            std::cout << "Returning 0 value for Thrust" << std::endl;
            Thrusts.setZero();
            return Thrusts;
        }
    }

    return Thrusts;
}
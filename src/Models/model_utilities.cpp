#include "Models/model_utilities.hpp"       
#include "Utilities/calculations.hpp"     
#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <fstream>
#include <casadi/casadi.hpp>

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

//------------------------------------------------------------------------------
// Helper function: m2c (added-mass to Coriolis matrix)
//   M  : 6×6 or 3×3 inertia (rigid‐body MRB or added‐mass MA) matrix
//   nu : 6×1 [u v w p q r]^T or 3×1 [u v r]^T velocity vector
// returns
//   C  : Coriolis‐centripetal matrix (6×6 or 3×3)
//------------------------------------------------------------------------------
Eigen::MatrixXd m2c(const Eigen::MatrixXd& M_in,
    const Eigen::VectorXd& nu)
{
    // 1) Symmetrize M
    Eigen::MatrixXd M = 0.5 * (M_in + M_in.transpose());

    // 2) 6-DOF branch
    if (nu.size() == 6) {
        assert(M.rows() == 6 && M.cols() == 6);

        Eigen::Matrix3d M11 = M.block<3,3>(0,0);
        Eigen::Matrix3d M12 = M.block<3,3>(0,3);
        Eigen::Matrix3d M21 = M12.transpose();
        Eigen::Matrix3d M22 = M.block<3,3>(3,3);

        Eigen::Vector3d nu1 = nu.segment<3>(0);
        Eigen::Vector3d nu2 = nu.segment<3>(3);

        Eigen::Vector3d nu1_dot = M11*nu1 + M12*nu2;
        Eigen::Vector3d nu2_dot = M21*nu1 + M22*nu2;

        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6,6);
        C.block<3,3>(0,3) = -Smtrx(nu1_dot);
        C.block<3,3>(3,0) = -Smtrx(nu1_dot);
        C.block<3,3>(3,3) = -Smtrx(nu2_dot);
        return C;
    }
    // 3) 3-DOF branch (surge, sway, yaw)
    else {
        assert(nu.size() == 3);
        assert(M.rows() == 3 && M.cols() == 3);

        double u = nu(0), v = nu(1), r = nu(2);
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

        C(0,2) = - ( M(1,1)*v + M(1,2)*r );
        C(1,2) =   ( M(0,0)*u );
        C(2,0) =   ( M(1,1)*v + M(1,2)*r );
        C(2,1) = - ( M(0,0)*u );
        return C;
    }
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

// Calculates real propeller revs based on relative propellar revs (n)
std::vector<double> nReal(std::vector<double> n_relative) {
    // Stub: returns zeros
    double n1_real = 0 * n_relative[0]; 
    double n2_real = 0 * n_relative[1]; 
    return {n1_real, n2_real};
}

// Calculating thrusts based on relative propellar revs (n)
Eigen::VectorXd ThrustsFromRealativeN(Eigen::Vector2d n_r, Eigen::VectorXd coeffs) {
    double g = 9.81;  // gravity to convert kg to Newtons

    int n_r_size = n_r.size();
    int degree = coeffs.size();  // polynomial degree (no constant term)
    Eigen::VectorXd Thrusts(n_r_size);
    Thrusts.setZero();

    for (int i = 0; i < n_r_size; ++i) {
        double n_i = n_r(i);

        if (n_i < -1.0 || n_i > 1.0) {
            std::cout << "Warning: n_r[" << i << "] outside expected interval [-1, 1] with value: " << n_i << std::endl;
            std::cout << "Returning 0 value for Thrust" << std::endl;
            Thrusts.setZero();
            return Thrusts;
        }

        // Clamp n_i to the range [-0.75, 0.75] because the polynomial
        // approximation is only valid in this range.
        n_i = std::min(std::max(n_i, -0.75), 0.75);

        double thrust_kg = 0.0;
        for (int power = degree; power >= 1; --power) {
            // coeff index: 0 corresponds to x^degree, 1 to x^(degree-1), ...
            thrust_kg += coeffs(degree - power) * std::pow(n_i, power);
        }

        // Discount because thrust was measured with both propellers at the same time,
        // not one at a time, in the bollard pull test.
        // Assuming no thruster interaction effects up to 75% of thrust signal.
        double discountFactor = 0.5; //1 / (2*(1-0.363));  

        //Thrust force in Newton
        Thrusts(i) = discountFactor * (g * thrust_kg);
    }

    return Thrusts;
}

casadi::MX ThrustFromRelativeN_MX(casadi::MX n_i) {
   
    double g = 9.81;
    double discountFactor = 0.5; 
    n_i = fmin(fmax(n_i, -0.75), 0.75);

    Eigen::VectorXd coeffs = NOrderApprox("../data/bollard_pull_data.csv", 5);

    casadi::MX thrust_kg = coeffs(0)*pow(n_i,5) +
                           coeffs(1)*pow(n_i,4) +
                           coeffs(2)*pow(n_i,3) +
                           coeffs(3)*pow(n_i,2) +
                           coeffs(4)*n_i;

    // Convert to Newtons and apply discount
    return discountFactor * g * thrust_kg;
}

Eigen::VectorXd NOrderApprox(const std::string& csv_file, int order) {
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + csv_file);
    }

    std::vector<double> thrust;
    std::vector<double> kg;

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) { // skip header
            first_line = false;
            continue;
        }

        // Find comma location
        size_t comma_pos = line.find(',');
        if (comma_pos == std::string::npos) {
            throw std::runtime_error("Malformed CSV line (no comma): " + line);
        }

        // Extract substrings for thrust and kg
        std::string t_str = line.substr(0, comma_pos);
        std::string k_str = line.substr(comma_pos + 1);

        // Convert strings to double
        double t = std::stod(t_str);
        double k = std::stod(k_str);

        thrust.push_back(t);
        kg.push_back(k);
    }
    file.close();

    int n = thrust.size();
    if (n == 0) {
        throw std::runtime_error("No data found in CSV file");
    }

    // Design matrix for polynomial: columns are x^N, x^(N-1), ..., x^1 (no constant term)
    Eigen::MatrixXd X(n, order);
    Eigen::VectorXd Y(n);

    for (int i = 0; i < n; ++i) {
        double x = thrust[i];
        for (int power = order; power >= 1; --power) {
            X(i, order - power) = std::pow(x, power);
        }
        Y(i) = kg[i];
    }

    // Solve least squares: (XᵀX)^-1 Xᵀ Y
    Eigen::VectorXd coeffs = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

    return coeffs;  // N coefficients for x^N down to x^1
}
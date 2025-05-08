#include <Eigen/Dense>
#include <cmath>

std::vector<double> pseudo_inverse_allocation(
    const std::vector<double>& tau_XYN,
    const Eigen::MatrixXd& B,
    double k_pos,
    double k_neg)
{
    Eigen::Vector3d tau;
    tau << tau_XYN[0], tau_XYN[1], tau_XYN[2];

    // 1) Compute Moore–Penrose pseudo-inverse of B:
    //    B^+ = B^T * (B * B^T)^(-1)
    Eigen::Matrix3d BBt = B * B.transpose();       // 3×3
    Eigen::Matrix3d BBt_inv = BBt.inverse();       // (B B^T)^(-1)
    Eigen::MatrixXd Bplus = B.transpose() * BBt_inv; // (cols_B)×3

    // 2) Solve for “extended” forces u_ext = B^+ * tau
    Eigen::VectorXd u_ext = Bplus * tau;  // 4×1 if B is 3×4

    // 3) Split into x/y components
    double u1x = u_ext(0), u1y = u_ext(1);
    double u2x = u_ext(2), u2y = u_ext(3);

    // 4) Recover azimuth angles
    double alpha1 = std::atan2(u1y, u1x);
    double alpha2 = std::atan2(u2y, u2x);
    //    (Assume alpha_i in [–pi/2, +pi/2] so cos(alpha_i) >= 0)

    // 5) Choose the correct gain based on the sign of u_i_x
    double k1 = (u1x >= 0 ? k_pos : k_neg);
    double k2 = (u2x >= 0 ? k_pos : k_neg);

    // 6) Back-substitute for speeds: u_i_x = k_i * n_i * cos(alpha_i)
    double n1 = u1x / (k1 * std::cos(alpha1));
    double n2 = u2x / (k2 * std::cos(alpha2));

    return { n1, n2, alpha1, alpha2 };
}


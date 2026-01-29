#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>

// tau_XYN:      {τX, τY, τN}
// B:            3×4 geometry‐only matrix T_e
// k_pos, k_neg: thrust coefficients for forward (n>=0) / reverse (n<0)
// returns:      {n1, alpha1, n2, alpha2}
std::vector<double> pseudo_inverse_allocation(
    const std::vector<double>& tau_XYN,
    const Eigen::MatrixXd& B,
    double k_pos,
    double k_neg)
{
    assert(tau_XYN.size()==3);
    assert(B.rows()==3 && B.cols()==4);

    // 1) Build τ
    Eigen::Vector3d tau(tau_XYN[0], tau_XYN[1], tau_XYN[2]);

    // 2) Compute B⁺ = Bᵀ (B Bᵀ)⁻¹ → 4×3
    Eigen::Matrix3d BBt    = B * B.transpose();
    Eigen::Matrix3d BBtInv = BBt.inverse();
    Eigen::Matrix<double,4,3> Bplus = B.transpose() * BBtInv;

    // 3) Solve for extended forces [u1x,u1y,u2x,u2y]ᵀ
    Eigen::Vector4d u_ext = Bplus * tau;
    double u1x = u_ext(0), u1y = u_ext(1);
    double u2x = u_ext(2), u2y = u_ext(3);

    // 4) Compute the total thrust magnitude and raw angle:
    double F1    = std::hypot(u1x, u1y);
    double alpha1 = std::atan2(u1y, u1x);  // in (−π,π]

    // 5) Clamp alpha into [−π/2,π/2] and record flip s1:
    double s1 = +1.0;
    if      (alpha1 >  M_PI_2) { alpha1 -= M_PI; s1 = -1; }
    else if (alpha1 < -M_PI_2) { alpha1 += M_PI; s1 = -1; }

    // 6) Pick the correct thrust coefficient:
    double k1 = (s1 > 0 ? k_pos : k_neg);

    // 7) Solve F1 = k1 * n1 * |n1|  ⇒  |n1| = sqrt(F1/k1), then re‐apply sign:
    double n1 = s1 * std::sqrt(std::abs(F1) / k1);

    // 8) Repeat for thruster 2:
    double F2    = std::hypot(u2x, u2y);
    double alpha2 = std::atan2(u2y, u2x);
    double s2 = +1.0;
    if      (alpha2 >  M_PI_2) { alpha2 -= M_PI; s2 = -1; }
    else if (alpha2 < -M_PI_2) { alpha2 += M_PI; s2 = -1; }
    double k2 = (s2 > 0 ? k_pos : k_neg);
    double n2 = s2 * std::sqrt(std::abs(F2) / k2);

    // 9) Scale the thrusts such that |n| <= 1:
    double max_n = std::max({std::abs(n1), std::abs(n2), 1.0});
    if (max_n > 1.0) {
        // pull everything back proportionally
        std::vector<double> tau_scaled = tau_XYN;
        tau_scaled[0] /= max_n;
        tau_scaled[1] /= max_n;
        tau_scaled[2] /= max_n;
      
        // Recompute
        return pseudo_inverse_allocation(tau_scaled, B, k_pos, k_neg);
      }

    // 8) Return in the order {n1, alpha1, n2, alpha2}
    return { n1, alpha1, n2, alpha2 };
}

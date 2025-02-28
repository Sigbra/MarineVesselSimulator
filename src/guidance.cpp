#include "guidance.hpp"


// ---------------------------------------------------
// DP controller function
// ---------------------------------------------------
void dynamicPositioning(const Waypoints &wpt,
    const Eigen::VectorXd &x,
    int &currentWptIndex,
    double &tau_X,
    double &tau_Y,
    double &tau_N,
    double &r_d,
    double &psi_d,
    double &z_psi,
    double h)
{
    // Controller gains (tune to your system)
    static double Kp_xy = 50.0;   // proportional gain for surge & sway
    static double Kd_xy = 40.0;   // derivative gain for surge & sway

    static double Kp_psi = 30.0;  // proportional gain for yaw
    static double Ki_psi = 10.0;  // integral gain for yaw
    static double Kd_psi = 10.0;  // derivative gain for yaw

    // Extract current states
    double x_pos = x(6);    // North position
    double y_pos = x(7);    // East position
    double psi    = x(11);  // heading
    double u      = x(0);   // surge velocity
    double v      = x(1);   // sway velocity
    double r      = x(5);   // yaw rate

    double x_d_check = wpt.x[currentWptIndex];
    double y_d_check = wpt.y[currentWptIndex];

    // Distance to current waypoint
    double e_x_check = x_d_check - x_pos;
    double e_y_check = y_d_check - y_pos;
    double dist      = std::sqrt(e_x_check * e_x_check + e_y_check * e_y_check);

    
    double switchDist = 5.0;  // [m], threshold for switching
    int nWpts = static_cast<int>(wpt.x.size());
    if (dist < switchDist && (currentWptIndex < (nWpts - 1) && currentWptIndex <= nWpts)) {
        currentWptIndex++;  // move to the next waypoint
    }

    // Desired position/heading from current waypoint
    double x_d = wpt.x[currentWptIndex];
    double y_d = wpt.y[currentWptIndex];
    psi_d      = wpt.angle[currentWptIndex];  // Desired heading

    // Compute errors
    double e_x   = x_d - x_pos;
    double e_y   = y_d - y_pos;
    double e_psi = ssa(psi_d - psi);   // ssa() ensures error is in (-pi, pi)

    // For DP, we want no velocity or rate:
    double u_d = 0.0;
    double v_d = 0.0;
    r_d        = 0.0;

    // --- PD in surge and sway ---
    // Negative derivative term uses actual velocity
    tau_X = Kp_xy * e_x + (-Kd_xy) * (u - u_d);
    tau_Y = Kp_xy * e_y + (-Kd_xy) * (v - v_d);

    // --- PID in yaw ---
    tau_N = Kp_psi * e_psi + Ki_psi * z_psi + (-Kd_psi) * (r - r_d);
}

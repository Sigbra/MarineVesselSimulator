#ifndef REFMODEL_HPP
#define REFMODEL_HPP

/**
 * @brief Reference model for propagating desired position, velocity, and acceleration.
 *
 * Based on Fossen (2021, Chapter 12.1.1), this function computes the desired
 * acceleration, velocity, and position at the next time step.
 *
 * @param x_d        [in,out] Current desired position at time t_k.
 *                   Updated to desired position at t_k+1.
 * @param v_d        [in,out] Current desired velocity at time t_k.
 *                   Updated to desired velocity at t_k+1.
 * @param a_d        [in,out] Current desired acceleration at time t_k.
 *                   Updated to desired acceleration at t_k+1.
 * @param x_ref      [in] Commanded (reference) position.
 * @param v_max      [in] Maximum allowed velocity.
 * @param zeta_d     [in] Desired relative damping factor.
 * @param w_d        [in] Desired natural frequency (rad/s).
 * @param h          [in] Sampling time (s).
 * @param eulerAngle [in] If true, x_d is an Euler angle and the error is computed
 *                     as the smallest signed angle difference.
 */
void refModel(double &x_d, double &v_d, double &a_d,
              double x_ref, double v_max, double zeta_d,
              double w_d, double h, bool eulerAngle);

#endif // REFMODEL_HPP

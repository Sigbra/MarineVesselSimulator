#ifndef MPC_GUIDANCE_HPP
#define MPC_GUIDANCE_HPP
 
#include <vector>
#include <tuple>
#include <casadi/casadi.hpp>
#include "Utilities/calculations.hpp"

using namespace casadi;

class MPCGuidance {
    public:

        MPCGuidance(double r_max, double U_dot_max);

        void update(double h, double x, double y, double chi, double U,
                    Vector2D wpt_prev, Vector2D wpt_goal);
        
        double get_chi_d(int idx) const;
        double get_U_d(int idx) const;
        double get_X_d(int idx) const;
        double get_Y_d(int idx) const;

    private:
        MX f(const MX& X, const MX& C);
        MX rk4(const MX& Xk, const MX& Ck, const MX& dt);

        double r_max_;
        double U_dot_max_;
        double step_length_ = 1; 

        std::vector<double> chi_d_;
        std::vector<double> U_d_;
        std::vector<double> X_d_;
        std::vector<double> Y_d_;
};
    


// class MPCGuidance {
// public:
//     MPCGuidance(double h, double r_max, double U_dot_max, double T_final);

//     std::tuple<double, double, bool> update(double x, double y, double chi, double U, Vector2D wp1, Vector2D wp2);
 
//     void reset();

//     double getFirstChi() const { return chi_d_trajectory_[1]; }
//     double getFirstSpeed() const { return U_d_trajectory_[1]; }
//     double getFirstX() const { return X_d_trajectory_[1]; }
//     double getFirstY() const { return Y_d_trajectory_[1]; }
 
// private:
//     void initializeSolver();

//     double h_;              // Sampling time
//     double r_max_;          // Maximum turning rate [rad/s]
//     double U_dot_max_;      // Maximum acceleration [m/s^2]
//     double T_final_;        // Final time [s]
//     int N_;                 // Number of discretization steps

//     // Solution trajectories
//     std::vector<double> chi_d_trajectory_;
//     std::vector<double> U_d_trajectory_;
//     std::vector<double> X_d_trajectory_;
//     std::vector<double> Y_d_trajectory_;

//     // Solver and related variables
//     casadi::Function solver_;
//     bool solver_initialized_;
// };

#endif // MPC_GUIDANCE_HPP


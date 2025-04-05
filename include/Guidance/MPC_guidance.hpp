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

        std::tuple<double, double, double, double> update(double h, double x, double y, double chi, double U,
                                                          Vector2D wpt_prev, Vector2D wpt_goal);
        
        std::vector<double> get_chi_d() const {return chi_d_;};
        std::vector<double> get_U_d() const { return U_d_;};
        std::vector<double> get_X_d() const { return X_d_;};
        std::vector<double> get_Y_d() const {return Y_d_;};

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

#endif // MPC_GUIDANCE_HPP


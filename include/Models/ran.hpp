// ran.hpp
#ifndef RAN_HPP
#define RAN_HPP

#include <Eigen/Dense>

//-------------------------------------------------------------------
// The ran() function
//
// Inputs:
//   x      - 12x1 state vector: [u, v, w, p, q, r, x, y, z, phi, theta, psi]'
//   n      - 2x1 propeller speeds [n_left, n_right] (rad/s)
//   mp     - payload mass (kg)
//   rp     - 3x1 payload location (m)
//   V_c    - ocean current speed (m/s)
//   beta_c - ocean current direction (rad)
//   alpha  - 2x1 azimuth angles [alpha_left, alpha_right] (rad)
// Outputs (by reference):
//   xdot      - 12x1 time derivative of state vector
//   U         - Speed (m/s) computed as sqrt(u^2+v^2)
//   M_out     - 6x6 system mass matrix (MRB + added mass)
//   B    - 3x2 propeller input matrix 
//-------------------------------------------------------------------

class RAN {
    public:
        RAN();
        // Function to compute dynamics (ran)
        void update(const Eigen::VectorXd x, double mp, double V_c, double beta_c, 
                double h, Eigen::Vector2d n, Eigen::Vector2d alpha);

        // Specialized RK4 integrator for the RAN model
        void rk4(Eigen::VectorXd& x, double mp, double V_c, double beta_c, double h, Eigen::Vector2d n, Eigen::Vector2d alpha);

        void update_n(Eigen::Vector2d& n, Eigen::Vector2d n_c, double h);
        void update_alpha(Eigen::Vector2d& alpha, Eigen::Vector2d alpha_c, double h);
        
        Eigen::VectorXd get_xdot() const { return xdot; }
        Eigen::MatrixXd get_M() const { return M; }
        Eigen::MatrixXd get_B() const { return B; }
        double get_U() const { return U; }

        double getT_n() const { return T_n; }
        double getT_alpha() const { return T_alpha; }

        void fail_state_n1() { n1_fail = true; }
        void fail_state_n2() { n2_fail = true; }
        void recover_n1() { n1_fail = false; }
        void recover_n2() { n2_fail = false; }
        
        void select_failure_mode();
        std::vector<bool> check_failstate();

    private:

        bool propagate;

        double T_n;
        double T_alpha;

        double U;
        Eigen::MatrixXd M;
        Eigen::MatrixXd B;
        Eigen::VectorXd xdot;

        Eigen::VectorXd xdot_rk4;

        // - gravitational acceleration (m/s^2)
        double g;                        
        // - water density (kg/m^3)
        double rho;                     
        // - vessel length (m)
        double L;                         
        // - vessel beam (m)
        double Beam;                           
        // - vessel mass (kg)
        double m;          

        // - radii of gyration in roll
        double R44;                  
        // - in pitch
        double R55;                    
        // - in yaw
        double R66;        

        // - surge time constant (s)
        double T_surge;
        // - sway time constant (s)
        double T_sway;                      
        // - yaw time constant (s)
        double T_yaw;          

        // - maximum forward speed (m/s)
        double Umax;                        

        // Data for one pontoon
        // - pontoon beam (m)
        double Beam_pont;                 
        // - lateral offset from centerline (m)
        double y_pont;                    
        // - waterline area coefficient
        double Cw_pont;                       
        // - block coefficient
        double Cb_pont;

        // Positive Bollard
        double k_pos;        
        // Negative Bollard
        double k_neg;       
        // relative propellar speed max
        double n_max;           
        // relative propellar speed min
        double n_min;           
        // maximum azimuth angle (rad)
        double alpha_max;   
        // minimum azimuth angle (rad)
        double alpha_min; 

        // Azimuth pod placement from CO / lever arms (m)
        double ly1;         
        double ly2;        
        double lx;

        // Failure state flags (true => proppeller speed 0)
        bool n1_fail;
        bool n2_fail;

        Eigen::VectorXd thrust_coeffs;
};
                
#endif // RAN_HPP

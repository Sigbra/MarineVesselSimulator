// ran.hpp
#ifndef RAN_HPP
#define RAN_HPP

#include <Eigen/Dense>
#include <random>
#include <vector>  // minimal add: needed for std::vector<bool> in API

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

        Eigen::VectorXd tau_pods(Eigen::Vector2d n, Eigen::Vector2d alpha);
        
        Eigen::VectorXd get_xdot() const { return xdot; }
        Eigen::MatrixXd get_M() const { return M; }
        Eigen::MatrixXd get_B() const { return B; }
        double get_U() const { return U; }

        double getT_n() const { return T_n; }
        double getT_alpha() const { return T_alpha; }
        Eigen::VectorXd getThrustCoeffs() const {return thrust_coeffs; }

        void fail_state_n1() { n1_fail = true; }
        void fail_state_n2() { n2_fail = true; }
        void recover_n1() { n1_fail = false; }
        void recover_n2() { n2_fail = false; }
        
        void select_failure_mode();
        std::vector<bool> check_failstate();

        // --- Wave API (public so your sim/IO can read & tune) ---
        void enable_waves(bool on) { waves_on = on; }

        void set_wave_params(      // quick bulk setter
            double wn_u, double z_u, double K_u, double sigw_u,
            double wn_v, double z_v, double K_v, double sigw_v,
            double wn_r, double z_r, double K_r, double sigw_r,
            double Td_drift, double sigw_X, double sigw_Y, double sigw_N);

        // NEW: 6-DOF setter (x,y,z,φ,θ,ψ) + drift
        void set_wave_params6(
            double wn_x, double z_x, double K_x, double sigw_x,     // x (surge)
            double wn_y, double z_y, double K_y, double sigw_y,     // y (sway)
            double wn_z, double z_z, double K_z, double sigw_z,     // z (heave)
            double wn_phi, double z_phi, double K_phi, double sigw_phi,   // roll
            double wn_theta, double z_theta, double K_theta, double sigw_theta, // pitch
            double wn_psi, double z_psi, double K_psi, double sigw_psi,   // yaw
            double Td_drift, double sigw_X, double sigw_Y, double sigw_N);
            
        // Wave outputs (END frame), full 6-DOF
        const Eigen::Matrix<double,6,1>& get_wave_eta6()  const { return eta_w_6; }   // [x y z φ θ ψ]
        const Eigen::Matrix<double,6,1>& get_wave_rate6() const { return etadot_w_6; }
        const Eigen::Matrix<double,6,1>& get_wave_acc6()  const { return etaddot_w_6; }

        // Back-compat 3-vector getters commonly used in your code
        const Eigen::Vector3d& get_wave_eta_EN()  const { return eta_w_EN; }     // [x y ψ]
        const Eigen::Vector3d& get_wave_rate_EN() const { return etadot_w_EN; }  // [ẋ ẏ ψ̇]
        const Eigen::Vector3d& get_wave_acc_EN()  const { return etaddot_w_EN; } // [ẍ ÿ ψ̈]

        // Drift (BODY frame) forces: [X, Y, N]
        const Eigen::Vector3d& get_wave_body_drift() const { return d_wave; }

        // Advance wave filters (call once per loop *before* reading the getters)
        void wave_step_WF(double dt);     // WF motions (6-DOF, END frame)
        void wave_step_drift(double dt);  // Drift forces (BODY frame)

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
        double ly1_o;         
        double ly2_o;        
        double lx_o;
        double pod_radius;

        // Failure state flags (true => proppeller speed 0)
        bool n1_fail;
        bool n2_fail;

        Eigen::VectorXd thrust_coeffs;

        // -------------------- Wave parameters --------------------
        bool waves_on{true};

        // 6-DOF WF parameters (one per DOF: x,y,z,φ,θ,ψ)
        std::array<double,6> wn  {{0.8, 0.8, 1.0, 0.8, 0.8, 0.8}};
        std::array<double,6> zeta{{0.25,0.25,0.40,0.20,0.20,0.20}};
        std::array<double,6> Kg  {{0.50,0.60,0.40,0.03,0.03,0.03}};
        std::array<double,6> sig{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

        // Drift: time constant and white-noise intensities (BODY frame)
        double Td{120.0};
        double sigw_X{40.0}, sigw_Y{40.0}, sigw_N{400.0};

        // --------------- Wave outputs (cached this step) ---------------
        // Full 6-DOF (END frame)
        Eigen::Matrix<double,6,1> eta_w_6     = Eigen::Matrix<double,6,1>::Zero();
        Eigen::Matrix<double,6,1> etadot_w_6  = Eigen::Matrix<double,6,1>::Zero();
        Eigen::Matrix<double,6,1> etaddot_w_6 = Eigen::Matrix<double,6,1>::Zero();

        // Back-compat 3-vectors used elsewhere: [x y ψ]
        Eigen::Vector3d eta_w_EN    {Eigen::Vector3d::Zero()};
        Eigen::Vector3d etadot_w_EN {Eigen::Vector3d::Zero()};
        Eigen::Vector3d etaddot_w_EN{Eigen::Vector3d::Zero()};

        // Drift (BODY frame)
        Eigen::Vector3d d_wave{Eigen::Vector3d::Zero()};               // [X_d, Y_d, N_d]
        Eigen::Vector3d tau_wave_body_cached{Eigen::Vector3d::Zero()};  // same for this step

        // Internal WF states: one (ξ, ξ̇) pair per DOF
        std::array<Eigen::Vector2d,6> xw{
            Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(),
            Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero()
        };

        // ---- RNG (repeatable unless reseeded) ----
        std::mt19937_64 rng{1234567ULL};
        std::normal_distribution<double> N01{0.0,1.0};

};
                
#endif // RAN_HPP

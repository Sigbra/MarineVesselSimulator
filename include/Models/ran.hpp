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
        Eigen::Vector3d get_wave_eta_EN() const { return eta_w_EN; }   // [xw,yw,psw]
        Eigen::Vector3d get_wave_body_drift() const { return d_wave; } // [X,Y,N]

        // minimal add: you use this in main to get wave rates
        Eigen::Vector3d get_wave_rate_EN() const { return etadot_w_EN; } // [ẋw,ẏw,ψ̇w]

        // minimal change: make WF step public so main can call ran_model.wave_step_WF(h)
        void wave_step_WF(double dt);      // updates xw_* and eta_w_EN (+ etadot_w_EN)

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

        // ---- Wave parameters ----
        bool   waves_on{true};
        // WF natural freqs, damping, gains, and driving intensities
        double wn_u{0.8}, z_u{0.25}, K_u{0.5}, sigw_u{1.0};
        double wn_v{0.8}, z_v{0.25}, K_v{0.6}, sigw_v{1.0};
        double wn_r{0.8}, z_r{0.20}, K_r{0.03}, sigw_r{1.0};
        // Drift: time constant and white-noise intensities (per axis)
        double Td{120.0}, sigw_X{40.0}, sigw_Y{40.0}, sigw_N{400.0};

        // ---- Outputs cached this step ----
        Eigen::Vector3d eta_w_EN{Eigen::Vector3d::Zero()};   // measurement offsets
        Eigen::Vector3d tau_wave_body_cached{Eigen::Vector3d::Zero()}; // body forces for update()
        // minimal add: rate of wave outputs (used by your IMU yaw-rate addition)
        Eigen::Vector3d etadot_w_EN{Eigen::Vector3d::Zero()}; // [ẋw,ẏw,ψ̇w]

        // minimal add: internal WF states per DOF (needed by wave_step_WF impl)
        Eigen::Vector2d xw_u{Eigen::Vector2d::Zero()}; // [ξ, ηw] surge
        Eigen::Vector2d xw_v{Eigen::Vector2d::Zero()}; // [ξ, ηw] sway
        Eigen::Vector2d xw_r{Eigen::Vector2d::Zero()}; // [ξ, ηw] yaw

        // minimal add: BODY drift state (used by get_wave_body_drift)
        Eigen::Vector3d d_wave{Eigen::Vector3d::Zero()}; // [X_d, Y_d, N_d]

        // ---- RNG (repeatable unless reseeded) ----
        std::mt19937_64 rng{1234567ULL};
        std::normal_distribution<double> N01{0.0,1.0};

        // ---- Internal wave helpers (called from rk4) ----
        void wave_step_drift(double dt);   // updates d_wave and tau_wave_body_cached
        // wave_step_WF(dt) is declared public above (no duplicate here)
};
                
#endif // RAN_HPP

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>    // For getenv

#include "utilities.hpp"
#include "ran.hpp"
//#include "ref_model.hpp"
//#include "ALOSpsi.hpp"
//#include "hermite_spline.hpp"
//#include "crosstrack_hermite.hpp"
//#include "los_observer.hpp"
#include "guidance.hpp"
#include "motion_control.hpp"
#include "control_allocation.hpp"


int main() {
    srand(time(0)); // Randomize seed

    // USER INPUTS
    double h = 0.05;               // Sampling time [s]
    double T_final = 500;         // Final simulation time [s]
 
    //Load condition
    double mp = 0;                 // Payload mass [kg]
    Eigen::Vector3d rp(1, 0, 0);   // Payload location [m]
    
    // Ocean current
    double V_c = 0.0; //0.3;       // Ocean current speed (m/s)
    double beta_c = deg2rad(30.0); // Ocean current direction (rad)

    // Waypoints: position + heading angles for DP control
    Waypoints wpt;
    wpt.x =     {0,           0,         50,          50,            0};
    wpt.y =     {0,          50,         50,           0,            0};
    wpt.angle = {0, deg2rad(90), deg2rad(45), deg2rad(90), deg2rad(-45)}; 
    
    // Additional parameter for straight-line path following
    double R_switch = 5;
     
    // Initial heading
    double psi0 = atan2(wpt.y[1] - wpt.y[0], wpt.x[1] - wpt.x[0]);
    
    //Calculating RAN USV input Matrix
    Eigen::VectorXd x_input = Eigen::VectorXd::Zero(12);  
    Eigen::VectorXd n_input = Eigen::VectorXd::Zero(2); 
    Eigen::VectorXd alpha_input = Eigen::VectorXd::Zero(2);                    
    Eigen::VectorXd xdot = Eigen::VectorXd::Zero(12); 
    double U = 0;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 6); 
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 4); 
    
    ran(x_input, n_input, alpha_input, mp, rp, V_c, beta_c, xdot, U, M, B);

    // Azimuth pod dynamics
    double T_n = 0.1;                               // Propeller time constant (s)
    Eigen::Vector2d n;                              // Initial propeller speed, [n_left n_right]'
    n << 0, 0;                                      // Initial values for propeller speeds

    double T_alpha = 0.1;                           // Azimuth angle time constant (s)
    Eigen::Vector2d alpha;                          // Initial azimuth angles, [alpha_left alpha_right]'
    alpha << deg2rad(0.0), deg2rad(0.0);            // Initial values for azimuth angles

    // Initial states and variables
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12);  // x = [u v w p q r xn yn zn phi theta psi]'
    x(6)  = wpt.x[0];                               // Initial x position
    x(7)  = wpt.y[0];                               // Initial y position
    x(11) = psi0;                                   // Initial heading angle

    // Control method selection
    GuidanceMethod guidance;
    int GuidanceFlag = guidance.selectMethod();

    ControlAllocationMethod controlAlloc;
    int ControlAllocFlag = controlAlloc.selectMethod();

    // Current waypoint
    int wpt_index  = 1; //First waypoint (index 0) is only used for initial heading
    
    // Disired state in NED
    double xn_d    = wpt.x[0];        
    double yn_d    = wpt.y[0];        
    double psi_d   = psi0;         

    // Control forces and moment
    double tau_X   = 0.0;         // desired surge force
    double tau_Y   = 0.0;         // desired sway force
    double tau_N   = 0.0;         // desired yaw moment

    // States for PID Motion Control
    double z_xn    = 0.0;         // integral state for surge
    double z_yn    = 0.0;         // integral state for sway
    double z_psi   = 0.0;         // integral state for heading
    double prev_error_xn  = 0.0;  // previous surge error
    double prev_error_yn  = 0.0;  // previous sway error
    double prev_error_psi = 0.0;  // previous heading error

    int num_steps = static_cast<int>(T_final / h) + 1; // Total number of time steps
    std::vector<double> t(num_steps);                  // Time vector from 0 to T_final

    Eigen::MatrixXd simdata_vessel_states(num_steps, 12 + 4);   
    Eigen::MatrixXd simdata_state_errors(num_steps, 4);        
    
    for (int i = 0; i < num_steps; ++i) {
        t[i] = i * h;
        
        // Navigation (Fake measurements using noise)
        double random = ((double)rand() / RAND_MAX - 0.5);
  
        double u     = x(0)  +  0.01 * random; //Surge velocity (BODY frame)
        double v     = x(1)  +  0.01 * random; //Sway velocity  (BODY frame)
        double w     = x(2)  +  0.01 * random; //Heave velocity (BODY frame)
        double p     = x(3)  + 0.001 * random; //Roll rate      (BODY frame)
        double q     = x(4)  + 0.001 * random; //Pitch rate     (BODY frame)
        double r     = x(5)  + 0.001 * random; //Yaw rate       (BODY frame)
    
        double xn    = x(6)  +  0.01 * random; //North position  (NED frame)
        double yn    = x(7)  +  0.01 * random; //East position   (NED frame)
        double zn    = x(8)  +  0.01 * random; //Down position   (NED frame)
        double phi   = x(9)  + 0.001 * random; //Roll angle      (NED frame)
        double theta = x(10) + 0.001 * random; //Pitch angle     (NED frame)
        double psi   = x(11) + 0.001 * random; //Heading angle   (NED frame)

        if (std::isnan(psi)) {
            std::cerr << "NaN detected for psi at iteration " << i << ", time: " << t[i] << "s\n";
            break; 
        }

        // Guidance
        //  Waypoint Update
        double pos_x_error = wpt.x[wpt_index] - xn;
        double pos_y_error = wpt.y[wpt_index] - yn;
        double pos_error_BODY   = sqrt(pow(pos_x_error, 2) + pow(pos_y_error, 2));

        if (pos_error_BODY < R_switch && wpt_index < wpt.x.size() - 1) {
            wpt_index = wpt_index + 1;
            std::cout << "Waypoint reached, next waypoint: (" << wpt.x[wpt_index] <<", "<< wpt.y[wpt_index] << ")" << std::endl;
        }

        switch (GuidanceFlag) {
            case 1: {
                // Station keeping
                std::vector<double> desired_states = StationKeeping(wpt, wpt_index);
                xn_d  = desired_states[0]; 
                yn_d  = desired_states[1]; 
                psi_d = desired_states[2]; 
                break;
            }
            case 2: {
                // Dynamic Positioning
                std::vector<double> desired_states = DynamicPositioning(wpt, wpt_index);
                xn_d  = desired_states[0];
                yn_d  = desired_states[1];
                psi_d = desired_states[2];
                break;
            }
        }

        // Motion Control System (calculate forces and moments from desired states)
        std::vector<double> tau = SISO_linear_PID_Control( 
            h,
            xn_d, yn_d, psi_d,
            xn, yn, psi,
            z_xn, z_yn, z_psi,
            prev_error_xn, prev_error_yn, prev_error_psi);

        // Control Allocation 
        tau_X = tau[0];
        tau_Y = tau[1];
        tau_N = tau[2];
        Eigen::Vector2d n_c = {0.0, 0.0};
        Eigen::Vector2d alpha_c = {0.0, 0.0};

        switch (ControlAllocFlag){
            case 1: {
                std::vector<double> control_allocation = NLOptControlAlloc(tau_X, tau_Y, tau_N);
                n_c     = {control_allocation[0], control_allocation[2]};
                alpha_c = {control_allocation[1], control_allocation[3]};
                break;
            }
            case 2: {
                //MPC
                break;
            }
        }                      

        // Marine Craft Model
        //  rk4 method for x(k+1)
        rk4_ran_step(x, n, alpha, mp, rp, V_c, beta_c, h);

        //  Euler's method
        n = n + h/T_n * (n_c - n);                             // Update propeller speeds
        alpha = alpha + h/T_alpha * (alpha_c - alpha);         // Update azimuth angles

        z_xn  = z_xn  + h * (pos_x_error);      // Update integral state for surge control
        z_yn  = z_yn  + h * (pos_y_error);      // Update integral state for sway control
        z_psi = z_psi + h * ssa(psi_d - psi);   // Update integral state for heading control

        // Show SIM progress
        if (i % 100 == 0) {
            std::cout << "Iteration:  " << i << ", Time: " << t[i] << "s, x: " << xn << "m, y: " << yn << "m, psi: " << rad2deg(psi) << "deg" << std::endl;
            std::cout << "proppellar: " << n(0)              << ", " <<n(1)              << std::endl;
            std::cout << "alpha:      " << rad2deg(alpha(0)) << ", " <<rad2deg(alpha(1)) << std::endl;
            std::cout << " " << std::endl;
        }

        // Storing SIM data
        simdata_vessel_states(i, Eigen::seq(0, 11)) = x.transpose();  
        simdata_vessel_states(i, 12) = n_c(0);                           
        simdata_vessel_states(i, 13) = n_c(1);
        simdata_vessel_states(i, 14) = alpha_c(0);
        simdata_vessel_states(i, 15) = alpha_c(1); 

        simdata_state_errors(i, 0) = t[i];
        simdata_state_errors(i, 1) = xn_d  - xn;
        simdata_state_errors(i, 2) = yn_d  - yn;
        simdata_state_errors(i, 3) = psi_d - psi;


    }
    std::cout<<"Simulation completed"<<std::endl;
    storeSimulationData(simdata_vessel_states, "simdata_vessel_states.csv");
    storeSimulationData(simdata_state_errors, "simdata_state_errors.csv");

    plotTrajectory();
    plotStateErrors();

    return 0;
}



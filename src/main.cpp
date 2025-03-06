#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <yaml-cpp/yaml.h>

#include "ran.hpp"
#include "guidance.hpp"
#include "utilities.hpp"
#include "motion_control.hpp"
#include "control_allocation.hpp"


int main() {
    srand(time(0)); 
    YAML::Node config = YAML::LoadFile("../config.yaml");

    // Simulation parameters
    double h = config["simulation"]["h"].as<double>();
    double T_final = config["simulation"]["T_final"].as<double>();
 
    //Load condition
    double mp = config["load_condition"]["mp"].as<double>(); 
    Eigen::Vector3d rp(0.75, 0, 0);   // Payload location (front cabin) [m]
    
    // Ocean current
    double V_c = config["ocean_current"]["V_c"].as<double>();   
    double beta_c = deg2rad(config["ocean_current"]["beta_c"].as<double>());

    // Waypoints: position + heading angles for DP control
    Waypoints wpt;
    for (const auto &elem : config["waypoints"]["x"]) {
        wpt.x.push_back(elem.as<double>());
    }
    for (const auto &elem : config["waypoints"]["y"]) {
        wpt.y.push_back(elem.as<double>());
    }
    for (const auto &elem : config["waypoints"]["angle"]) {
        // Convert each angle from degrees to radians
        wpt.angle.push_back(deg2rad(elem.as<double>()));
    }
    
    // Additional parameter for straight-line path following
    double R_switch = config["path_following"]["R_switch"].as<double>();
     
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
    double T_n = 0.5;                                // Propeller time constant (s)
    Eigen::Vector2d n = Eigen::Vector2d::Zero();     // Init: [n_left, n_right] = [0, 0]

    double T_alpha = 1;                              // Azimuth angle time constant (s)
    Eigen::Vector2d alpha = Eigen::Vector2d::Zero(); // Init: [angle_left, angle_right] = [0, 0]

    // Initial states and variables
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12);  // x = [u v w p q r xn yn zn phi theta psi]'
    x(6)  = wpt.x[0];                               // Initial x position
    x(7)  = wpt.y[0];                               // Initial y position
    x(11) = psi0;                                   // Initial heading angle

    // Control method selection for path following
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

    Eigen::MatrixXd simdata(num_steps, 24);         
    
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

        if (pos_error_BODY < R_switch && wpt_index < wpt.x.size() - 2) {
            wpt_index = wpt_index + 1;

        } 
        else if (pos_error_BODY < R_switch && wpt_index == wpt.x.size() - 1){
            wpt_index = wpt_index + 1;
            GuidanceFlag = 1; // DP
            R_switch = 0.2;    //m
        }

        std::vector<double> wpt_start        = {wpt.x[wpt_index-1], wpt.y[wpt_index-1]};
        std::vector<double> wpt_goal         = {wpt.x[wpt_index], wpt.y[wpt_index]};
        std::vector<double> current_position = {xn, yn};

        switch (GuidanceFlag) {
            case 1: { // Dynamic Positioning
                std::vector<double> desired_states = DynamicPositioning(wpt_start, wpt_goal, current_position);
                xn_d  = desired_states[0];
                yn_d  = desired_states[1];
                psi_d = desired_states[2];
                break;
            }
            case 2: { // Station Keeping
                std::vector<double> desired_states = StationKeeping(wpt, wpt_index, xn, yn, psi_d);
                xn_d  = desired_states[0]; 
                yn_d  = desired_states[1]; 
                psi_d = desired_states[2]; 
                break;
            }
            case 3: { // other 
                
            }
        }

        // Control System 
        // (Finds n_c and alpha_c from the desired states)
        std::vector<double> tau = {0.0, 0.0, 0.0};
        Eigen::Vector2d n_c     = {0.0, 0.0};
        Eigen::Vector2d alpha_c = {0.0, 0.0};

        // DP mode for berthing
        if (GuidanceFlag==1) {
            // Motion control system
            std::vector<double> tau = tau_XYN_PID( 
                h,
                xn_d, yn_d, psi_d,
                xn, yn, psi,
                z_xn, z_yn, z_psi,
                prev_error_xn, prev_error_yn, prev_error_psi);

            tau_X = tau[0];
            tau_Y = tau[1];
            tau_N = tau[2];
            
            // Control allocation
            std::vector<double> control_allocation = NLOptControlAlloc(tau_X, tau_Y, tau_N);
            n_c     = {control_allocation[0], control_allocation[2]};
            alpha_c = {control_allocation[1], control_allocation[3]};

        } 
        // Path following
        else { 
            // Motion control system
            //std::vector<double> tau = tau_XN_PID(); //Not yet implemented
            std::vector<double> tau = tau_XYN_PID( //To be removed
                h,
                xn_d, yn_d, psi_d,
                xn, yn, psi,
                z_xn, z_yn, z_psi,
                prev_error_xn, prev_error_yn, prev_error_psi);

            tau_X = tau[0];
            tau_Y = tau[1];
            tau_N = tau[2];

            // Control allocation
            std::vector<double> control_allocation = NLOptControlAlloc(tau_X, tau_Y, tau_N);
            n_c     = {control_allocation[0], control_allocation[2]};
            alpha_c = {control_allocation[1], control_allocation[3]};
        }


        // Marine Craft Model
        //  rk4 method for x(k+1)
        rk4_ran_step(x, n, alpha, mp, rp, V_c, beta_c, h);
        x(11) = ssa(x(11));

        //  Euler's method
        n = n + h/T_n * (n_c - n);                      // Update propeller speeds
        alpha = alpha + h/T_alpha * (alpha_c - alpha);  // Update azimuth angles


        // Saturate:
        double k_pos = 220*9.81;     // Positive Bollard
        double k_neg = 220*9.81;     // Negative Bollard
        double n_max =  1;           // relative propellar speed max 
        double n_min = -1;           // relative propellar speed min
        double alpha_max = M_PI/2;   // maximum azimuth angle (rad)
        double alpha_min = -M_PI/2;  // minimum azimuth angle (rad)
        for (int i = 0; i < n.size(); ++i) {
            if (n(i) > n_max) n(i) = n_max;
            else if (n(i) < n_min) n(i) = n_min;
        }
        for (int i = 0; i < alpha.size(); ++i) {
            if (alpha(i) > alpha_max) alpha(i) = alpha_max;
            else if (alpha(i) < alpha_min) alpha(i) = alpha_min;
        }

        z_xn  = z_xn  + h * (pos_x_error);      // Update integral state for surge control
        z_yn  = z_yn  + h * (pos_y_error);      // Update integral state for sway control
        z_psi = z_psi + h * ssa(psi_d - psi);   // Update integral state for heading control

        // Show SIM progress once per second
        if (i % 20 == 0) {
            std::cout << std::fixed << std::setprecision(0)
            << "################################################" << std::endl
            << "Iteration: " << i << ", Time: " << t[i] << "s, "
            << "Active WP: ("<< wpt.x[wpt_index] << "," << wpt.y[wpt_index] << ")" << std::endl
            << "------------------------------------------------" << std::endl
            << "x_d: " << xn_d << "m, y_d: " << yn_d << "m, psi_d: " << rad2deg(psi_d) << "deg" << std::endl
            << "x:   " << xn << "m, y:   " << yn << "m, psi:   " << rad2deg(psi) << "deg" << std::endl
            << "------------------------------------------------" << std::endl
            << "n_c(0), n_c(1):         " << n_c(0) << ", " << n_c(1) << std::endl
            << "n(0),   n(1):           " << n(0) << ", " << n(1) << std::endl
            << "------------------------------------------------" << std::endl
            << "alpha_c(0), alpha_c(1): " << rad2deg(alpha_c(0)) << ", " << rad2deg(alpha_c(1)) << std::endl
            << "alpha(0), alpha(1):     " << rad2deg(alpha(0)) << ", " << rad2deg(alpha(1)) << std::endl
            << "------------------------------------------------" << std::endl
            << "tauX, tauY, tauN: " << tau_X << ", " << tau_Y << ", " << tau_N << std::endl
            << "" << std::endl
            << std::defaultfloat;
        }

        // Storing SIM data
        simdata(i, 0) = t[i];
        simdata(i, Eigen::seq(1, 12)) = x.transpose();  
        simdata(i, 13) = xn_d;
        simdata(i, 14) = yn_d;
        simdata(i, 15) = psi_d;
        simdata(i, 16) = n_c(0);                           
        simdata(i, 17) = n_c(1);
        simdata(i, 18) = n(0);
        simdata(i, 19) = n(1);
        simdata(i, 20) = alpha_c(0);
        simdata(i, 21) = alpha_c(1);
        simdata(i, 22) = alpha(0); 
        simdata(i, 23) = alpha(1); 

        // Calculating real, from relative, propellar revolutions per second. 
        // To boat n_real = r_max * n

    }
    std::cout<<"Simulation completed"<<std::endl;
    storeSimulationData(simdata, "simdata.csv");

    plotTrajectory();
    plotStateErrors();
    plotAngles();

    return 0;
}



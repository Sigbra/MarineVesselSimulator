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
    double V_c = 0.3;              // Ocean current speed (m/s)
    double beta_c = deg2rad(30.0); // Ocean current direction (rad)

    // Waypoints: position + heading angles for DP
    Waypoints wpt;
    wpt.x =     {          0,           0,          150,          150,          -100};
    wpt.y =     {          0,         200,          200,          -50,           -50};
    wpt.angle = {deg2rad(90), deg2rad(90),  deg2rad(45),  deg2rad(90),  deg2rad(-45)}; 
    
    // Additional parameter for straight-line path following
    double R_switch = 5;
    double K_f = 0.3;
    
    // Initial heading
    double psi0 = wpt.angle[0];
    
    //Calculating RAN USV input Matrix
    Eigen::VectorXd x_input = Eigen::VectorXd::Zero(12);  
    Eigen::VectorXd n_input = Eigen::VectorXd::Zero(2); 
    Eigen::VectorXd alpha_input = Eigen::VectorXd::Zero(2);                    
    Eigen::VectorXd xdot = Eigen::VectorXd::Zero(12); 
    double U = 0;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 6); 
    Eigen::MatrixXd B_prop = Eigen::MatrixXd::Zero(3, 2); 
    
    ran(x_input, n_input, alpha_input, mp, rp, V_c, beta_c, xdot, U, M, B_prop);

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
    ControlMethod control;
    int ControlFlag = control.selectMethod();

    // Current waypoint
    int wpt_index  = 0;           

    // Control forces and moment
    double tau_X   = 0.0;         // desired surge force
    double tau_Y   = 0.0;         // desired sway force
    double tau_N   = 0.0;         // desired yaw moment

    // States for PID control
    double z_xn    = 0.0;         // integral state for surge
    double z_yn    = 0.0;         // integral state for sway
    double z_psi   = 0.0;         // integral state for heading
    double prev_error_xn  = 0.0;  // previous surge error
    double prev_error_yn  = 0.0;  // previous sway error
    double prev_error_psi = 0.0;  // previous heading error

    int num_steps = static_cast<int>(T_final / h) + 1; // Total number of time steps
    std::vector<double> t(num_steps);                  // Time vector from 0 to T_final

    Eigen::MatrixXd simdata(num_steps, 12 + 4);        // Simulation data storage (x, n_c, alpha_c)
    
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
        switch (ControlFlag) {
            case 1: {
                NoDesiredForcesOrMoments(tau_X, tau_Y, tau_N);
                break;
            }
            case 2: {
                // Using meaurements for u, v, xn, yn, psi
                SISO_linear_PID_Control(u, v, xn, yn, psi,
                                        wpt, wpt_index, h,
                                        tau_X, tau_Y, tau_N,
                                        z_xn, z_yn, z_psi,
                                        prev_error_xn, prev_error_yn, prev_error_psi);
                break;
            }
            case 3: {
                std::cout << "ALOSpsi control method is not implemented" << std::endl;
                
                break;
            }
        }

        // Control Allocation 
        std::vector<double> control_allocation = NLOptControlAlloc(tau_X, tau_Y, tau_N);
        Eigen::Vector2d n_c, alpha_c;
        n_c << control_allocation[0], control_allocation[2];
        alpha_c << control_allocation[1], control_allocation[3];

        // Storing SIM data
        simdata(i, Eigen::seq(0, 11)) = x.transpose();  
        simdata(i, 12) = n_c(0);                           
        simdata(i, 13) = n_c(1);
        simdata(i, 14) = alpha_c(0);
        simdata(i, 15) = alpha_c(1);                       

        // rk4 method for x(k+1)
        rk4_ran_step(x, n, alpha, mp, rp, V_c, beta_c, h);

        // Euler's method
        n = n + h/T_n * (n_c - n);                             // Update propeller speeds
        alpha = alpha + h/T_alpha * (alpha_c - alpha);         // Update azimuth angles

        z_xn  = z_xn  + h * (xn - wpt.x[wpt_index]);            // Update integral state for surge control
        z_yn  = z_yn  + h * (yn - wpt.y[wpt_index]);            // Update integral state for sway control
        z_psi = z_psi + h * ssa(psi - wpt.angle[wpt_index]);   // Update integral state for heading control

        // Planning (Waypoint Update)
        double position_error = sqrt(pow(xn - wpt.x[wpt_index], 2) + pow(yn - wpt.y[wpt_index], 2));
        if (position_error < R_switch && wpt_index < wpt.x.size() - 1) {
            wpt_index = wpt_index + 1;
        }

        if (i % 100 == 0) {
            std::cout << "Iteration: " << i << ", Time: " << t[i] << "s, x: " << xn << "m, y: " << yn << "m, psi: " << psi << "rad" << std::endl;
        }
    }
    std::cout<<"Simulation completed"<<std::endl;
    storeSimulationData(simdata);

    plotTrajectory();

    return 0;
}



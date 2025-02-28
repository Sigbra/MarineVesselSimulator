#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib> // For getenv
#include <Eigen/SVD>  // SVD for pseudoinverse computation

#include "utilities.hpp"
#include "ran.hpp"
#include "ref_model.hpp"
#include "ALOSpsi.hpp"
#include "hermite_spline.hpp"
#include "crosstrack_hermite.hpp"
#include "los_observer.hpp"
#include "control_method.hpp"
#include "control_allocation.hpp"
#include "guidance.hpp"



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

    // Waypoints for initial position + 3DOF DP square test
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

    // Azimuth dynamics
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
    std::vector<std::string> methods = {
        "Dynamic Positioning (DP)",
    };

    ControlMethod control(methods);
    int ControlFlag = control.selectMethod();

    int wpt_index  = 0;           // which waypoint are we trying to hold?
    double z_psi   = 0.0;         // integral state for heading
    double tau_X   = 0.0;         // desired surge force
    double tau_Y   = 0.0;         // desired sway force
    double tau_N   = 0.0;         // desired yaw moment
    double r_d     = 0.0;         // desired yaw rate (for logging)
    double psi_d   = 0.0;         // desired heading (for logging)

    int num_steps = static_cast<int>(T_final / h) + 1; // Total number of time steps
    std::vector<double> t(num_steps);                  // Time vector from 0 to T_final

    Eigen::MatrixXd simdata(num_steps, 12 + 2);        // Simulation data storage (Does not cause segmentation fault)
    
    for (int i = 0; i < num_steps; ++i) {
        t[i] = i * h;
        
        //Measurements with noise
        double u   = x(0) + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        double v   = x(1) + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        double r   = x(5) + 0.001 * ((double)rand() / RAND_MAX - 0.5);  

        double xn  = x(6) + 0.01 * ((double)rand() / RAND_MAX - 0.5);   
        double yn  = x(7) + 0.01 * ((double)rand() / RAND_MAX - 0.5);     
        double psi = x(11) + 0.001 * ((double)rand() / RAND_MAX - 0.5);

        if (std::isnan(psi)) {
            std::cerr << "NaN detected for psi at iteration " << i << ", time: " << t[i] << "s\n";
            break; 
        }

        //Guidance and control system
        switch (ControlFlag) {
            case 1: {
                // Dynamic Positioning (DP)
                dynamicPositioning(wpt, x, wpt_index,
                                    tau_X, tau_Y, tau_N,
                                    r_d, psi_d, z_psi, h);
                break;
            }
        }

        // Control Allocation 
        std::vector<double> control_allocation = controlAllocation(tau_X, tau_Y, tau_N);
        Eigen::Vector2d n_c, alpha_c;
        n_c << control_allocation[0], control_allocation[2];
        alpha_c << control_allocation[1], control_allocation[3];

        // Storing SIM data
        simdata(i, Eigen::seq(0, 11)) = x.transpose();  
        simdata(i, 12) = r_d;                           
        simdata(i, 13) = psi_d;                         

        // rk4 method for x(k+1)
        rk4_ran_step(x, n, alpha, mp, rp, V_c, beta_c, h);

        // Euler's method
        // if ((n_c - n).isZero()) {
        //     std::cout << "Error: Division by zero" << std::endl;
        //     break;
        // }
        
        n = n + h/T_n * (n_c - n);                                    // Update propeller speeds
        alpha = alpha + h/T_alpha * (alpha_c - alpha);                // Update azimuth angles

        z_psi = z_psi + h * ssa(psi - psi_d);                         // Update integral state for heading control

        if (i % 100 == 0) {
            std::cout << "Iteration: " << i << ", Time: " << t[i] << "s, x: " << xn << "m, y: " << yn << "m, psi: " << psi << "rad" << std::endl;
        }
    }
    std::cout<<"Simulation completed"<<std::endl;
    storeSimulationData(simdata);

    plotTrajectory();

    return 0;
}



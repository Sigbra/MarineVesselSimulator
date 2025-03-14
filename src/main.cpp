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
    
    // Ocean current
    double V_c = config["ocean_current"]["V_c"].as<double>();   
    double beta_c = deg2rad(config["ocean_current"]["beta_c"].as<double>());

    // Waypoints: position
    Waypoints wpt;
    for (const auto &elem : config["waypoints"]["x"]) {
        wpt.x.push_back(elem.as<double>());
    }
    for (const auto &elem : config["waypoints"]["y"]) {
        wpt.y.push_back(elem.as<double>());
    }
    wpt = addIntermediateWaypoints(wpt, 10.0);

    std::cout << "wpt.x: ";
    for (double x : wpt.x) {
        std::cout << x << " ";
    }
    std::cout << "\nwpt.y: ";
    for (double y : wpt.y) {
        std::cout << y << " ";
    }
    std::cout << std::endl;

    // DP points: 
    Waypoints dp_points;
    for (const auto &elem : config["dp_points"]["x"]) {
        dp_points.x.push_back(elem.as<double>());
    }
    for (const auto &elem : config["dp_points"]["y"]) {
        dp_points.y.push_back(elem.as<double>());
    }
    dp_points = addIntermediateWaypoints(dp_points, 5.0);
    std::cout << "dp_points.x: ";
    for (double x : dp_points.x) {
        std::cout << x << " ";
    }
    std::cout << "\ndp_points.y: ";
    for (double y : dp_points.y) {
        std::cout << y << " ";
    }
    std::cout << std::endl;

    // Additional parameter for straight-line path following
    double R_switch = config["path_following"]["R_switch"].as<double>();
    double K_f = config["path_following"]["K_f"].as<double>();

    // ALOS and ILOS parameters
    double Delta_h = config["path_following"]["Delta_h"].as<double>();                    
    double gamma_h = config["path_following"]["gamma_h"].as<double>();               
    double kappa = config["path_following"]["kappa"].as<double>();   

    // Initializing guidance classes
    DynamicPositioning dp(dp_points, 5);
    ALOS alos(wpt, Delta_h, gamma_h, h, 0.5);
    LOSObserver losObserver(h, K_f);
     
    // Initial heading
    double psi0 = atan2(wpt.y[1] - wpt.y[0], wpt.x[1] - wpt.x[0]);
    
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
    //int wpt_index  = 1;      // First waypoint (index 0) is only used for initial heading, not goal position.
    //int dp_points_index = 1; 

    // Disired states in NED
    double xn_d    = wpt.x[0];        
    double yn_d    = wpt.y[0];        
    double psi_d   = psi0; 

    // ALOS output
    double psi_ref;
    double y_e;
    bool at_goal;     
    
    // Motion control classes
    PositionPIDController posPID;
    HeadingPIDController headPID;

    // - Desired rate of turn
    double r_d = 0; 
    // - Desired acceleration
    double a_d = 0;

    // Marine vessel Dynamics
    // - Derivative of state vector
    Eigen::VectorXd xdot = Eigen::VectorXd::Zero(12);
    // - System mass matrix (MRB + added mass)
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 6); 
    // - Input Matrix
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 4); 
    // - Speed (surge and sway)
    double U = 0.0;

    // Control system variables
    std::vector<double> tau_XYN = {0.0, 0.0, 0.0};
    std::vector<double> control_allocation = {0.0, 0.0, 0.0, 0.0};
    Eigen::Vector2d n_c     = {0.0, 0.0};
    Eigen::Vector2d alpha_c = {0.0, 0.0};
    
    // Total number of time steps
    int num_steps = static_cast<int>(T_final / h) + 1;
    // Time vector from 0 to T_final
    std::vector<double> t(num_steps);    

    // SIM data storage
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

        // Update model dynamics
        ran(x, n, alpha, mp, V_c, beta_c, xdot, U, M, B);

        // Guidance
        switch (GuidanceFlag) {
            case 1: { // Dynamic Positioning
                auto [xn_d, yn_d, psi_d] = dp.update(xn, yn);
                break;
            }
            case 2: { // ALOS heading autopilot, straight-line path following
                auto [psi_ref, y_e, at_goal] = alos.update(xn, yn);

                losObserver.update(psi_ref);
                psi_d = losObserver.getLOSAngle();
                r_d = losObserver.getLOSRate();

                if (at_goal) {
                    alos.reset();
                    GuidanceFlag = 1;
                }
                break;
            }
        }

        // Control System 

        // - Station Keeping
        if (GuidanceFlag==1) {
            // - Motion control system
            tau_XYN = posPID.update(h, xn_d, yn_d, psi_d, xn, yn, psi);

            
            // - Control allocation
            control_allocation = NLOptControlAlloc(tau_XYN[0], tau_XYN[1], tau_XYN[2], U);
            n_c     = {control_allocation[0], control_allocation[2]};
            alpha_c = {control_allocation[1], control_allocation[3]};

        } 
        // - Path following with ALOS or MPC
        else if (GuidanceFlag==2 || GuidanceFlag==3) { 
            // - Motion control system
            tau_XYN = headPID.update(h, M, psi, psi_d, r, r_d, a_d);

            // - Control allocation
            control_allocation = NLOptControlAlloc(tau_XYN[0], tau_XYN[1], tau_XYN[2], U);
            n_c     = {control_allocation[0], control_allocation[2]};
            alpha_c = {control_allocation[1], control_allocation[3]};
        }

        // Marine Craft Model
        rk4_ran_step(x, n, alpha, mp, V_c, beta_c, h);
        x(11) = ssa(x(11));

        // - Euler's method
        n = n + h/T_n * (n_c - n);                      
        alpha = alpha + h/T_alpha * (alpha_c - alpha);  


        // Saturate:
        double k_pos = 200*9.81;     
        double k_neg = 200*9.81;     
        double n_max =  1;             
        double n_min = -1;           
        double alpha_max = M_PI/2;   
        double alpha_min = -M_PI/2;  
        for (int i = 0; i < n.size(); ++i) {
            if (n(i) > n_max) n(i) = n_max;
            else if (n(i) < n_min) n(i) = n_min;
        }
        for (int i = 0; i < alpha.size(); ++i) {
            if (alpha(i) > alpha_max) alpha(i) = alpha_max;
            else if (alpha(i) < alpha_min) alpha(i) = alpha_min;
        }

        // Show SIM progress once per second
        if (i % 100 == 0) {
            std::cout << std::fixed << std::setprecision(0)
            << "################################################" << std::endl
            << "Iteration: " << i << ", Time: " << floor(t[i]/60) << "min, " << fmod(t[i], 60) << "s, "
            << "Guidance flag: " << GuidanceFlag << std::endl
            << "------------------------------------------------" << std::endl;
            if (GuidanceFlag == 1){
                std::cout << std::fixed << std::setprecision(1)
                << "x_d: " << xn_d << "m, y_d: " << yn_d << "m, psi_d: " << rad2deg(psi_d) << "deg" << std::endl;
            }
            else if (GuidanceFlag == 2) {
                std::cout << std::fixed << std::setprecision(1)
                << "At goal: " << at_goal << ", psi_ref: " << psi_ref << ", y_e: " << y_e <<std::endl;
            }
            std::cout << "x:   " << xn << "m, y:   " << yn << "m, psi:   " << rad2deg(psi) << "deg" << std::endl
            << "------------------------------------------------" << std::endl
            << std::fixed << std::setprecision(3)
            << "n_c(0), n_c(1):         " << n_c(0) << ", " << n_c(1) << std::endl
            << "n(0),   n(1):           " << n(0) << ", " << n(1) << std::endl
            << "------------------------------------------------" << std::endl
            << std::fixed << std::setprecision(0)
            << "alpha_c(0), alpha_c(1): " << rad2deg(alpha_c(0)) << ", " << rad2deg(alpha_c(1)) << std::endl
            << "alpha(0), alpha(1):     " << rad2deg(alpha(0)) << ", " << rad2deg(alpha(1)) << std::endl
            << "------------------------------------------------" << std::endl
            << "tauX, tauY, tauN: " << tau_XYN[0] << ", " << tau_XYN[1] << ", " << tau_XYN[2] << std::endl
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

        // To physical boat n_real = nReal(n)

    }
    std::cout<<"Simulation completed"<<std::endl;
    storeSimulationData(simdata, "simdata.csv");

    plotTrajectory();
    plotStateErrors();
    plotAngles();

    return 0;
}



#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <yaml-cpp/yaml.h>

#include "Models/ran.hpp"
#include "Guidance/guidance.hpp"
#include "Utilities/utilities.hpp"
#include "Control/motion_control.hpp"
#include "Control/control_allocation.hpp"
#include "Planning/path_generation.hpp"

// Helper function to select path type
int selectPathType() {
    std::cout << "Choose Path Type:" << std::endl;
    std::cout << "1. Straight Line Path" << std::endl;
    std::cout << "2. Continuous-Curvature Path Using Fermat's Spiral" << std::endl;
    
    int choice = 0;
    while (true) {
        std::cout << "Enter the number of your choice: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
        }
        else if (choice >= 1 && choice <= 2) {
            break;
        }
        else {
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    return choice;
}

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

    // Load original waypoints from config
    Waypoints wpt;
    auto waypointsNode = config["waypoints"];
    for (size_t i = 0; i < waypointsNode["x"].size(); i++) {
        Vector2D point;
        point.x = waypointsNode["x"][i].as<double>();
        point.y = waypointsNode["y"][i].as<double>();
        wpt.push_back(point);
    }
    
    std::cout << "Waypoints: ";
    for (const auto& point : wpt) {
        std::cout << "(" << point.x << ", " << point.y << ") ";
    }
    std::cout << std::endl;

    
    // Parameters for path following
    double R_switch = config["path_following"]["R_switch"].as<double>();
    double K_f = config["path_following"]["K_f"].as<double>();
    double Delta_h = config["path_following"]["Delta_h"].as<double>();                    
    double gamma_h = config["path_following"]["gamma_h"].as<double>();               
    double kappa = config["path_following"]["kappa"].as<double>();   


    // Create the straight line path.
    StraightLinePath straightLinePath;
    straightLinePath.updateWaypoints(wpt);
    Waypoints pathLine = straightLinePath.samplePath(0.1);
    std::cout << "Path size: " << pathLine.size() << std::endl;
    plotPath(pathLine);

    // Create the Fermat spiral path.
    // - Set the curvature constraint (κ_max in rad/m).
    double kappa_max = 0.2; //Do not change this value without changing the theta_kappa_max in the path_generation.cpp file.
    FermatSpiralPath spiral(kappa_max);
    spiral.updateWaypoints(wpt);
    Waypoints pathFS = spiral.samplePath(0.001);
    std::cout << "Path size: " << pathFS.size() << std::endl;
    plotPath(pathFS);

    

    // Initialize guidance methods and LOS observer 
    double delta = 5.0; // Lookahead distance
    LOSObserver losObserver(h, K_f);
     
    // Initial states - will be properly set after path generation
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12);  // x = [u v w p q r xn yn zn phi theta psi]'
    x(11) = std::atan2(wpt[1].y - wpt[0].y, wpt[1].x - wpt[0].x);
    
    // Azimuth pod dynamics
    double T_n = 0.5;                                // Propeller time constant (s)
    Eigen::Vector2d n = Eigen::Vector2d::Zero();     // Init: [n_left, n_right] = [0, 0]

    double T_alpha = 1;                              // Azimuth angle time constant (s)
    Eigen::Vector2d alpha = Eigen::Vector2d::Zero(); // Init: [angle_left, angle_right] = [0, 0]

    // Choose path type
    int pathType = selectPathType();

    // Initialize path following variables
    int wpt_index = 1;
    PathPoint closest;
    closest.pos = Vector2D(0.0, 0.0);
    closest.dpos = Vector2D(0.0, 0.0);
    closest.ddpos = Vector2D(0.0, 0.0);

    double path_x = wpt[wpt_index-1].x;
    double path_y = wpt[wpt_index-1].y;
    double path_x_dot = 0.0;
    double path_y_dot = 0.0;
    double path_x_ddot = 0.0;
    double path_y_ddot = 0.0;

    // Control method selection for path following
    GuidanceMethod guidance;
    int GuidanceFlag = guidance.selectMethod();

    ControlAllocationMethod controlAlloc;
    int ControlAllocFlag = controlAlloc.selectMethod();

    // Initial desired states (will be set properly after path generation)
    double xn_d = 0.0;        
    double yn_d = 0.0;        
    double psi_d = 0.0; 

    // ALOS variables
    double psi_ref = 0.0;
    double y_e = 0.0; 
    
    // Motion control classes
    PositionPIDController posPID;
    HeadingPIDController headPID;

    // Desired rate of turn and acceleration
    double r_d = 0.0; 
    double a_d = 0.0;

    // Marine vessel Dynamics
    Eigen::VectorXd xdot = Eigen::VectorXd::Zero(12);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 6); 
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 4); 
    double U = 0.0;

    // Control system variables
    std::vector<double> tau_XYN = {0.0, 0.0, 0.0};
    std::vector<double> control_allocation = {0.0, 0.0, 0.0, 0.0};
    Eigen::Vector2d n_c = {0.0, 0.0};
    Eigen::Vector2d alpha_c = {0.0, 0.0};
    
    // Total number of time steps
    int num_steps = static_cast<int>(T_final / h) + 1;
    std::vector<double> t(num_steps);    

    // SIM data storage
    Eigen::MatrixXd simdata(num_steps, 24);         
    
    // Main simulation loop
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

        // - Mode switching condition
        if (wpt_index == wpt.size()-1) {
            if (R_switch > std::sqrt(std::pow(xn - wpt[wpt_index].x, 2) + std::pow(yn - wpt[wpt_index].y, 2))) {
                pathType = 0;
                GuidanceFlag = 1; 
            }
        }

        // - Type of path
        // - path functions responsible for wpt_index increment
        switch (pathType) {
            case 0: { // Dynamic Positioning does not use a path. Use Voronoi space with MPC
                break; 
            }
            case 1: { // Straight line path.
                straightLinePath.updateWaypoints(wpt);
                closest = straightLinePath.getClosestPoint(Vector2D(xn, yn), wpt_index);
                break;
            }
            case 2: { // Continuous-Curvature Path Using Fermat's Spiral.
                spiral.updateWaypoints(wpt);
                closest = spiral.getClosestPoint(Vector2D(xn, yn), wpt_index);
                break;
            }
        }
        
        path_x = closest.pos.x;
        path_y = closest.pos.y;
        path_x_dot = closest.dpos.x;
        path_y_dot = closest.dpos.y;
        path_x_ddot = closest.ddpos.x;
        path_y_ddot = closest.ddpos.y;


        // - Guidance law
        switch (GuidanceFlag) {
            case 1: { // Dynamic Positioning
                auto [xn_d, yn_d, psi_d] = DP(xn, yn, wpt[wpt_index].x, wpt[wpt_index].y, wpt[wpt_index-1].x, wpt[wpt_index-1].y);

                break;
            }
            case 2: { // LOS heading autopilot
                auto [psi_ref, y_e] = LOS(xn, yn, delta, path_x, path_y, path_x_dot, path_y_dot);

                losObserver.update(psi_ref);
                psi_d = losObserver.getLOSAngle();
                r_d = losObserver.getLOSRate();

                break;
            }
            case 3: { // ALOS heading autopilot
                auto [psi_ref, y_e] = ALOS(xn, yn, delta, path_x, path_y, path_x_dot, path_y_dot);

                losObserver.update(psi_ref);
                psi_d = losObserver.getLOSAngle();
                r_d = losObserver.getLOSRate();

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
        // - Path following 
        else if (GuidanceFlag==2) { 
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
        for (int j = 0; j < n.size(); ++j) {
            if (n(j) > n_max) n(j) = n_max;
            else if (n(j) < n_min) n(j) = n_min;
        }
        for (int j = 0; j < alpha.size(); ++j) {
            if (alpha(j) > alpha_max) alpha(j) = alpha_max;
            else if (alpha(j) < alpha_min) alpha(j) = alpha_min;
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
                << "psi_ref: " << psi_ref << ", psi_d: " << psi_d << ", y_e: " << y_e <<std::endl;
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
    }

    std::cout << "Simulation completed" << std::endl;
    storeSimulationData(simdata, "simdata.csv");

    plotTrajectory();
    plotStateErrors();
    plotAngles();

    return 0;
}





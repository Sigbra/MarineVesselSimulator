#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <thread>

#include "Control/control_alloc_selector.hpp"
#include "Control/pseudo_inverse_allocation.hpp"
#include "Control/PID_MIMO_motion_control.hpp"
#include "Control/PID_heading_motion_control.hpp"
#include "Control/MPC_control_system.hpp"
#include "Control/MPC_control_alloc.hpp"
#include "Control/non_lin_constrained_control_alloc.hpp"

#include "Guidance/guidance_selector.hpp"
#include "Guidance/LOS.hpp"
#include "Guidance/ALOS.hpp"
#include "Guidance/LOS_observer.hpp"
#include "Guidance/dynamic_positioning.hpp"

#include "Models/ran.hpp"
#include "Models/ref_model.hpp"
#include "Models/model_utilities.hpp"

#include "Planning/plan_selector.hpp"
#include "Planning/straight_line_planning.hpp"
#include "Planning/fermat_spiral_planning.hpp"

#include "Utilities/calculations.hpp"
#include "Utilities/plotting.hpp"


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

    // Load angles from config
    // - If no angles are provided, the default is to use the angle given by the wpt's or closest point.
    std::vector<double> angles;
    if (waypointsNode["angles"]) {
        auto anglesNode = waypointsNode["angles"];
        for (size_t i = 0; i < anglesNode.size(); i++) {
            angles.push_back(deg2rad(anglesNode[i].as<double>()));
        }
    }
    std::cout << "Angles: ";
    if (angles.empty()) {
        std::cout << "No angles provided in the config." << std::endl;
    } else {
        for (const auto& angle : angles) {
            std::cout << angle << " ";
        }
        std::cout << std::endl;
    }
    
    // Parameters for path following
    double R_switch = config["path_following"]["R_switch"].as<double>();
    double K_f = config["path_following"]["K_f"].as<double>();
    double Delta_h = config["path_following"]["Delta_h"].as<double>();                    
    double gamma_h = config["path_following"]["gamma_h"].as<double>();               

    // x = [u v w p q r xn yn zn phi theta psi]'
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12); 
    x(6) = wpt[0].x; // North position (NED frame)
    x(7) = wpt[0].y; // East position (NED frame)
    x(11) = std::atan2(wpt[1].y - wpt[0].y, wpt[1].x - wpt[0].x);

    // Control system variables
    std::vector<double> tau_XYN = {0.0, 0.0, 0.0};
    std::vector<double> control_allocation = {0.0, 0.0, 0.0, 0.0};
    Eigen::Vector2d n_c = {0.0, 0.0};
    Eigen::Vector2d alpha_c = {0.0, 0.0};

    // Model;   
    RAN ran_model;
    ran_model.update(x, mp, V_c, beta_c, h, n_c, alpha_c);
    Eigen::MatrixXd M = ran_model.get_M();
    Eigen::MatrixXd B = ran_model.get_B();
    double U = ran_model.get_U();
    Eigen::VectorXd xdot = ran_model.get_xdot();

    double T_n = ran_model.getT_n();          // Propeller time constant (s)
    double T_alpha = ran_model.getT_alpha();  // Azimuth angle time constant (s)

    ran_model.select_failure_mode();
    std::vector<bool> failstate = ran_model.check_failstate();

    Eigen::Vector2d n = ran_model.get_n();           
    Eigen::Vector2d alpha = ran_model.get_alpha();

    // Create the straight line path.
    StraightLinePath straightLinePath;
    straightLinePath.updateWaypoints(wpt);
    Waypoints pathLine = straightLinePath.samplePath(0.05);
    std::cout << "Path size: " << pathLine.size() << std::endl;
    plotPath(pathLine);

    // Create the Fermat spiral path.
    // - Set the curvature constraint (k_max in rad/m).
    double kappa_max = 0.2; 
    FermatSpiralPath spiral(kappa_max);
    spiral.updateWaypoints(wpt);
    Waypoints pathFS = spiral.samplePath(0.05);
    std::cout << "Path size: " << pathFS.size() << std::endl;
    plotPath(pathFS);

    // Initialize guidance methods and LOS observer 
    ALOS ALOS(Delta_h, gamma_h, 0.1);
    LOSObserver losObserver(h, K_f, x(11));

    // Choose path type
    int pathType = selectPathType();

    // Initialize path following variables
    int wpt_index = 1;
    PathTrackingInfo closest;
    closest.x_e = 0.0;
    closest.y_e = 0.0;
    closest.point.pos = Vector2D(0.0, 0.0);
    closest.point.dpos = Vector2D(0.0, 0.0);
    closest.point.ddpos = Vector2D(0.0, 0.0);

    double path_x = wpt[wpt_index-1].x;
    double path_y = wpt[wpt_index-1].y;
    double path_x_dot = 0.0;
    double path_y_dot = 0.0;

    // Control method selection for path following
    GuidanceMethod guidance;
    int GuidanceFlag = guidance.selectMethod();

    ControlAllocationMethod controlAlloc;
    int ControlAllocFlag = controlAlloc.selectMethod();

    // Initial desired states
    double xn_d  = x(6);        
    double yn_d  = x(7);        
    double psi_d = x(11); 
    double U_d   = 0.0;

    // ALOS variables
    double psi_ref = 0.0;
    double y_e = closest.y_e; 

    double x_e = closest.x_e;
    
    // Motion control classes
    MIMOPIDController MIMO_PID;
    HeadingPIDController headPID;
    MPC_Control_System mpc_control(30, 3*h); 

    // Desired rate of turn and acceleration
    double r_d = 0.0; 
    double a_d = 0.0;

    // Marine vessel Dynamics
    Eigen::VectorXd eta = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd nu = Eigen::VectorXd::Zero(6);
    
    // Total number of time steps
    int num_steps = static_cast<int>(T_final / h) + 1;
    std::vector<double> t(num_steps);    

    // SIM data storage
    Eigen::MatrixXd simdata(num_steps, 31);         
    
    RealTimePlotter plotter;
    if (pathType == 1 || pathType == 2) {
        plotter.setSampledPath(pathLine);
    }
    else if (pathType == 3) {
        plotter.setSampledPath(pathFS);
    }

    bool break_flag = false;

    

    // Main simulation loop
    for (int i = 0; i < num_steps; ++i) {

        t[i] = i * h;
        
        // ------------------------------ Navigation System ------------------------------
        // (Fake measurements using noise)

        double random = ((double)rand() / RAND_MAX - 0.5);
  
        double u     = x(0)  +  0.01 * random; // Surge velocity (BODY frame)
        double v     = x(1)  +  0.01 * random; // Sway velocity  (BODY frame)
        double w     = x(2)  +  0.01 * random; // Heave velocity (BODY frame)
        double p     = x(3)  + 0.001 * random; // Roll rate      (BODY frame)
        double q     = x(4)  + 0.001 * random; // Pitch rate     (BODY frame)
        double r     = x(5)  + 0.001 * random; // Yaw rate       (BODY frame)
    
        double xn    = x(6)  +  0.01 * random; // North position  (NED frame)
        double yn    = x(7)  +  0.01 * random; // East position   (NED frame)
        double zn    = x(8)  +  0.01 * random; // Down position   (NED frame)
        double phi   = x(9)  + 0.001 * random; // Roll angle      (NED frame)
        double theta = x(10) + 0.001 * random; // Pitch angle     (NED frame)
        double psi   = x(11) + 0.001 * random; // Heading angle   (NED frame)

        if (std::isnan(psi)) {
            std::cerr << "NaN detected for psi at iteration " << i << ", time: " << t[i] << "s\n";
            break; 
        }

        // ------------------------------ Update model dynamics ------------------------------
        ran_model.update(x, mp, V_c, beta_c, h, n_c, alpha_c);
        M = ran_model.get_M();
        U = ran_model.get_U();
        n = ran_model.get_n();
        alpha = ran_model.get_alpha();
        B = ran_model.get_B();

        // ------------------------------ Mode switch ------------------------------

        // - Switch criteria for path following to DP mode. 
        // if (GuidanceFlag != 1) {
        //     if (R_switch > std::sqrt(std::pow(xn - wpt[wpt.size()-1].x, 2) + std::pow(yn - wpt[wpt.size()-1].y, 2))) {
        //         if (pathType)
        //         GuidanceFlag = 1; 
        //         pathType = 1;
        //         wpt_index = wpt.size()-1;
        //     }
        //     if (path_x == wpt.back().x && path_y == wpt.back().y) {
        //         GuidanceFlag = 1; 
        //         pathType = 1;
        //         wpt_index = wpt.size()-1;
        //     }
        // }

        // ------------------------------ Path planning: connecting waypoints ------------------------------
        switch (pathType) {
            case 1: { // Dynamic Positioning.
                if (R_switch > std::sqrt(std::pow(xn - wpt[wpt_index].x, 2) + std::pow(yn - wpt[wpt_index].y, 2))){
                    if (std::abs(ssa(psi_d-psi)) < deg2rad(1) && U < 0.01) {
                        if (wpt_index < wpt.size()-1) {
                            wpt_index += 1;
                        }
                        else {
                            std::cout << "Reached the last waypoint." << std::endl;
                            break_flag = true;
                        }
                    }
                }
                break; 
            }
            case 2: { // Straight line path.
                closest = straightLinePath.getClosestPoint(Vector2D(xn, yn), wpt_index);
                y_e = closest.y_e;
                x_e = closest.x_e;
                path_x = closest.point.pos.x;
                path_y = closest.point.pos.y;
                path_x_dot = closest.point.dpos.x;
                path_y_dot = closest.point.dpos.y;
                if (wpt_index == wpt.size()-1){
                    if (closest.point.pos.x == wpt[wpt.size()-1].x && closest.point.pos.y == wpt[wpt.size()-1].y) {
                        break_flag = true;
                    }
                }
                break;
            }
            case 3: { // Continuous-Curvature Path Using Fermat's Spiral.
                closest = spiral.getClosestPoint(Vector2D(xn, yn), wpt_index);
                y_e = closest.y_e;
                x_e = closest.x_e;
                path_x = closest.point.pos.x;
                path_y = closest.point.pos.y;
                path_x_dot = closest.point.dpos.x;
                path_y_dot = closest.point.dpos.y;
                if (wpt_index == wpt.size()-1){
                    if (closest.point.pos.x == wpt[wpt.size()-1].x && closest.point.pos.y == wpt[wpt.size()-1].y) {
                        break_flag = true;
                    }
                }
                break;
            }
        }

        // ------------------------------ Guidance laws ------------------------------
        switch (GuidanceFlag) {
            case 1: { // Dynamic Positioning wpt path
                if (angles.empty()) {
                    auto [xn_ref, yn_ref, psi_ref] = DP(xn, yn, wpt[wpt_index].x, wpt[wpt_index].y, wpt[wpt_index-1].x, wpt[wpt_index-1].y);
                    xn_d = xn_ref;
                    yn_d = yn_ref;
                    psi_d = psi_ref;
                }
                else {
                    auto [xn_ref, yn_ref, psi_ref] = DP(xn, yn, wpt[wpt_index].x, wpt[wpt_index].y, wpt[wpt_index-1].x, wpt[wpt_index-1].y, angles[wpt_index-1]);
                    xn_d = xn_ref;
                    yn_d = yn_ref;
                    psi_d = psi_ref;
                }
                break;
            }
            case 2: { // LOS heading autopilot
                auto [psi_ref, _ ] = LOS(xn, yn, Delta_h, path_x, path_y, path_x_dot, path_y_dot, y_e);

                losObserver.update(psi_ref);
                psi_d = losObserver.getLOSAngle();
                r_d = losObserver.getLOSRate();
                break;
            }
            case 3: { // ALOS heading autopilot
                auto [psi_ref, _ ] = ALOS.update(xn, yn, path_x, path_y, path_x_dot, path_y_dot, y_e);

                losObserver.update(psi_ref);
                psi_d = losObserver.getLOSAngle();
                r_d = losObserver.getLOSRate();
                break;
            }
        }

        // ------------------------------ Control System ------------------------------

        // - Motion Control: Dynamic positioning
        if (GuidanceFlag==1 && ControlAllocFlag != 4){
            eta << u, v, w, p, q, r;
            nu  << xn, yn, zn, phi, theta, psi;
            tau_XYN = MIMO_PID.update(h, xn_d, yn_d, psi_d, M, eta, nu);
        } 
        // - Motion Control: Path following: 
        else if (GuidanceFlag==2 || GuidanceFlag==3) { 
            tau_XYN[0] = 3;
            tau_XYN[1] = 0;
            tau_XYN[2] = headPID.update(h, M, psi, psi_d, r, r_d, a_d);
        }              

        // - Control allocation
        switch (ControlAllocFlag) {
            case 1: { // Pseudo-inverse control allocation
                control_allocation = pseudo_inverse_allocation(tau_XYN, B, 200, 200);
                n_c     = {control_allocation[0], control_allocation[2]};
                alpha_c = {control_allocation[1], control_allocation[3]};
                break;
            }
            case 2: { // Nonlinear optimization with constraints
                control_allocation = NLOptControlAlloc(tau_XYN[0], tau_XYN[1], tau_XYN[2], U, n, alpha, failstate);
                n_c     = {control_allocation[0], control_allocation[2]};
                alpha_c = {control_allocation[1], control_allocation[3]};
                break;
            }
            case 3: { // Nonlinear optimization with constraints over a horizon taking rate constriants into account
                control_allocation = MPC_control_alloc(tau_XYN[0], tau_XYN[1], tau_XYN[2], U, T_n, T_alpha, n, alpha, failstate);
                n_c     = {control_allocation[0], control_allocation[2]};
                alpha_c = {control_allocation[1], control_allocation[3]};
                break;
            }
            case 4: { // Model Predictive Control System (Motion control and control allocation using vessel model)
                std::vector<double> x0 = {xn, yn, psi, u, v, r}; 
                mpc_control.solve(x0, wpt[wpt_index-1].x, wpt[wpt_index-1].y, xn_d, yn_d, psi_d, n, alpha, failstate);
                n_c = mpc_control.get_n_opt();
                alpha_c = mpc_control.get_alpha_opt();
                break;
            }
            default: {
                std::cerr << "Invalid control allocation method selected." << std::endl;
                break_flag = true;
                break;
            }
        }

        // ------------------------------ State update (x) ------------------------------

        // Marine Craft Model
        ran_model.rk4(x, mp, V_c, beta_c, h, n_c, alpha_c);
        //x(11) = ssa(x(11)); //makes plotting look bad              

        // ------------------------------ Plotting and Info ------------------------------

        // Show SIM progress once in a while
        if (i % 5 == 0) {
            std::vector<double> GuidanceVectorX;
            std::vector<double> GuidanceVectorY;
            if (GuidanceFlag == 1){
                GuidanceVectorX = {wpt[wpt_index].x};
                GuidanceVectorY = {wpt[wpt_index].y};
            }
            else if (GuidanceFlag == 2 || GuidanceFlag == 3){
                GuidanceVectorX = {path_x};
                GuidanceVectorY = {path_y};
            }

            plotter.updatePlot(xn, yn, psi, 0.2, GuidanceVectorX, GuidanceVectorY);

            std::cout << std::fixed << std::setprecision(0)
            << "################################################" << std::endl
            << "Iteration: " << i << ", Time: " << floor(t[i]/60) << "min, " << fmod(t[i], 60) << "s, " <<std::endl
            << "------------------------------------------------" << std::endl
            << "Path type: " << pathType << ", Guidance flag: " << GuidanceFlag << ", Control flag: " << ControlAllocFlag << std::endl
            << "wpt index: " << wpt_index
            << std::fixed << std::setprecision(0)
            << ", current wpt: (" << wpt[wpt_index].x << ", " << wpt[wpt_index].y << ")" << std::endl
            << "------------------------------------------------" << std::endl
            << "closest point: " << closest.point.pos.x << ", " << closest.point.pos.y << std::endl
            << std::fixed << std::setprecision(1)
            << "x_e: " << x_e << ", y_e: " << y_e << std::endl
            << "------------------------------------------------" << std::endl;
            if (GuidanceFlag == 1){
                std::cout << std::fixed << std::setprecision(2)
                << "x_d: " << xn_d << "m, y_d: " << yn_d << "m, psi_d: " << rad2deg(psi_d) << "deg" << std::endl;
            }
            else if (GuidanceFlag == 2 || GuidanceFlag == 3) {
                std::cout << std::fixed << std::setprecision(3)
                << "psi_d: " << rad2deg(psi_d) << ", r_d: " << rad2deg(r_d) << std::endl;
            }
            std::cout << "x:   " << xn << "m, y:   " << yn << "m, psi:   " << rad2deg(psi) << "deg" << ", U: " << U << std::endl
            << "------------------------------------------------" << std::endl
            << "CO offset (x, y, z): " << CO_Offset(U).transpose() << std::endl
            << "------------------------------------------------" << std::endl
            << std::fixed << std::setprecision(4)
            << "n_c(0), n_c(1):         " << n_c(0) << ", " << n_c(1) << std::endl
            << "n(0),   n(1):           " << n(0) << ", " << n(1) << std::endl
            << "------------------------------------------------" << std::endl
            << std::fixed << std::setprecision(2)
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
        simdata(i, 24) = tau_XYN[0];
        simdata(i, 25) = tau_XYN[1];
        simdata(i, 26) = tau_XYN[2];
        simdata(i, 27) = closest.point.pos.x;
        simdata(i, 28) = closest.point.pos.y;
        simdata(i, 29) = closest.x_e;
        simdata(i, 30) = closest.y_e;

        if (break_flag == true) {
            for (int j = i; j < num_steps; ++j) {
                simdata.row(j) = simdata.row(i);
                simdata(j, 0) = j * h;
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "Simulation completed" << std::endl;
    storeSimulationData(simdata, "simdata.csv");

    plotter.finalizePlot();

    plotTrajectory();
    plotClosestPointErrors();
    plotStateErrors();
    plotAngles();
    plotPropellerSpeeds();
    plotAlphas();

    return 0;
}





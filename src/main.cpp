#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib> // for getenv
//#include <boost/numeric/odeint.hpp>
#include "utilities.hpp"
#include "ran.hpp"
#include "ref_model.hpp"
#include "ALOSpsi.hpp"
#include "hermite_spline.hpp"
#include "crosstrack_hermite.hpp"
#include "los_observer.hpp"
#include "control_method.hpp"

int main() {
    srand(time(0)); // Randomize seed

    // USER INPUTS
    double h = 0.05;            // Sampling time [s]
    double T_final = 1000;      // Final simulation time [s]

    //Load condition
    double mp = 0;                  // Payload mass [kg]
    Eigen::Vector3d rp(1, 0, 0);    // Payload location [m]
    
    // Ocean current
    double V_c = 0.3; // Ocean current speed (m/s)
    double beta_c = M_PI / 6; // Ocean current direction (rad)
    
    // Waypoints
    Waypoints wpt;
    wpt.x = {0, 0, 150, 150, -100, -100, 200};
    wpt.y = {0, 200, 200, -50, -50, 250, 250};
    
    // ALOS and ILOS parameters
    double Delta_h = 10;
    double gamma_h = 0.001;
    double kappa = 0.001;
    
    // Additional parameter for straight-line path following
    double R_switch = 5;
    double K_f = 0.3;
    
    // Initial heading
    double psi0 = atan2(wpt.y[1] - wpt.y[0], wpt.x[1] - wpt.x[0]);
    
    // Additional parameters for Hermite spline path following
    double Umax = 2;
    int idx_start = 1;
    SplineResult spline = hermiteSpline(wpt, Umax, h);
    
    //Calculating RAN USV input Matrix
    Eigen::VectorXd x_input = Eigen::VectorXd::Zero(12);  
    Eigen::VectorXd n_input = Eigen::VectorXd::Zero(2);                     
    Eigen::VectorXd xdot = Eigen::VectorXd::Zero(12); 
    double U = 0;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 6); 
    Eigen::MatrixXd B_prop = Eigen::MatrixXd::Zero(3, 3); 
    
    ran(x_input, n_input, mp, rp, V_c, beta_c, xdot, U, M, B_prop);
    //Assuming B_prop can be inverted
    Eigen::MatrixXd Binv = B_prop.inverse();
    
    //PID heading autopilot parameters(Nomoto model: M(6,6)=T/K)
    double T = 1.0;  // Nomoto time constant
    double K = T / M(5, 5);  // Nomoto gain constant (Note: M(6,6) in MATLAB corresponds to M(5,5) in C++ 0-based indexing)

    double wn = 1.5;  // Closed-loop natural frequency (rad/s)
    double zeta = 1.0;  // Closed-loop relative damping factor (-)
    
    double Kp = M(5, 5) * wn * wn;  // Proportional gain
    double Kd = M(5, 5) * (2 * zeta * wn - 1 / T);  // Derivative gain
    double Td = Kd / Kp; // Derivative time constant
    double Ti = 10 / wn; // Integral time constant

    // Reference model parameters
    double wn_d = 1.0;
    double zeta_d = 1.0;
    double r_max = M_PI / 18;

    //Propellar dynamics
    double T_n = 0.1;  // Propeller time constant (s)
    Eigen::Vector2d n; // Initial propeller speed, [n_left n_right]'
    n << 0, 0;  // Initial values for propeller speeds

    // Initial states and variables
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12);  // x = [u v w p q r xn yn zn phi theta psi]'
    x(11) = psi0;  // Heading angle (Note: x(12) in MATLAB corresponds to x(11) in C++ 0-based indexing)
    double z_psi = 0;  // Integral state for heading control
    double psi_d = psi0;  // Desired heading angle
    double r_d = 0;  // Desired rate of turn
    double a_d = 0;  // Desired acceleration
    
    LOSObserver losObserver(h, K_f);

    // Control method selection
    std::vector<std::string> methods = {
        "PID heading autopilot, no path following",
        "ALOS path-following control using straight lines and waypoint switching",
        "ILOS path-following control using straight lines and waypoint switching",
        "ALOS path-following control using Hermite splines"
    };
    ControlMethod control(methods);
    int ControlFlag = control.selectMethod();

    int num_steps = static_cast<int>(T_final / h) + 1; // Total number of time steps
    std::vector<double> t(num_steps);                  // Time vector from 0 to T_final
    Eigen::MatrixXd simdata(num_steps, 12 + 2);        // Simulation data storage (Does not cause segmentation fault)

    
    for (int i = 0; i < num_steps; ++i) {
        t[i] = i * h;
        
        //Measurements with noise
        double r = x(5) + 0.001 * ((double)rand() / RAND_MAX - 0.5);     
        double xn = x(6) + 0.01 * ((double)rand() / RAND_MAX - 0.5);   
        double yn = x(7) + 0.01 * ((double)rand() / RAND_MAX - 0.5);     
        double psi = x(11) + 0.001 * ((double)rand() / RAND_MAX - 0.5);  

        if (std::isnan(psi)) {
            std::cerr << "NaN detected for psi at iteration " << i << ", time: " << t[i] << "s\n";
            break; 
        }

        //Guidance and control system
        switch (ControlFlag) {
            case 1: {
                double psi_ref = psi0;
                if (t[i] > 100) {
                    psi_ref = 0;
                }
                if (t[i] > 500) {
                    psi_ref = -M_PI / 2;
                }
                refModel(psi_d, r_d, a_d, psi_ref, r_max, zeta_d, wn_d, h, true);
                break;
            }
            case 2: {
                auto [psi_ref, _] = ALOSpsi(xn, yn, Delta_h, gamma_h, h, R_switch, wpt);
                if (std::isnan(psi_ref)) {
                    std::cout << "NaN detected in ALOSpsi output!" << std::endl;
                    break;  // Exit or handle the error appropriately
                }
                losObserver.update(psi_ref);
                //std::cout <<"hello2" << std::endl;
                psi_d = losObserver.getLOSAngle();
                //std::cout <<"hello3, psi_d: " << psi_d << std::endl;
                r_d = losObserver.getLOSRate();
                //std::cout <<"hello4, r_d: " << r_d << std::endl;
                break;
            }
            case 3: {
                // Implement ILOS guidance function call here
                break;
            }
            case 4: {
                // Implement Hermite spline path following here
                break;
            }
        }

        //PID heading (yaw moment) autopilot and forward thrust
        double tau_X = 100;
        //std::cout << "hello5.0" << std::endl;
        //std::cout<< "psi: " << psi << ", psi_d: " << psi_d << std::endl;
        //double psi_diff = ssa(psi - psi_d);
        //std::cout << "hello5.0.1, psi_diff: " << psi_diff << std::endl;
        double tau_N = (T/K) * a_d + (1/K) * r_d             // Yaw moment control
                       - Kp * (ssa(psi - psi_d)              // Proportional term
                       + Td * (r - r_d) + (1/Ti) * z_psi);   // Derivative and integral terms
        //std::cout << "hello5.1" << std::endl;
        //Control Allocation
        Eigen::Vector2d controlInputs;
        controlInputs << tau_X, tau_N;
        Eigen::Vector2d u = Binv * controlInputs;                         // Conpute control inputs for propellers
        Eigen::Vector2d n_c = u.array().sign() * u.array().abs().sqrt();  // Convert to required propellar speed
        //std::cout << "hello6" << std::endl;
        
        //Storing SIM data
        simdata(i, Eigen::seq(0, 11)) = x.transpose();  
        simdata(i, 12) = r_d;                           
        simdata(i, 13) = psi_d;                         
        //std::cout << "hello7" << std::endl;

        //rk4 method for x(k+1)
        //std::cout << "size x:       " << x << std::endl;
        //std::cout << "size n_input: " << n_input.size() << std::endl;
        rk4_ran_step(x, n_input, mp, rp, V_c, beta_c, h);
        //std::cout << "size x:       " << x << std::endl;
        //std::cout << "size n_input: " << n_input.size() << std::endl;

        //Euler's method
        if ((n_c - n).isZero()) {
            std::cout << "Error: Division by zero" << std::endl;
            break;
        }
        
        n = n + h/T_n * (n_c - n);              // Update propeller speeds
        z_psi = z_psi + h * ssa(psi - psi_d);   // Update integral state for heading control

        std::cout << "Iteration: " << i << ", Time: " << t[i] << "s, x: " << xn << "m, y: " << yn << "m, psi: " << psi << "rad" << std::endl;
    }
    std::cout<<"Simulation completed"<<std::endl;
    storeSimulationData(simdata);

    return 0;
}



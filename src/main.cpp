#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <atomic>
#include <csignal>
#include <cstdio>     // fprintf, fflush
#include <unistd.h>   // _exit (alt)

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

#include "Observers/EKF13.hpp"
#include "Observers/EKF15.hpp"
#include "Observers/EKF18.hpp"
#include "Observers/nn_ekf_v11.hpp"
#include "Observers/nn_ekf_v12.hpp"
#include "Observers/quatObserver.hpp"
#include "Observers/observer_selector.hpp"


#include "Planning/plan_selector.hpp"
#include "Planning/straight_line_planning.hpp"
#include "Planning/fermat_spiral_planning.hpp"

#include "Sensors/GNSS.hpp"
#include "Sensors/IMU.hpp"

#include "Utilities/calculations.hpp"
#include "Utilities/plotting.hpp"

using Eigen::Vector3d;
using Eigen::Matrix3d;

static std::atomic<bool> g_stop{false};

extern "C" void on_sigint(int) {
    static std::atomic<bool> first{true};
    if (first.exchange(false)) {
        g_stop.store(true, std::memory_order_relaxed);   // ask loop to stop
        // arm "second Ctrl-C = immediate" behavior:
        struct sigaction sa{};
        sa.sa_handler = [](int){ _exit(130); };
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        return;
    }
    _exit(130); // second ^C: hard exit
}

static void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // ensure blocking syscalls return EINTR
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

// Optional: your own cleanup (join threads, close files/sockets, flush logs).
static void graceful_shutdown() {
    // example:
    // if (writer.joinable()) writer.join();
    // out.flush(); out.close();
    std::fflush(nullptr); // flush all C stdio streams
}

static double surgeTauSchedule(int wpt_index) {
    if (wpt_index <= 7)   return 100.0;
    if (wpt_index <= 10)  return 50.0;
    if (wpt_index <= 16)  return 120.0;
    if (wpt_index <= 22)  return 160.0;
    if (wpt_index <= 35)  return 250.0;

    // Waypoints 36–43 (8 waypoints total): alternate 25 and 100, switching every 2 waypoints
    if (wpt_index <= 43) {
        int block = (wpt_index - 36) / 2;        // 0 for 36–37, 1 for 38–39, 2 for 40–41, 3 for 42–43
        return (block % 2 == 0) ? 80.0 : 100.0;  // 36–37:25, 38–39:100, 40–41:25, 42–43:100
    }

    // From waypoint 44 onward: alternate 50 and 150, switching every 2 waypoints
    {
        int block = (wpt_index - 44) / 2;        // 0 for 44–45, 1 for 46–47, ...
        return (block % 2 == 0) ? 200.0 : 150.0;  // 44–45:50, 46–47:150, 48–49:50, ...
    }
}

// V_c: random walk with a weak pull toward 0 (so it doesn’t drift off forever)
static double oceanCurrentV(double V_c, double dt, std::mt19937& gen)
{
    static std::normal_distribution<double> N(0.0, 1.0);
    if (!(dt > 0.0)) return V_c;

    const double sigmaV = 2e-4; // (m/s)/sqrt(s)
    const double kV     = 8e-5; // 1/s (weak pull toward 0)

    return V_c + (-kV * V_c) * dt + sigmaV * std::sqrt(dt) * N(gen);
}

// beta_c: pure angular random walk (no pull), wrapped by ssa()
static double oceanCurrentB(double beta_c, double dt, std::mt19937& gen)
{
    static std::normal_distribution<double> N(0.0, 1.0);
    if (!(dt > 0.0)) return ssa(beta_c);

    const double sigmaB = deg2rad(0.3); // rad/sqrt(s)

    return ssa(beta_c + sigmaB * std::sqrt(dt) * N(gen));
}

int main() {
    install_signal_handlers();
    bool interrupted = false;
    srand(time(0)); 
    YAML::Node config = YAML::LoadFile("../config.yaml");

    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen(123456u);

    // Distributions for each o (std dev)
    std::normal_distribution<double> noise_nu(0.0, 0.01);  
    std::normal_distribution<double> noise_eta(0.0, 0.001);

    // Simulation parameters
    double h = config["simulation"]["h"].as<double>();
    double T_final = config["simulation"]["T_final"].as<double>();
 
    //Load condition
    double mp = config["load_condition"]["mp"].as<double>(); 
    
    // Ocean current
    double V_c    = config["ocean_current"]["V_c"].as<double>(); 
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
    x(6) = wpt[0].x; // Xn (LON/East)
    x(7) = wpt[0].y; // Yn (LAT/North)
    x(11) = std::atan2(wpt[1].x - wpt[0].x, wpt[1].y - wpt[0].y);
    Eigen::VectorXd x_est = x;
    Eigen::Vector3d v_end_est = Eigen::Vector3d::Zero();

    Eigen::VectorXd xdot = Eigen::VectorXd::Zero(12);

    // Sensor Measurements

    Eigen::Vector3d lever_arm_port_body( -2, -1, -1.5 ); //Measure!
    Eigen::Vector3d lever_arm_stbd_body( -2,  1, -1.5 ); //Measure!

    Eigen::Vector3d ant1_meas = raw_GNSS(x, lever_arm_port_body, gen, 0.5*0.5);
    Eigen::Vector3d ant2_meas = raw_GNSS(x, lever_arm_stbd_body, gen, 0.5*0.5);
    Eigen::Vector3d nav_pos_1 = origin_from_raw_GNSS(x, ant1_meas, lever_arm_port_body);
    Eigen::Vector3d nav_pos_2 = origin_from_raw_GNSS(x, ant2_meas, lever_arm_stbd_body);
    double psi_gnss = gnss_heading_from_two_antennas(ant1_meas, ant2_meas);
    bool have_gnss_now = false;

    Eigen::Vector3d ba = Eigen::Vector3d::Constant(0);//1e-4);     // accel bias state (m/s^2)
    Eigen::Vector3d bgyro = Eigen::Vector3d::Constant(0);//1e-5);  // gyro  bias state (rad/s)
    const double acc_nd  = 1e-4;  // 1.2e-3 m/s^2 / sqrt(Hz)  (~122 µg/√Hz)
    const double gyro_nd = 1e-6;  // 7.0e-5 rad/s  / sqrt(Hz) (~0.24 °/√hr)
    IMUData imu = raw_IMU(x, xdot, gen, ba, bgyro, h, acc_nd, gyro_nd);

    Eigen::Vector3d acc_bias_est = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias_est = Eigen::Vector3d::Zero();

    // Observers
    const double gnss_period = 0.5;
    //   imu_period = h

    EKF18 ekf18;

    EKF15 ekf15;
    Eigen::Vector3d r0_end(x(6), x(7), x(8));  
    ekf15.initObserverEKF15(x(11), r0_end);

    EKF13 ekf13(h, gnss_period);  
    {
        Eigen::Matrix<double,EKF13::NX,1> x0_ekf13;
        x0_ekf13.setZero();

        x0_ekf13(0) = x(6);       
        x0_ekf13(1) = x(7);                
        x0_ekf13(6) = psi_gnss;    

        ekf13.setState(x0_ekf13);
    }

    //--------------- Attitude observer -----------------
    qobs::Config qcfg;
    qcfg.Ki = Matrix3d::Zero(); // gyro-bias integral gain
    qcfg.Ki(0,0) = 0.01;
    qcfg.Ki(1,1) = 0.01;
    qcfg.Ki(2,2) = 0.002;

    qcfg.k1  = 1.0;                         // accel vector gain
    qcfg.k2  = 0.8;                         // heading (compass) gain
    qcfg.accel_min_norm = 1e-6;             // guard against near-zero |f|
    qcfg.mag_min_norm   = 1e-6;             // unused in 7-DOF, but fine

    const double Xn0    = x_est(6);
    const double Yn0    = x_est(7);
    const double Zn0    = x_est(8);
    const double phi0   = x_est(9);
    const double theta0 = x_est(10);
    const double psi0   = ssa(x_est(11));

    Quat q_nb0 = quatFromEulerEND(phi0, theta0, psi0);
    Mat3 R_nb0 = RnbFromQuatCustom(q_nb0);

    qobs::QuatObserver quatObs(qcfg);
    quatObs.setQuat(q_nb0);                    
    quatObs.setBiasGyro(Eigen::Vector3d::Zero());
    Quat q_nb = q_nb0;
    Vec3 w_est{0.0, 0.0, 0.0};

    // --------------EKF observer v11---------------
    nnqekf_v11::Config_v11 cfg_v11;
    cfg_v11.g               = 9.81;
    cfg_v11.sigma_a         = 5e-3;     // m/s^2 (IMU accel white noise)
    cfg_v11.sigma_ba_rw     = 3e-6;     // m/s^2/s (accel bias RW)
    cfg_v11.tau_ba          = 0;        // set >0 if you want bias leak; else 0
    cfg_v11.chi2_gate_pos3  = 16.27;    // ~95% gate in 3D
    cfg_v11.chi2_gate_vec3  = -1.0;     // disable velocity gate (like v9)

    nnqekf_v11::NN_qObs_Aided_EKF_v11 ekf_v11(cfg_v11);

    // Provide initial attitude as BODY→NAV (END convention)
    ekf_v11.setRotationNavFromBody(R_nb0);

    // Initial state/cov
    nnqekf_v11::State9_v11 x0_v11;
    x0_v11.p   = Eigen::Vector3d(Xn0, Yn0, Zn0);  // END position (Down positive)
    x0_v11.v.setZero();
    x0_v11.b_a.setZero();

    Eigen::Matrix<double,9,9> P0_v11 = Eigen::Matrix<double,9,9>::Identity();
    P0_v11.block<3,3>(0,0) *= 10;//0.01;   // pos
    P0_v11.block<3,3>(3,3) *= 10;//0.01;   // vel
    P0_v11.block<3,3>(6,6) *= 0.1;  // accel bias

    ekf_v11.setState(x0_v11, P0_v11);   // <-- fix: use ekf_v10, not ekf_v9

    // (Optional but recommended) align heave equilibrium with your start Down:
    ekf_v11.setHeaveEquilibrium(Zn0);   // so spring term is zero at t0

    // NN init (use the streaming one-step **stateful** members)
    static nnqekf_v11::NN_v11 nn_v11;
    bool ok_v11 = nn_v11.init(
        "data/model00_v11_ens4_h001_wd4_lr5_qw64_seq256_d3_sign/ts",              // <-- stateful files live here
        "data/model00_v11_ens4_h001_wd4_lr5_qw64_seq256_d3_sign/norm_used.json",         // <-- the same norm used in training
        /*use_cuda=*/true);
    if (!ok_v11) { std::cerr << "[NNv11] init failed; running without NN.\n"; }

    // Gate fusion until we’ve buffered seq_len samples; after that fuse every stride steps.
    const int nn_warmup_len = 256;   // e.g. 20 samples of warmup
    const int nn_stride     = 1;    // fuse every sample after warmup
    ekf_v11.setNN(&nn_v11, nn_warmup_len, nn_stride);

    // -------------- EKF observer v12 ---------------
    nnqekf_v12::Config_v12 cfg_v12;
    cfg_v12.g               = 9.81;
    cfg_v12.sigma_a         = 5e-3;     // m/s^2 (IMU accel white noise)
    cfg_v12.sigma_ba_rw     = 3e-6;     // m/s^2/s (accel bias RW)
    cfg_v12.tau_ba          = 0.0;      // set >0 if you want bias leak; else 0
    cfg_v12.chi2_gate_pos3  = 16.27;    // gate for position update
    cfg_v12.chi2_gate_vec3  = -1.0;     // disable vec3 gate

    nnqekf_v12::NN_qObs_Aided_EKF_v12 ekf_v12(cfg_v12);

    // Provide initial attitude as BODY→NAV (END convention)
    ekf_v12.setRotationNavFromBody(R_nb0);

    // Initial state/cov
    nnqekf_v12::State9_v12 x0_v12;
    x0_v12.p   = Eigen::Vector3d(Xn0, Yn0, Zn0);  // END position (Down positive)
    x0_v12.v.setZero();
    x0_v12.b_a.setZero();

    nnqekf_v12::Mat99 P0_v12 = nnqekf_v12::Mat99::Identity();
    P0_v12.block<3,3>(0,0) *= 10;   // pos
    P0_v12.block<3,3>(3,3) *= 10;   // vel
    P0_v12.block<3,3>(6,6) *= 0.1;  // accel bias

    ekf_v12.setState(x0_v12, P0_v12);

    // (Optional but recommended) align heave equilibrium with your start Down:
    ekf_v12.setHeaveEquilibrium(Zn0);

    // NN init (use the streaming one-step **stateful** members)
    static nnqekf_v12::NN_v12 nn_v12;
    bool ok_v12 = nn_v12.init(
        "data/model00_v12_ens4_h001_wd4_lr5_qw64_seq256_d3_euler/ts",
        "data/model00_v12_ens4_h001_wd4_lr5_qw64_seq256_d3_euler/norm_used.json",
        /*use_cuda=*/true);

    if (!ok_v12) {
    std::cerr << "[NNv12] init failed; running without NN.\n";
    } else {
    // Gate fusion until we’ve buffered seq_len samples; after that fuse every stride steps.
    const int nn_warmup_len = 256;
    const int nn_stride     = 1;
    ekf_v12.setNN(&nn_v12, nn_warmup_len, nn_stride);
    }


    //Observer selector
    ObserverSelector selector;
    ObserverKind observer_type = selector.select();
    const bool use_gnss = selector.use_gnss();

    // Control system variables
    std::vector<double> tau_XYN_c = {0.0, 0.0, 0.0};
    std::vector<double> tau_XYN = {0.0, 0.0, 0.0};
    std::vector<double> control_allocation = {0.0, 0.0, 0.0, 0.0};
    Eigen::Vector2d n_c = {0.0, 0.0};
    Eigen::Vector2d alpha_c = {0.0, 0.0};

    // Model;   
    RAN ran_model;
    ran_model.update(x, mp, V_c, beta_c, h, n_c, alpha_c);
    xdot = ran_model.get_xdot(); 

    double T_n = ran_model.getT_n();          // Propeller time constant (s)
    double T_alpha = ran_model.getT_alpha();  // Azimuth angle time constant (s)

    ran_model.select_failure_mode();
    std::vector<bool> failstate = ran_model.check_failstate();

    Eigen::VectorXd thrustCoeffs = ran_model.getThrustCoeffs();
    std::cout << "Thrust Coefficients (" << thrustCoeffs.size() << "): "
          << thrustCoeffs.transpose() << '\n';


    Eigen::Vector2d n = Eigen::Vector2d::Zero(); // Propeller speeds (rad/s)      
    Eigen::Vector2d alpha = Eigen::Vector2d::Zero(); // Azimuth angles (rad)

    // Wave setup
    ran_model.enable_waves(true);
    const double wn_xy=1.5, z_xy=0.25, wn_psi=2.0, z_psi=0.35;
    const double K_xy  = 0.10 * std::sqrt(4.0*z_xy*wn_xy);                    // 0.10 m RMS
    const double K_psi = (0.25*M_PI/180.0) * std::sqrt(4.0*z_psi*wn_psi);     // 0.25 deg RMS

    ran_model.set_wave_params(
        wn_xy,  z_xy,  K_xy,     // surge-like WF
        wn_xy,  z_xy,  K_xy,     // sway-like WF
        wn_psi, z_psi, K_psi,    // yaw-like WF
        0.20, 0.20, 1.00         // drift sigma [X,Y,N] (BODY) 
    );

    // Model est
    RAN ran_model_est;

    if (failstate[0] == true){ran_model_est.fail_state_n1();} 
    else {ran_model_est.recover_n1();}

    if (failstate[1] == true){ran_model_est.fail_state_n2();} 
    else {ran_model_est.recover_n2();}

    ran_model_est.update(x_est, mp, V_c, beta_c, h, n_c, alpha_c);
    Eigen::MatrixXd M_est = ran_model_est.get_M();
    Eigen::MatrixXd B_est = ran_model_est.get_B();
    double U_est = ran_model_est.get_U();

    // Create the straight line path.
    StraightLinePath straightLinePath;
    straightLinePath.updateWaypoints(wpt);
    Waypoints pathLine = straightLinePath.samplePath(0.05);
    std::cout << "Path size: " << pathLine.size() << std::endl;
    plotPath(wpt, pathLine);

    // Create the Fermat spiral path.
    // - Set the curvature constraint (k_max in rad/m).
    double kappa_max = 0.10; 
    FermatSpiralPath spiral(kappa_max);
    spiral.updateWaypoints(wpt);
    Waypoints pathFS = spiral.samplePath(0.05);
    std::cout << "Path size: " << pathFS.size() << std::endl;
    plotPath(wpt, pathFS);

    // Initialize guidance methods and LOS observer 
    ALOS ALOS(Delta_h, gamma_h, 0.1);
    LOSObserver losObserver(h, K_f, x_est(11));

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
    MPC_Control_System mpc_control(20, 0.1); 

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
    Eigen::MatrixXd simdata(num_steps, 76);         
    
    RealTimePlotter plotter;
    if (pathType == 1 || pathType == 2) {
        plotter.setSampledPath(pathLine);
    }
    else if (pathType == 3) {
        plotter.setSampledPath(pathFS);
    }

    bool break_flag = false;
    std::vector<double> wpt_change_times;        

    // Time since last update. Init to x to get first update. 
    static double gnss_time = 1;
    static double planning_time = 2;
    static double guidance_control_time = 0.1;
    static double plotting_time = 2;

    // Main simulation loop
    for (int i = 0; i < num_steps; ++i) {
        t[i] = i * h;

        Eigen::VectorXd tau_full = ran_model_est.tau_pods(n, alpha);
        tau_XYN = {tau_full(0), tau_full(1), tau_full(5)};

        //Comment out when making dataset
        //V_c    = oceanCurrentV(V_c, h, gen);
        //beta_c = oceanCurrentB(beta_c, h, gen);

        ran_model.wave_step_WF(h);

        imu = raw_IMU(x, xdot, gen, ba, bgyro, h, acc_nd, gyro_nd);

        have_gnss_now = false;
        gnss_time += h;
        if (gnss_time >= 0.1) { // 2 Hz gnss updates
            do { gnss_time -= 0.1; } while (gnss_time >= 0.1);

            const auto& eta6 = ran_model.get_wave_eta6(); // END frame [x y z φ θ ψ]

            // Wave-corrupted pose used ONLY for pose/position measurements
            Eigen::VectorXd x_w = x;
            // Apply WF displacement to NAV position and yaw
            x_w(6) += eta6(0);                 // East
            x_w(7) += eta6(1);                 // North
            x_w(11) = ssa(x(11) + eta6(5));    // yaw

            // Simulate raw GNSS measurements for both antennas
            ant1_meas = raw_GNSS(x_w, lever_arm_port_body, gen, 0.02);
            ant2_meas = raw_GNSS(x_w, lever_arm_stbd_body, gen, 0.02);

            // Transform raw_GNSS position to origin position.
            nav_pos_1 = origin_from_raw_GNSS(x_est, ant1_meas, lever_arm_port_body);
            nav_pos_2 = origin_from_raw_GNSS(x_est, ant2_meas, lever_arm_stbd_body);

            // Compute heading from the two GNSS measurements
            psi_gnss = gnss_heading_from_two_antennas(ant1_meas, ant2_meas);
            have_gnss_now = !std::isnan(psi_gnss);
            if (!have_gnss_now) {
            std::cerr << "NaN psi_gnss at i=" << i << ", t=" << t[i] << "s\n";
            }
        }

        // Deadreconing
        if (!use_gnss){
            have_gnss_now = false;
            ekf_v11.zeroAccelBias();
            ekf_v12.zeroAccelBias();
        }

        // 1) External attitude (q-Obs)
        if (have_gnss_now) {
            quatObs.step7DOF(h, imu.accel, imu.gyro, psi_gnss); //add accel bias
        } else {
            quatObs.step6DOF(h, imu.accel, imu.gyro); //add accel bias
        }
        q_nb =  quatObs.quat();
        w_est = quatObs.w_est();   
        gyro_bias_est = quatObs.bias_gyro();

        // ------------------ Switch between observers ------------------
        switch (observer_type) {
            case ObserverKind::TrueState: {
                x_est = x;
                v_end_est = RnbFromEuler(x(9), x(10), x(11)) * x.segment<3>(0);

                break;
            }

            case ObserverKind::EKF13: {

                EKF13::Input u13; 
                u13.ax = imu.accel(0); u13.ay = imu.accel(1); u13.az = imu.accel(2);
                u13.wx = imu.gyro(0);  u13.wy = imu.gyro(1);  u13.wz = imu.gyro(2);

                std::optional<EKF13::GnssMeas> g13; 
                if (have_gnss_now) {
                    const double px = 0.5*(nav_pos_1.x() + nav_pos_2.x());
                    const double py = 0.5*(nav_pos_1.y() + nav_pos_2.y());
                    g13.emplace();          
                    g13->px  = px;
                    g13->py  = py;
                    g13->psi = psi_gnss;    
                }

                ekf13.step(u13, g13);

                x_est = ekf13.getState12();
                break;
            }

            case ObserverKind::EKF18: {

                ekf18.predict(imu.accel, h);       
                ekf18.updateGyro(imu.gyro);        

                if (have_gnss_now) {
                    Eigen::Vector3d pos1(nav_pos_1.x(), nav_pos_1.y(), 0.0);
                    Eigen::Vector3d pos2(nav_pos_2.x(), nav_pos_2.y(), 0.0);
                    ekf18.updatePos(pos1);
                    ekf18.updatePos(pos2);
                    ekf18.updateHeading(psi_gnss);
                }

                x_est = ekf18.getState12();
                break;
            }
            case ObserverKind::EKF15: {

                // Predict with IMU
                ekf15.predict(imu.accel, imu.gyro, h);

                if (have_gnss_now) {
                    ekf15.updatePos(ant1_meas, lever_arm_port_body);
                    ekf15.updatePos(ant2_meas, lever_arm_stbd_body);
                    ekf15.updateHeading(psi_gnss);
                }

                // Build x_est = [u v w p q r x y z phi theta psi]
                Eigen::Matrix<double,12,1> x_est_local;
                auto xhat = ekf15.getState();

                // u, v, w (body)
                x_est_local.segment<3>(0) = xhat.segment<3>(0);

                // p, q, r from gyro 
                acc_bias_est = xhat.segment<3>(9);
                gyro_bias_est = xhat.segment<3>(12);         
                Eigen::Vector3d pqr = imu.gyro - gyro_bias_est;         
                x_est_local.segment<3>(3) = pqr;

                // x, y, z (END)
                x_est_local.segment<3>(6) = xhat.segment<3>(3);

                // phi, theta, psi
                x_est_local(9)  = xhat(6);
                x_est_local(10) = xhat(7);
                x_est_local(11) = xhat(8);

                x_est = x_est_local;  
                v_end_est = RnbFromEuler(x(9), x(10), x(11)) * x_est_local.segment<3>(0);
                break;
            }
            case ObserverKind::nn_EKF_v11: {
                ekf_v11.setRotationFromQuat(q_nb);     // BODY->END (custom END conv)

                // 2) Propagate using IMU (acceleration and gyro)
                ekf_v11.propagate(imu.gyro, imu.accel, h);

                // 3) NN giving pseudo correction for velocity estimates)
                ekf_v11.feedNN(imu.accel, q_nb, tau_XYN[0], tau_XYN[1], tau_XYN[2]);
                
                // 4) GNSS position giving position and velocity corrections:  t[i] for testing 
                if (have_gnss_now || (t[i]<0)) {
                    Eigen::Matrix3d Rpos_port = Eigen::Matrix3d::Identity() * std::pow(0.5, 2); 
                    Eigen::Matrix3d Rpos_stbd = Eigen::Matrix3d::Identity() * std::pow(0.5, 2);
                    ekf_v11.updateGnssPos(ant1_meas, Rpos_port, lever_arm_port_body, 1);
                    ekf_v11.updateGnssPos(ant2_meas, Rpos_stbd, lever_arm_stbd_body, 1);
                }

                // 5) Read state
                v_end_est = ekf_v11.get_end_vel_est(); //For plotting only
                acc_bias_est = ekf_v11.get_acc_bias_est();
                //gyro_bias_est = quatObs.bias_gyro();
                x_est = ekf_v11.getState12(gyro_bias_est);   // [u v w p q r x y z phi theta psi]^T
                break;
            }
            case ObserverKind::nn_EKF_v12: {
                // 1) Attitude (BODY->END)
                ekf_v12.setRotationFromQuat(q_nb);

                // 2) Propagate using IMU (dt must be time step)
                ekf_v12.propagate(imu.gyro, imu.accel, h);

                // 3) NN pseudo-measurement update (END velocity)
                ekf_v12.feedNN(imu.accel, q_nb, tau_XYN[0], tau_XYN[1], tau_XYN[2]);

                // 4) GNSS position updates: t[i] for testing
                if (have_gnss_now || (t[i] < 0)) {
                    Eigen::Matrix3d Rpos_port = Eigen::Matrix3d::Identity() * (0.5 * 0.5);
                    Eigen::Matrix3d Rpos_stbd = Eigen::Matrix3d::Identity() * (0.5 * 0.5);
                    ekf_v12.updateGnssPos(ant1_meas, Rpos_port, lever_arm_port_body, 1);
                    ekf_v12.updateGnssPos(ant2_meas, Rpos_stbd, lever_arm_stbd_body, 1);
                }

                // 5) Read state
                v_end_est = ekf_v12.get_end_vel_est();
                acc_bias_est = ekf_v12.get_acc_bias_est();
                //gyro_bias_est = quatObs.bias_gyro();
                x_est = ekf_v12.getState12(gyro_bias_est);
                break;
            }
        }

        double u     = x_est(0);  // Surge velocity (BODY frame)
        double v     = x_est(1);  // Sway velocity  (BODY frame)
        double w     = x_est(2);  // Heave velocity (BODY frame)
        double p     = x_est(3);  // Roll rate      (BODY frame)
        double q     = x_est(4);  // Pitch rate     (BODY frame)
        double r     = x_est(5);  // Yaw rate       (BODY frame)
    
        double xn    = x_est(6);  // East position   (END frame)
        double yn    = x_est(7);  // North position  (END frame)
        double zn    = x_est(8);  // Down position   (END frame)
        double phi   = ssa(x_est(9));  // Roll angle      (END frame)
        double theta = ssa(x_est(10)); // Pitch angle     (END frame)
        double psi   = ssa(x_est(11)); // Heading angle   (END frame)


        // ------------------------------ Update model real dynamics and estimates ------------------------------
        ran_model.update(x, mp, V_c, beta_c, h, n, alpha);

        ran_model_est.update(x_est, mp, V_c, beta_c, h, n, alpha);
        M_est = ran_model_est.get_M(); // Const?
        U_est = ran_model_est.get_U(); // Dependent on x.
        B_est = ran_model_est.get_B(); // Dependent on alpha

        // ------------------------------ Mode switch ------------------------------

        // - Switch criteria for path following to DP mode. 
        // if (GuidanceFlag != 1) {
        //     if (R_switch > std::sqrt(std::pow(xn - wpt[wpt.size()-1].x, 2) + std::pow(yn - wpt[wpt.size()-1].y, 2))) {
        //         if (std::abs(ssa(psi_d-psi)) < deg2rad(3) && U_est < 0.01) {
        //             if (wpt_index < wpt.size()-1) {
        //                 wpt_index += 1;
        //                 MIMO_PID.reset();
        //                 wpt_change_times.push_back(t[i]);
        //             }
        //             else {
        //                 std::cout << "Reached the last waypoint." << std::endl;
        //                 break_flag = true;
        //             }
        //         }
        //     }
        //     if (path_x == wpt.back().x && path_y == wpt.back().y) {
        //         GuidanceFlag = 1; 
        //         pathType = 1;
        //         wpt_index = wpt.size()-1;
        //     }
        // }

        // ------------------------------ Path planning: connecting waypoints ------------------------------
        planning_time += h;
        if (planning_time >= 2) { // 2 Hz gnss updates
            do { planning_time -= 2; } while (planning_time >= 2);
            switch (pathType) {
                case 1: { // Dynamic Positioning.
                    if (R_switch > std::sqrt(std::pow(xn - wpt[wpt_index].x, 2) + std::pow(yn - wpt[wpt_index].y, 2))){
                        if (std::abs(ssa(psi_d-psi)) < deg2rad(5) && U_est < 0.05) {
                            if (wpt_index < wpt.size()-1) {
                                wpt_index += 1;
                                MIMO_PID.reset();
                                wpt_change_times.push_back(t[i]);
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

            
        }

        // ------------------------------ Guidance laws ------------------------------
        guidance_control_time += h;
        if (guidance_control_time >= 0.1) { // 10 Hz control update
            do { guidance_control_time -= 0.1; } while (guidance_control_time >= 0.1);
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
                nu << u, v, w, p, q, r;
                eta  << xn, yn, zn, phi, theta, psi;
                tau_XYN_c = MIMO_PID.update(h, xn_d, yn_d, psi_d, M_est, eta, nu, V_c, beta_c);
            } 
            // - Motion Control: Path following: 
            else if (GuidanceFlag==2 || GuidanceFlag==3) { 
                tau_XYN_c[0] = surgeTauSchedule(wpt_index);
                tau_XYN_c[1] = 0;
                tau_XYN_c[2] = 2*headPID.update(h, M_est, psi, psi_d, r, r_d, a_d);
            }

            // - Control allocation
            switch (ControlAllocFlag) {
                case 1: { // Pseudo-inverse control allocation
                    control_allocation = pseudo_inverse_allocation(tau_XYN_c, B_est, 880, 880);
                    n_c     = {control_allocation[0], control_allocation[2]};
                    alpha_c = {control_allocation[1], control_allocation[3]};
                    break;
                }
                case 2: { // Nonlinear optimization with constraints
                    control_allocation = NLOptControlAlloc(tau_XYN_c[0], tau_XYN_c[1], tau_XYN_c[2], U_est, n, alpha, failstate);
                    n_c     = {control_allocation[0], control_allocation[2]};
                    alpha_c = {control_allocation[1], control_allocation[3]};
                    break;
                }
                case 3: { // Nonlinear optimization with constraints over a horizon taking rate constriants into account
                    control_allocation = MPC_control_alloc(tau_XYN_c[0], tau_XYN_c[1], tau_XYN_c[2], U_est, T_n, T_alpha, n, alpha, failstate);
                    n_c     = {control_allocation[0], control_allocation[2]};
                    alpha_c = {control_allocation[1], control_allocation[3]};
                    break;
                }
                case 4: { // Model Predictive Control System (Motion control and control allocation using vessel model)
                    std::vector<double> x0 = {xn, yn, psi, u, v, r}; 
                    mpc_control.solve(x0, wpt[wpt_index-1].x, wpt[wpt_index-1].y, xn_d, yn_d, psi_d, V_c, beta_c, n, alpha, failstate);
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
        }
        // ------------------------------ State updates ------------------------------

        // Marine Craft Model, update states: x
        ran_model.wave_step_drift(h);
        ran_model.rk4(x, mp, V_c, beta_c, h, n, alpha);
        x(11) = ssa(x(11)); //makes plotting look bad    
        
        // Pod model, update states: n and alpha
        ran_model.update_n(n, n_c, h);
        ran_model.update_alpha(alpha, alpha_c, h);

        // ------------------------------ Plotting and Info ------------------------------

        // Show SIM progress once in a while
        plotting_time += h;
        if (plotting_time >= 1) { // 2 Hz gnss updates
            do { plotting_time -= 1; } while (plotting_time >= 1);
    
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

            plotter.updatePlot(x(6), x(7), x(11), x_est(6), x_est(7), x_est(11), 0.2, GuidanceVectorX, GuidanceVectorY);
        }

        if ((i % 100) == 0) {
            const double psi_raw   = yawFromQuatEND(q_nb);
            const double psi_canon = yawFromQuatEND(q_nb);
            const double psi_ekf   = x_est(11);  // from R_nb_ inside EKF
            std::cerr << "[att] psi_raw=" << rad2deg(psi_raw)
                    << " deg, psi_can=" << rad2deg(psi_canon)
                    << " deg, psi_ekf=" << rad2deg(psi_ekf)
                    << " deg\n";
        }

        if (i % 100 == 0) {
            std::cout << std::fixed << std::setprecision(0)
            << "################################################" << std::endl
            << "Iteration: " << i << ", Time: " << floor(t[i]/60) << "min, " << fmod(t[i], 60) << "s, " <<std::endl
            << "------------------------------------------------" << std::endl
            << "Path type: " << pathType << ", Guidance flag: " << GuidanceFlag << ", Control flag: " << ControlAllocFlag << std::endl
            << "wpt index: " << wpt_index
            << ", current wpt: (" << wpt[wpt_index].x << ", " << wpt[wpt_index].y << ")" << std::endl
            << "Failstate: [" << failstate[0] << ", " << failstate[1]
            << std::fixed << std::setprecision(4)
            << "], V_c: " << V_c << ", beta_c: " << beta_c << std::endl
            << std::fixed << std::setprecision(1)
            << "------------------------------------------------" << std::endl
            << "closest point: " << closest.point.pos.x << ", " << closest.point.pos.y << std::endl
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
            std::cout << "xn_est:   " << xn << "m, yn_est:   " << yn << "m, psi_est:   " << rad2deg(psi) << "deg" << ", U_est: " << U_est << std::endl
            << "------------------------------------------------" << std::endl
            //<< "CO offset (x, y, z): " << CO_Offset(U).transpose() << std::endl
            //<< "------------------------------------------------" << std::endl
            << std::fixed << std::setprecision(4)
            << "n_c(0), n_c(1):         " << n_c(0) << ", " << n_c(1) << std::endl
            << "n(0),   n(1):           " << n(0) << ", " << n(1) << std::endl
            << "------------------------------------------------" << std::endl
            << std::fixed << std::setprecision(2)
            << "alpha_c(0), alpha_c(1): " << rad2deg(alpha_c(0)) << ", " << rad2deg(alpha_c(1)) << std::endl
            << "alpha(0), alpha(1):     " << rad2deg(alpha(0)) << ", " << rad2deg(alpha(1)) << std::endl
            << "------------------------------------------------" << std::endl
            << "tauX_c, tauY_c, tauN_c: " << tau_XYN_c[0] << ", " << tau_XYN_c[1] << ", " << tau_XYN_c[2] << std::endl
            << "tauX,   tauY,   tauN  : " << tau_XYN[0] << ", " << tau_XYN[1] << ", " << tau_XYN[2] << std::endl
            << "------------------------------------------------" << std::endl
            << "Nav data GNSS; " << std::endl
            << "Pos 1 (x,y,z): " << nav_pos_1(0) << ", " << nav_pos_1(1) << ", " << nav_pos_1(2) << std::endl
            << "Pos 2 (x,y,z): " << nav_pos_2(0) << ", " << nav_pos_2(1) << ", " << nav_pos_2(2) << std::endl
            << "HDG   (deg)  : " << rad2deg(psi_gnss)    << std::endl
            << "------------------------------------------------" << std::endl
            << "Nav data IMU : "  << std::endl
            << std::fixed << std::setprecision(8)
            << "Accelerometer (body frame): " << imu.accel.transpose() << " m/s^2" << std::endl
            << "Gyroscope (body frame)    : " << imu.gyro.transpose() << " rad/s" << std::endl
            << "Accel bias ba (body frame): " << ba.transpose()       << " m/s^2" << std::endl
            << "Gyro  bias bgyro (body frame): " << bgyro.transpose() << " rad/s" << std::endl
            << std::fixed << std::setprecision(4)
            << "" << std::endl
            << "------------------------------------------------" << std::endl
            << "True state:      " << x(0) << ", " << x(1) << ", " << x(2) << ", " << x(3) << ", " << x(4) << ", " << x(5) 
            << ", " << x(6) << ", " << x(7) << ", " << x(8) << ", " << x(9) << ", " << x(10) << ", " << x(11) << std::endl 
            << "Estimated state: " << x_est(0) << ", " << x_est(1) << ", " << x_est(2) << ", " << x_est(3) << ", " << x_est(4) << ", " << x_est(5)
            << ", " << x_est(6) << ", " << x_est(7) << ", " << x_est(8) << ", " << x_est(9) << ", " << x_est(10) << ", " << x_est(11) << std::endl 
            << "------------------------------------------------" << std::endl
            << "Estimated biases:" << std::endl
            << std::fixed << std::setprecision(8)
            << "Accel bias ba: " << acc_bias_est.transpose() << std::endl
            << "Gyro bias bgyro: " << gyro_bias_est.transpose() << std::endl
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
        simdata(i, 24) = tau_XYN_c[0];
        simdata(i, 25) = tau_XYN_c[1];
        simdata(i, 26) = tau_XYN_c[2];
        simdata(i, 27) = tau_XYN[0];
        simdata(i, 28) = tau_XYN[1];
        simdata(i, 29) = tau_XYN[2];
        simdata(i, 30) = closest.point.pos.x;
        simdata(i, 31) = closest.point.pos.y;
        simdata(i, 32) = closest.x_e;
        simdata(i, 33) = closest.y_e;
        simdata(i, Eigen::seq(34, 45)) = x_est.transpose();  
        simdata(i, 46) = imu.accel(0);
        simdata(i, 47) = imu.accel(1);
        simdata(i, 48) = imu.accel(2);
        simdata(i, 49) = imu.gyro(0);
        simdata(i, 50) = imu.gyro(1);
        simdata(i, 51) = imu.gyro(2);
        simdata(i, 52) = q_nb.w;  
        simdata(i, 53) = q_nb.x;
        simdata(i, 54) = q_nb.y;
        simdata(i, 55) = q_nb.z;
        simdata(i, 56) = w_est(0);
        simdata(i, 57) = w_est(1);
        simdata(i, 58) = w_est(2);
        simdata(i, Eigen::seq(59, 61)) = acc_bias_est.transpose();
        simdata(i, Eigen::seq(62, 64)) = gyro_bias_est.transpose();
        simdata(i, Eigen::seq(65, 67)) = ba.transpose();               
        simdata(i, Eigen::seq(68, 70)) = bgyro.transpose();    
        simdata(i, 71) = V_c;
        simdata(i, 72) = beta_c;
        simdata(i, Eigen::seq(73, 75)) = v_end_est.transpose();  

        if (break_flag == true) {
            for (int j = i; j < num_steps; ++j) {
                simdata.row(j) = simdata.row(i);
                simdata(j, 0) = j * h;
            }
            break;
        }

        if (g_stop.load(std::memory_order_relaxed)) {
            interrupted = true;
            break;
        }
    
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    if (interrupted) {
        // Do ONLY your own safe teardown. Avoid calling into plotting/Python/etc.
        graceful_shutdown();

        std::fprintf(stderr, "Interrupted (SIGINT). Exiting cleanly.\n");
        std::fflush(stderr);

        // Exit WITHOUT running static/global destructors (prevents segfaults)
        std::_Exit(130);  // or _exit(130)
    }

    std::cout << "Simulation completed" << std::endl;
    storeSimulationData(simdata, "simdata.csv");
    storeWaypointChangeTimes(wpt_change_times, "wpt_change_times.csv");

    plotter.finalizePlot();

    plotIMUAccelBiasCompare();
    plotIMUGyroBiasCompare();  

    plotOceanCurrent();

    if (pathType == 1 || pathType == 2) {
        //plotTrajectory(wpt, pathLine);
        plotTrajectories(wpt, pathLine);
    } else if (pathType == 3) {
        //plotTrajectory(wpt, pathFS);
        plotTrajectories(wpt, pathFS);
    }
    //plotClosestPointErrors();
    //plotStateErrors();

    plotTau();
    plotPropellerSpeeds();
    plotAlphas();

    plotIMUAccel();
    plotIMUGyro();

    plotQuaternionQnb();
    plotHeadingComparison();
    plotAngles();

    plotStateEstimateErrors();

    plotEndVelocities();
    plotEndVelocitiesVsEstimates();

    return 0;
}
# Optimal Constraint Thruster Allocation README


## Dependancy installation guide
Recreation of environment used to run this repository on a new computer; 
Install Anaconda
'''
source ~/anaconda3/bin/activate
conda --version
conda env create -f environment.yml
conda activate cpp_env
conda activate cpp_env
mkdir build && cd build
cmake ..
make
'''

## Usage of conda environment
Do this when new terminal is used;
'''
source ~/anaconda3/bin/activate thrust_alloc_cpp_env
'''

When done with terminal, close it or do
'''
conda deactivate
'''

### Dependancy update guide
Exporting environment to a YAML file; 
'''
conda env export > environment.yml
'''

Adding or updating environment.yml after chenging dependancies
'''
git add environment.yml
git commit -m "Dependacy update in environment.yml file"
git pull
git push
'''


## TODO:

### Define mission in 3 parts: 
Initialization: Where to go and what to do. 
Trajectory: Manuver to designated spot at the pier. ( TT to PF ? using ex ALOS?)
Berthing: Enter DP mode. 

 ### Fix CO to be dynamic (changing based on mission status) (DONE)
 CO in front of center for Initialization and Trajectory.
 CO at Center of boat for Berthing.
 CO can be speed dependant if we lowpass speed such that it's value is stable.
 Idea: Scale CO based on U_low_threshold < U < U_max_threshold, from mid_ship < CO < front_ship, if over or under U_low_threshold and U_max_threshold cap value to mid_ship or front_ship. 

 ### Make n relative (DONE)
 Let n be scaled between 0-1, and set k_pos = k_neg = Max Thrust (~200 kg-f)
 Get specifications for thrusters (rpm and max Thrust) from SeaDrive.
 
 ### Fix calculation ran()
 Fix the calculation of input matrix B in ran().
 Check that the numbers used in equations make sence for ran. 

 ### MPC guidance
 Not MPC for guidance, but to follow the best path possible.
 Can be calculated by a thread in the background and updated once in a while.

 Minimization based on the inputs 
 - chi_d (Desired Course angle)
 - U_d (Desired speed)
 - Tf (final time)

 chi = psi + beta_c 
 U = sqrt(u² + v²)
 beta_c = atan(v/u)
 
 Finding:
 X_d_dot = U_d * cos(chi_d)
 Y_d_dot = U_d * sin(chi_d)

 Inequality Constraints:
 abs(chi_d[k+1] - chi_d[k]) / h <= r_max (max turning)
 abs(U_d[k+1] - U_d[k]) / h <= U_dot_max (max desired acceleration)

 Equality constraints:
 X_d(0)
 Y_d(0)
 X_d(Tf)
 Y_d(Tf)
 Tf = 60 seconds (example value)
 
 ### Define all variables outside of loops (for-loop in main.cpp). 
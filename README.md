# Marine Vessel Simulator

This repository contains c++ code for simulation of a USV, named RAN.

The "Handbook of Marine Craft Hydrodynamics and Motion Control" by Thor I. Fossen was used actively for theoretical background material, as well as papers mentioned in "Optimal_Constrained_Control_Allocation_for_catamarans_with_dual_azimuth_thrusters.pdf" under the literature folder. 

The repository is implemented using c++ for fast computations and open-source purposes. 

#### Models;
- ran(), a marine reserch vessel with a catamaran hull, with one azimuth pod placed aft underneeth each hull

#### Path generation algorithms;
- Straight line path. (For path following)
- Continuous-curvature path based on "Continuous-Curvature Path Generation Using Fermat's Spiral" by Anastasios M. Lekkas, Andreas R. Dahl, Morten Breivik, Thor I. Fossen. In Modeling, Identification and Control, Vol. 34, No. 4, 2013, pp. 183–198, ISSN 1890–1328. (For path following and trajectory tracking)

#### Guidance laws; 
- Dynamic positioning (DP) (direct to wpt)
- DP using MPC (stepwize path towards wpt)
- Line-of-sight (LOS).
- Adaptive Line-of-sight (ALOS). 

#### Motion Control
- MIMO PID for force (surge X and sway Y) and moment (yaw N) control.
- Heading PID based on Nomoto model. 

#### Control Allocation
- Pseudo-iverse-based 
- Nonlinear constrained optimization - based
- MPC Thrust model - based
- MPC Full model - based


## Showcase of capabilities;


### Live plotting: 
The SIM has a class for Live plotting of the vessel position, catamaran body outline, guidance points, path traversed and the path we try to follow. 
It's a nice-to-have tool, made using matplotlibcpp, for experimentation, debugging and tuning. 
<table>
  <tr>
    <td>
      <img src="data/live-plotting/Live-DP-MPC_NLOpt.png" width="400"/>
      <div align="center">Live, 4 square test, DP, POD Model MPC Control allocation</div>
    </td>
    <td>
      <img src="data/live-plotting/Live-DP-MPC_Control.png" width="400"/>
      <div align="center">Live, 4 square test, DP, Full Model MPC Control allocation</div>
    </td>
  </tr>

  <tr>
    <td>
      <img src="data/live-plotting/Live-StraightPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Live, Straight path, LOS, POD Model Control allocation</div>
    </td>
    <td>
      <img src="data/live-plotting/Live-CurvedPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Live, Curved path, LOS, POD Model Control allocation</div>
    </td>
  </tr>
</table>

### Dynamic positioning performance: 4 square test results; 
The 4 square test, as described in ... , is a way to test the manuvering capabilities of a vessel using a given control allocation, 
by speciffying destination and the angle at which to move to the destination. In the test implemented in this SIM, the waypoint 
index in not updated before the boat is at the right position, with the desired heading and 0 speed, with a small margin for error.
The 4 parts of the this test are: 
- Straight forward: (0,0) to (0,40) with psi = 90.
- Sideways: (0,40) to (40,40) with psi = 90.
- Backwards: (40, 40) to (40, 0) wth psi = 90.
- Sideways with 45 degree angle: (40, 0) to (0,0) with psi = 135.

<div class="scrollable-table">
<table>
  <tr>
    <td>
        <img src="data/DP-pictures/DP-NLOpt.png" width="400"/>
        <div align="center">4 square test, DP, NL-Opt POD Control allocation</div>
    </td>
    <td>
        <img src="data/DP-pictures/DP-NLOpt-state_errors.png" width="400"/>
        <div align="center">state errors</div>
    </td>
    <td>
        <img src="data/DP-pictures/DP-NLOpt-Psi.png" width="400"/>
        <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>
</div>

<div class="scrollable-table">
<table>
  <tr>
    <td>
        <img src="data/DP-pictures/DP-MPC_NLOpt.png" width="400"/>
        <div align="center">4 square test, DP, Pod Model MPC Control allocation</div>
    </td>
    <td>
        <img src="data/DP-pictures/DP-MPC_NLOpt-state_errors.png" width="400"/>
        <div align="center">state errors</div>
    </td>
    <td>
        <img src="data/DP-pictures/DP-MPC_NLOpt-Psi.png" width="400"/>
        <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>
</div>

<div class="scrollable-table">
<table>
  <tr>
    <td>
        <img src="data/DP-pictures/DP-MPC_Control.png" width="400"/>
        <div align="center">4 square test, DP, Full Model Control allocation</div>
    </td>
    <td>
        <img src="data/DP-pictures/DP-MPC_Control-state_errors.png" width="400"/>
        <div align="center">state errors</div>
    </td>
    <td>
        <img src="data/DP-pictures/DP-MPC_Control-Psi.png" width="400"/>
        <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>
</div>

### Path following performance;
We test manuverability using 3 simple tests, each showing the performance on a
straight line path and smooth path. For path Following LOS is used with a lookahead of 10 
from the closest point on the path from the vessel along the tangent line on the path at that point. 

#### Right turn
<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-StraightPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Straight path, LOS, NL-Opt Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-StraightPath-LOS-NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td> 
    <td>
      <img src="data/path-following-pictures/right-turn/Right-StraightPath-LOS-NLOpt-Psi.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td> 
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-CurvedPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Curved path, LOS, NL-Opt Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-CurvedPath-LOS-NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-CurvedPath-LOS-NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs psi_d</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-StraightPath-LOS-MPC_NLOpt.png" width="400"/>
      <div align="center">Straight path, LOS, MPC Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-StraightPath-LOS-MPC_NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-StraightPath-LOS-MPC_NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-CurvedPath-LOS-MPC_NLOpt.png" width="400"/>
      <div align="center">Curved path, LOS, MPC Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-CurvedPath-LOS-MPC_NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/right-turn/Right-CurvedPath-LOS-MPC_NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>

#### Left turn
<table>
  <tr>
    <td>
      <img src="data/Left-StraightPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Straight path, LOS, NL-Opt Control alloc</div>
    </td>
    <td>
      <img src="data/Left-StraightPath-LOS-NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/Left-CurvedPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Curved path, LOS, NL-Opt Control alloc</div>
    </td>
    <td>
      <img src="data/Left-CurvedPath-LOS-NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/Left-StraightPath-LOS-MPC.png" width="400"/>
      <div align="center">Straight path, LOS, MPC Control alloc</div>
    </td>
    <td>
      <img src="data/Left-StraightPath-LOS-MPC-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/Left-CurvedPath-LOS-MPC_MPC.png" width="400"/>
      <div align="center">Curved path, LOS, MPC Control alloc</div>
    </td>
    <td>
      <img src="data/Left-CurvedPath-LOS-MPC_MPC-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
  </tr>
</table>

#### Zigzag
Straight line:
<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-StraightPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Straight path, LOS, NL-Opt Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-StraightPath-LOS-NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-StraightPath-LOS-NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs psi_d</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-CurvedPath-LOS-NLOpt.png" width="400"/>
      <div align="center">Curved path, LOS, NL-Opt Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-CurvedPath-LOS-NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-CurvedPath-LOS-NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-StraightPath-LOS-MPC_NLOpt.png" width="400"/>
      <div align="center">Straight path, LOS, MPC Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-StraightPath-LOS-MPC_NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-StraightPath-LOS-MPC_NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-CurvedPath-LOS-MPC_NLOpt.png" width="400"/>
      <div align="center">Curved path, LOS, MPC Control alloc</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-CurvedPath-LOS-MPC_NLOpt-x_y_error.png" width="400"/>
      <div align="center">along track (x_e) and cross track (y_e) error</div>
    </td>
    <td>
      <img src="data/path-following-pictures/zigzag-pictures/Zigzag-CurvedPath-LOS-MPC_NLOpt-Psi.png" width="400"/>
      <div align="center">Psi vs Psi_d</div>
    </td>
  </tr>
</table>


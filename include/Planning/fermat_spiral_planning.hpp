#ifndef FERMAT_SPIRAL_PATH_HPP
#define FERMAT_SPIRAL_PATH_HPP

#include "Utilities/calculations.hpp"
#include <cmath>
#include <vector>

//----------------------------------------------------------
// FermatSpiralPath Class Declaration
//----------------------------------------------------------
// Generates a path/trajectory combining straight line segments and Fermat's spiral curves.
//
// Based on the paper "Continuous-Curvature Path Generation Using Fermat's Spiral"
// by Anastasios M. Lekkas, Andreas R. Dahl, Morten Breivik, Thor I. Fossen.
// Modeling, Identification and Control, Vol. 34, No. 4, 2013, pp. 183–198, ISSN 1890–1328
//----------------------------------------------------------
class FermatSpiralPath {
    public:
        FermatSpiralPath(double kappa_max);
    
        void updateWaypoints(const Waypoints& waypoints);
    
        // Path Following:
        // - Finds the point minimizing cross_track error on 4 "active" segments,
        //   and returns the point with less positional error of the 4.
        //   When on the last segment, the waypoint index is updated. To ensure 
        //   a continous path, the line segments overlap by one waypoint.
        PathTrackingInfo getClosestPoint(const Vector2D vessel_position, int& index);
    
        // Trajectory tracking:
        // - Not yet implemented
    
        // Data access
        Waypoints samplePath(double delta); //Add const
    
        void printParameters() const;
    
    private:
        // Input waypoints.
        Waypoints waypoints_;
    
        // Maximum curvature.
        double kappa_max_;
        // Maximum theta given kappa_max.
        double theta_kappa_max;

        Vector2D stored_pull_out;
    
        // Structure to store FS parameters
        struct FSParameters {
            // - |delta chi|
            double delta_chi; 
            // - Turning direction (+1 for anticlockwise, -1 for clockwise)
            double rho;    
            // - Initial course angle for forward FS.
            double chi0;
            // - Final course angle for forward FS (mirrored segment uses this)
            double chi_end;
            // - Solution of theta + arctan(2*theta) = |delta chi|
            double theta_end; 
            // - u_max = sqrt(theta_end)
            double u_max;    
            //Meeting point of two FS segments
            double u_mid;
            //Midpoint position of the FS segment
            Vector2D midpoint_pos;
            // - Scaling constant.
            double k;         
            // - Starting point and end point of the spiral.
            Vector2D wheel_over, pull_out;
            // - Previous waypoint.
            double x0, y0;
            // - Next waypoint.
            double x_end, y_end; 
        };
    
    
        // Compute FS parameters from three waypoints (A, B, C).
        FSParameters computeFSParameters(const Vector2D &A, const Vector2D &B, const Vector2D &C) const;
    
        // Parameterization of FS and FS mirrored curves
        // - Forward FS segment; lambda > 0 
        // - Mirrored FS segment; lambda < 0
        Vector2D computeSpiralPoint(double u, double base_angle, double lambda,
                                    double k, double rho, double x, double y, double u_max) const;
    
        Vector2D computeSpiralDerivative(double u, double base_angle, double lambda,
                                         double k, double rho, double u_max) const;
    
        Vector2D computeSpiralSecondDerivative(double u, double base_angle, double lambda,
                                               double k, double rho, double u_max) const;
    
    
        // Helper functions.
        double f(double theta, double delta_chi) const;
        double fprime(double theta) const;
        double fsecond(double theta) const;
    
        double computeThetaEnd(double delta_chi) const;
        double computeScalingConstant(double theta_kappa_max, double kappa_max) const;
        double computeDistanceForCorner(double k, double theta_end, double delta_chi) const;
    
        PathPoint projectOntoLine(const Vector2D &A, const Vector2D &B, const Vector2D &vessel);
        PathPoint projectOntoSpiral(Vector2D vessel, FSParameters params, double lamda);
    
        double alongTrackError(Vector2D vessel, PathPoint path_point);
        double crossTrackError(Vector2D vessel, PathPoint path_point);
    
    };
    
    // Add this non-member function after the Vector2D class definition
    inline Vector2D operator*(double s, const Vector2D& v) {
        return v * s;  // Reuse the existing Vector2D * double operator
    }

#endif
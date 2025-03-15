#ifndef CLOTHOID_SMOOTHER_HPP
#define CLOTHOID_SMOOTHER_HPP

#include <vector>
#include <cmath>
#include "utilities.hpp"  // For Point2D

/** 
 * A class for smoothing a piecewise linear path (waypoints) with curvature-continuous 
 * Fermat's Spiral (FS) transitions. Generates a G^2-continuous path within a curvature limit.
 */
class ClothoidSmoother {
public:
    /** Container for a path point with curvature (for validation or sampling results). */
    struct PathPoint { 
        double x; 
        double y; 
        double curvature; 
    };

    /**
     * Constructor.
     * @param maxCurvature The maximum allowable curvature (inverse of minimum turn radius).
     */
    ClothoidSmoother(double maxCurvature);

    /**
     * Smooths the given waypoints into a curvature-continuous path.
     * @param waypoints Input sequence of waypoints (must contain at least 2 points).
     * @return A sequence of path points (with curvature values) describing the smoothed path.
     *         This includes sampled points on each FS segment and the original waypoints.
     */
    std::vector<PathPoint> smoothPath(const std::vector<Point2D>& waypoints);

    /**
     * Extract only position data from PathPoints to create Waypoints.
     * @param pathPoints Input sequence of PathPoints with curvature.
     * @return A sequence of waypoints (Point2D) without curvature.
     */
    static Waypoints extractWaypoints(const std::vector<PathPoint>& pathPoints);

private:
    double maxCurv;  // maximum curvature constraint

    /**
     * Solves for the FS parametric length (theta_end) required for a given half turn angle.
     * Uses the equation χ(θ) = θ + atan(2θ), solved via bisection.
     * @param halfTurnAngle The half of the total deflection angle (in radians) that a single FS segment must achieve.
     * @return θ_end such that the course change χ(θ_end) ≈ halfTurnAngle.
     */
    double solveThetaForHalfAngle(double halfTurnAngle) const;

    /**
     * Computes the curvature of an FS segment at a given parametric length θ for scaling constant k.
     * @param theta The FS parameter θ.
     * @param k Scaling constant of the FS.
     * @param turnSign +1 for left turn, -1 for right turn.
     * @return The curvature value κ at parameter θ.
     */
    double computeCurvature(double theta, double k, int turnSign) const;

    /**
     * Generates a sequence of PathPoints along a Fermat's spiral segment from start to end.
     * @param start Starting point of the FS segment.
     * @param startDir Unit direction vector of the entry line (direction of travel at start).
     * @param end    Ending point of the FS segment.
     * @param endDir Unit direction vector of the exit line (direction of travel at end).
     * @param k      Scaling constant for the FS.
     * @param theta_end Parametric length of the half spiral (from 0 to θ_end).
     * @param turnSign +1 for left turn, -1 for right turn.
     * @param outPoints Vector to append the sampled PathPoints (with curvature).
     * @param samples Number of sample points to generate along the segment (including endpoints).
     */
    void sampleFermatSpiral(const Point2D& start, const Point2D& startDir,
                            const Point2D& end,   const Point2D& endDir,
                            double k, double theta_end, int turnSign,
                            std::vector<PathPoint>& outPoints, int samples = 50) const;
};

#endif // CLOTHOID_SMOOTHER_HPP



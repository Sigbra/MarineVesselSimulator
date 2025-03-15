#include "clothoid_smoother.hpp"
#include <stdexcept>
#include <cassert>

ClothoidSmoother::ClothoidSmoother(double maxCurvature)
    : maxCurv(maxCurvature) 
{
    if (maxCurv <= 0) {
        throw std::invalid_argument("maxCurvature must be positive.");
    }
}

double ClothoidSmoother::solveThetaForHalfAngle(double halfTurnAngle) const {
    // Use bisection to solve θ + atan(2θ) = halfTurnAngle (&#8203;:contentReference[oaicite:9]{index=9}).
    double lo = 0.0;
    double hi = (halfTurnAngle + 1.0);  // initial upper bound guess (a bit above halfTurnAngle)
    for (int iter = 0; iter < 50; ++iter) {
        double mid = 0.5 * (lo + hi);
        double chi = mid + atan(2.0 * mid);  // χ(mid)
        if (chi < halfTurnAngle) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

double ClothoidSmoother::computeCurvature(double theta, double k, int turnSign) const {
    // Compute FS curvature κ(θ) using polar derivative formula&#8203;:contentReference[oaicite:10]{index=10}.
    if (theta <= 0.0) {
        return 0.0;
    }
    // For numerical stability, use the explicit curvature formula from the polar representation:
    // κ(θ) = (2θ^2 + ... ) / (k^2 * sqrt(θ) * (1+4θ^2)^(3/2)), derived from&#8203;:contentReference[oaicite:11]{index=11}.
    double t2 = theta * theta;
    double numerator = 3.0 + 4.0 * t2;
    double denominator = pow(1.0 + 4.0 * t2, 1.5);
    double kappa = numerator / denominator;
    kappa /= (k * k * sqrt(theta));
    // turnSign is not needed for magnitude, since we return absolute curvature (sign can be inferred from turn direction).
    return kappa;
}

void ClothoidSmoother::sampleFermatSpiral(const Point2D& start, const Point2D& startDir,
                                          const Point2D& end,   const Point2D& endDir,
                                          double k, double theta_end, int turnSign,
                                          std::vector<PathPoint>& outPoints, int samples) const {
    if (samples < 2) samples = 2;
    // Ensure output includes the end point once.
    if (!outPoints.empty()) {
        outPoints.pop_back();
    }

    // Construct orthonormal local frame for start (x along startDir, y to left of startDir).
    // startDir assumed unit length.
    Point2D u = startDir; // unit tangent at start
    // Compute left-normal (perpendicular) unit vector for startDir
    Point2D nL = { -u.y, u.x };  // rotate startDir 90° left
    // Similarly, for end, define a local frame for orientation interpolation (not directly used in sampling).
    Point2D v = endDir;
    Point2D nL_end = { -v.y, v.x };

    // Sample points along the FS param θ from 0 to θ_end
    for (int i = 0; i < samples; ++i) {
        double t = theta_end * (double)i / (samples - 1);
        // Parametric coordinates (polar form): r = k * sqrt(t)
        double r = k * sqrt(t);
        double angle = turnSign * t;        // polar angle relative to startDir
        double x_local = r * cos(angle);
        double y_local = r * sin(angle);
        // Transform local (x_local, y_local) to global coordinates:
        // Global = start + x_local * u + y_local * nL
        Point2D p;
        p.x = start.x + x_local * u.x + y_local * nL.x;
        p.y = start.y + x_local * u.y + y_local * nL.y;
        // Compute curvature at this point (using param t)
        double curv = computeCurvature(t, k, turnSign);
        outPoints.push_back({ p.x, p.y, curv });
    }
}

std::vector<ClothoidSmoother::PathPoint> 
ClothoidSmoother::smoothPath(const Waypoints& waypoints) {
    if (waypoints.size() < 2) {
        throw std::invalid_argument("At least two waypoints are required.");
    }
    std::vector<PathPoint> smoothedPath;
    smoothedPath.reserve(waypoints.size() * 50); // reserve rough space

    // Utility lambda: compute unit direction vector from p1 to p2
    auto unitDir = [](const Point2D& p1, const Point2D& p2) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double len = sqrt(dx*dx + dy*dy);
        return Point2D{ dx/len, dy/len };
    };

    // Start with the first waypoint
    smoothedPath.push_back({ waypoints[0].x, waypoints[0].y, 0.0 });

    for (size_t i = 1; i < waypoints.size(); ++i) {
        if (i == waypoints.size() - 1) {
            // Last leg: straight line from previous segment end to final waypoint.
            smoothedPath.push_back({ waypoints[i].x, waypoints[i].y, 0.0 });
        } else {
            // Smoothing at waypoint i (with prev=i-1, next=i+1).
            Point2D P_prev = waypoints[i-1];
            Point2D P_curr = waypoints[i];
            Point2D P_next = waypoints[i+1];
            // Compute directions of incoming and outgoing segments
            Point2D d1 = unitDir(P_prev, P_curr);
            Point2D d2 = unitDir(P_curr, P_next);
            // Calculate turn angle and direction (sign) between d1 and d2
            // Using dot and cross to find angle
            double cross = d1.x * d2.y - d1.y * d2.x;
            double dot = d1.x * d2.x + d1.y * d2.y;
            int turnSign = (cross >= 0) ? 1 : -1;  // left turn if cross positive, right if negative
            double turnAngle = acos(std::max(-1.0, std::min(1.0, dot)));
            if (turnSign < 0) {
                // For right turn, angle stays positive but sign indicates direction
            }
            // If turnAngle is near zero (collinear), no smoothing needed; continue as straight line
            if (turnAngle < 1e-6) {
                continue;
            }
            // Compute half-angle and param length θ_end for half transition
            double halfAngle = 0.5 * turnAngle;
            double theta_end = solveThetaForHalfAngle(halfAngle);
            // Determine scaling k such that max curvature <= maxCurv.
            // Curvature at θ_end (end of half FS) is highest on that spiral&#8203;:contentReference[oaicite:12]{index=12}, use it to scale.
            double baseCurv = computeCurvature(theta_end, 1.0, turnSign);
            double k = (baseCurv > maxCurv ? baseCurv / maxCurv : baseCurv / maxCurv);
            // If baseCurv < maxCurv, we can tighten curve (smaller k) to use higher curvature up to maxCurv,
            // so set k = baseCurv/maxCurv (<1). If baseCurv > maxCurv, then we need larger k to reduce curvature.
            if (baseCurv > 0) {
                k = baseCurv / maxCurv;
            } else {
                k = 1.0; // straight case (shouldn't happen if turnAngle > 0)
            }

            // Compute offset distance along each leg to start/end FS (use symmetric formula d = k√θ_end cos θ_end).
            double d_offset = k * sqrt(theta_end) * cos(theta_end);
            // Entry and exit points for FS transition
            Point2D P_entry = { P_curr.x - d_offset * d1.x, P_curr.y - d_offset * d1.y };
            Point2D P_exit  = { P_curr.x + d_offset * d2.x, P_curr.y + d_offset * d2.y };

            // Sample first half of FS (increasing curvature) from P_entry into the turn apex.
            std::vector<PathPoint> firstHalf;
            sampleFermatSpiral(P_entry, d1, P_curr, /*not used*/ d2, k, theta_end, turnSign, firstHalf, 25);
            // Sample second half (decreasing curvature) from P_exit backward to the turn apex.
            // We can generate it from P_entry as well but reverse direction.
            std::vector<PathPoint> secondHalf;
            // Use reversed direction for second half: define direction into the turn (opposite of d2)
            Point2D d2_back = { -d2.x, -d2.y };
            sampleFermatSpiral(P_exit, d2_back, P_curr, /*not used*/ d1, k, theta_end, turnSign, secondHalf, 25);

            // The two halves both include the apex (at their end points). We need to connect them.
            // We will trim the last point of firstHalf (apex) as secondHalf includes it as first point.
            if (!firstHalf.empty()) firstHalf.pop_back();
            // Append second half to first half
            firstHalf.insert(firstHalf.end(), secondHalf.begin(), secondHalf.end());
            // Append to smoothedPath (skip duplicating the entry point which is last of previous segment)
            smoothedPath.insert(smoothedPath.end(), firstHalf.begin(), firstHalf.end());
        }
    }
    return smoothedPath;
}

Waypoints ClothoidSmoother::extractWaypoints(const std::vector<PathPoint>& pathPoints) {
    Waypoints waypoints;
    waypoints.reserve(pathPoints.size());
    
    for (const auto& point : pathPoints) {
        waypoints.push_back({point.x, point.y});
    }
    
    return waypoints;
}

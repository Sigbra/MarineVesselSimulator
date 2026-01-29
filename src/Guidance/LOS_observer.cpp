#include "Guidance/LOS_observer.hpp"
#include "Utilities/calculations.hpp"
#include <cmath>

LOSObserver::LOSObserver(double h, double K_f, double LOSAngle)
    : h(h), K_f(K_f), LOSangle(LOSAngle), LOSrate(0.0) {
    T_f = 1.0 / (K_f + 2 * std::sqrt(K_f) + 1);
}

void LOSObserver::update(double LOScommand) {
    // Step k
    xi = LOSangle - LOSrate;

    // Step k+1
    LOSangle += h * (LOSrate + K_f * ssa(LOScommand - LOSangle));

    double PHI = std::exp(-h / T_f);          
    xi = PHI * xi + (1 - PHI) * LOSangle;

    LOSrate = LOSangle - xi;
}

double LOSObserver::getLOSAngle() const {
    return ssa(LOSangle);
}

double LOSObserver::getLOSRate() const {
    return LOSrate;
}
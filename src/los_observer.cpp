#include "los_observer.hpp"
#include "utilities.hpp"
#include <cmath>

LOSObserver::LOSObserver(double h, double K_f)
    : h(h), K_f(K_f), LOSangle(0.0), LOSrate(0.0) {
    T_f = 1.0 / (K_f + 2 * std::sqrt(K_f) + 1);
    xi = LOSangle - LOSrate;
}

void LOSObserver::update(double LOScommand) {
    double PHI = std::exp(-h / T_f);
    LOSangle += h * (LOSrate + K_f * ssa(LOScommand - LOSangle));
    xi = PHI * xi + (1 - PHI) * LOSangle;
    LOSrate = LOSangle - xi;
}

double LOSObserver::getLOSAngle() const {
    return LOSangle;
}

double LOSObserver::getLOSRate() const {
    return LOSrate;
}
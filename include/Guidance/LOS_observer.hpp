#ifndef LOSObserver_HPP
#define LOSObserver_HPP

class LOSObserver {
    public:
        // Constructor
        LOSObserver(double h, double K_f, double LOSAngle);
    
        // Update function for the LOS observer
        void update(double LOScommand);
    
        // Getters for LOS angle and LOS rate
        double getLOSAngle() const;
        double getLOSRate() const;
    
    private:
        double h;        // Sampling time
        double K_f;      // Observer gain
        double T_f;      // Differentiator time constant
        double LOSangle; // Estimated LOS angle
        double LOSrate;  // Estimated LOS rate
        double xi;       // Internal differentiator state
    };

#endif
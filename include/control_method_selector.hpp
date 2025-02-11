#ifndef CONTROLMETHOD_HPP
#define CONTROLMETHOD_HPP

#include <vector>
#include <string>

class ControlMethod {
public:
    // Constructor
    ControlMethod(const std::vector<std::string>& methods);
    
    // Displays the selection menu and returns the chosen method index
    int selectMethod();

private:
    std::vector<std::string> methods; // Available control methods
};

#endif // CONTROLMETHOD_HPP
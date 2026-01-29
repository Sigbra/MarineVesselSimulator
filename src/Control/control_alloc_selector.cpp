#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include "Control/control_alloc_selector.hpp"

ControlAllocationMethod::ControlAllocationMethod(){}
    
int ControlAllocationMethod::selectMethod() {
    std::cout << "Choose Control Allocation Method:" << std::endl;
    for (size_t i = 0; i < methods.size(); ++i) {
        std::cout << i + 1 << ". " << methods[i] << std::endl;
    }
    
    int choice = 0;
    while (true) {
        std::cout << "Enter the number of your choice: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
        }
        else if (choice > 0 && choice <= static_cast<int>(methods.size())) {
            break;
        }
        else {
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    return choice;
}
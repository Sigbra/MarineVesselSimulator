#include "Planning/plan_selector.hpp"
#include <iostream>
#include <limits>

int selectPathType() {
    std::cout << "Choose Path Type:" << std::endl;
    std::cout << "1. DP path." << std::endl;
    std::cout << "2. Straight Line Path." << std::endl;
    std::cout << "3. Continuous-Curvature Path Using Fermat's Spiral." << std::endl;
    
    int choice = 0;
    while (true) {
        std::cout << "Enter the number of your choice: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
        }
        else if (choice >= 1 && choice <= 3) {
            break;
        }
        else {
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    return choice;
}
#include <iostream>
#include <vector>
#include <string>

class ControlMethod {
    public:
        ControlMethod(const std::vector<std::string>& methods);
        int selectMethod();
    
    private:
        std::vector<std::string> methods;
    };
    
    ControlMethod::ControlMethod(const std::vector<std::string>& methods)
        : methods(methods) {}
    
    int ControlMethod::selectMethod() {
        std::cout << "Choose Control Method:" << std::endl;
        for (size_t i = 0; i < methods.size(); ++i) {
            std::cout << i + 1 << ". " << methods[i] << std::endl;
        }
        
        int choice = 0;
        while (true) {
            std::cout << "Enter the number of your choice: ";
            std::cin >> choice;
            if (choice > 0 && choice <= static_cast<int>(methods.size())) {
                break;
            }
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
        return choice;
    }

// Example usage
// int main() {
//     std::vector<std::string> methods = {"PID Control", "Adaptive Control", "Model Predictive Control"};
//     ControlMethodSelector selector(methods);
//     int selectedMethod = selector.selectMethod();
//     std::cout << "You selected: " << methods[selectedMethod - 1] << std::endl;
//     return 0;
// }
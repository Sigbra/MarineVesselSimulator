#include "Observers/observer_selector.hpp"
#include <iostream>
#include <limits>

ObserverSelector::ObserverSelector()
: options_({"True states (observer bypass)", "EKF13", "EKF18", "EKF15", "NN EKF V11", "NN EKF V12"})
{}

ObserverKind ObserverSelector::select() const {
  std::cout << "Choose Navigation Observer:\n";
  for (size_t i = 0; i < options_.size(); ++i) {
    std::cout << "  " << (i + 1) << ". " << options_[i] << "\n";
  }

  int choice = 0;
  while (true) {
    std::cout << "Enter the number of your choice: ";
    std::cin >> choice;

    if (std::cin.fail()) {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      std::cout << "Invalid input. Please enter a number.\n";
      continue;
    }
    if (choice >= 1 && choice <= static_cast<int>(options_.size())) break;

    std::cout << "Invalid choice. Please try again.\n";
  }

  int gnss_choice = 1;
  while (true) {
    std::cout << "Use GNSS measurements? (1=yes, 0=no): ";
    std::cin >> gnss_choice;

    if (std::cin.fail()) {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      std::cout << "Invalid input. Please enter 1 or 0.\n";
      continue;
    }
    if (gnss_choice == 0 || gnss_choice == 1) break;

    std::cout << "Invalid choice. Please enter 1 or 0.\n";
  }
  use_gnss_ = (gnss_choice == 1);

  switch (choice) {
    case 1:  return ObserverKind::TrueState;
    case 2:  return ObserverKind::EKF13;
    case 3:  return ObserverKind::EKF18;
    case 4:  return ObserverKind::EKF15;
    case 5:  return ObserverKind::nn_EKF_v11;
    case 6:  return ObserverKind::nn_EKF_v12;
    default: return ObserverKind::TrueState;
  }
}
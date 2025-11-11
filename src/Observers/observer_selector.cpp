#include "Observers/observer_selector.hpp"
#include <iostream>
#include <limits>

ObserverSelector::ObserverSelector()
: options_({"True states (observer bypass)", "EKF13", "EKF18", "NN EKF V1", "NN EKF V2",
   "NN EKF V3", "NN EKF V4", "NN EKF V5", "NN EKF V6", "NN EKF V7", "NN EKF V8", "NN EKF V9", "NN EKF V10", "NN EKF V11"})
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

  switch (choice) {
    case 1:  return ObserverKind::TrueState;
    case 2:  return ObserverKind::EKF13;
    case 3:  return ObserverKind::EKF18;
    case 4:  return ObserverKind::nn_EKF_v1;
    case 5:  return ObserverKind::nn_EKF_v2;
    case 6:  return ObserverKind::nn_EKF_v3;
    case 7:  return ObserverKind::nn_EKF_v4;
    case 8:  return ObserverKind::nn_EKF_v5;
    case 9:  return ObserverKind::nn_EKF_v6;
    case 10: return ObserverKind::nn_EKF_v7;
    case 11: return ObserverKind::nn_EKF_v8;
    case 12: return ObserverKind::nn_EKF_v9;
    case 13: return ObserverKind::nn_EKF_v10;
    case 14: return ObserverKind::nn_EKF_v11;
    default: return ObserverKind::TrueState;
  }
}


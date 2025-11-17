#ifndef OBSERVER_SELECTOR_HPP
#define OBSERVER_SELECTOR_HPP

#pragma once
#include <string>
#include <vector>

// Three options: current EKF18, EKF13, and a placeholder for a future observer.
enum class ObserverKind {TrueState = 1, EKF13 = 2, EKF18 = 3, nn_EKF_v1 = 4, nn_EKF_v2 = 5, nn_EKF_v3 = 6,
   nn_EKF_v4 = 7, nn_EKF_v5 = 8, nn_EKF_v6 = 9, nn_EKF_v7 = 10, nn_EKF_v8 = 11, nn_EKF_v9 = 12, nn_EKF_v10 = 13, nn_EKF_v11 = 14};

// Small, self-contained selector that asks on stdin which observer to use.
class ObserverSelector {
public:
  ObserverSelector();

  // Blocks until a valid choice is entered and returns the chosen kind.
  ObserverKind select() const;

  // Utility: get display names (index aligned with menu numbering 1..N).
  const std::vector<std::string>& options() const { return options_; }

private:
  std::vector<std::string> options_;
};

#endif // OBSERVER_SELECTOR_HPP

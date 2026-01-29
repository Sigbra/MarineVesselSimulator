#ifndef OBSERVER_SELECTOR_HPP
#define OBSERVER_SELECTOR_HPP

#pragma once
#include <string>
#include <vector>

// Three options: current EKF18, EKF13, and a placeholder for a future observer.
enum class ObserverKind {TrueState = 1, EKF13 = 2, EKF18 = 3, EKF15 = 4, nn_EKF_v11 = 5, nn_EKF_v12 = 6};

// Small, self-contained selector that asks on stdin which observer to use.
class ObserverSelector {
public:
  ObserverSelector();

  // Blocks until a valid choice is entered and returns the chosen kind.
  ObserverKind select() const;

  // NEW: Whether GNSS measurements are enabled (chosen during select()).
  bool use_gnss() const { return use_gnss_; }

  // Utility: get display names (index aligned with menu numbering 1..N).
  const std::vector<std::string>& options() const { return options_; }

private:
  std::vector<std::string> options_;

  // NEW: stored configuration choice (mutable so select() can stay const).
  mutable bool use_gnss_{true};
};

#endif // OBSERVER_SELECTOR_HPP
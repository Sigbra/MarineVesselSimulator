import numpy as np
import matplotlib.pyplot as plt

def cubic_fit(thrust, force):
    # Design matrix: [x^3, x^2, x]
    X = np.vstack([thrust**3, thrust**2, thrust]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, force, rcond=None)
    return coeffs  # [a, b, c]

def cubic_eval(x, coeffs):
    a, b, c = coeffs
    return a * x**3 + b * x**2 + c * x

if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    data = np.loadtxt("../data/bollard_pull_data.csv", delimiter=',', skiprows=1)
    thrust = data[:, 0]
    force = data[:, 1]

    # Use only data in [-1, 1]
    mask = (thrust >= -0.8) & (thrust <= 0.8)
    thrust_filt = thrust[mask]
    force_filt = force[mask]

    # -----------------------------
    # Fit cubic model
    # -----------------------------
    coeffs = cubic_fit(thrust_filt, force_filt)
    print("Cubic coefficients [a, b, c]:", coeffs)

    # -----------------------------
    # Evaluate fit
    # -----------------------------
    x_fit = np.linspace(-1, 1, 500)
    y_fit = cubic_eval(x_fit, coeffs)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8,6))
    plt.scatter(thrust_filt, force_filt, color='blue', s=15, label='Data')
    plt.plot(x_fit, y_fit, 'r-', label='Cubic fit (p(0)=0)')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Thrust (normalized)')
    plt.ylabel('Force (kg)')
    plt.title('Cubic Polynomial Least-Squares Fit (p(0)=0)')
    plt.legend()
    plt.grid(True)
    plt.show()

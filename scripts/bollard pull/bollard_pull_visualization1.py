import numpy as np
import matplotlib.pyplot as plt

def constrained_poly_fit(thrust, force, order=6):
    # Design matrix: x^order down to x^1, no constant term
    X = np.vstack([thrust**i for i in range(order, 0, -1)]).T
    
    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, force, rcond=None)
    return coeffs  # coefficients from x^7 down to x^1

def poly_eval(x, coeffs):
    order = len(coeffs)
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        power = order - i
        y += c * x**power
    return y

if __name__ == "__main__":
    data = np.loadtxt("../data/bollard_pull_data.csv", delimiter=',', skiprows=1)
    thrust = data[:, 0]
    force = data[:, 1]

    # Filter to -1 to 1 range
    # (Using 0.8 because battery power limited thrust signal to 80% during bollard pull tests)
    mask = (thrust >= -0.75) & (thrust <= 0.75)
    thrust_filt = thrust[mask]
    force_filt = force[mask]

    # Fit 7th order polynomial with p(0)=0
    coeffs = constrained_poly_fit(thrust_filt, force_filt, order=5)
    print("Fitted coefficients (x^5 to x^1):", coeffs)

    # Generate fitted curve
    x_fit = np.linspace(-1, 1, 500)
    y_fit = poly_eval(x_fit, coeffs)

    # Plot results
    plt.scatter(thrust_filt, force_filt, color='blue', label='Data points')
    plt.plot(x_fit, y_fit, color='red', label='5th order fit (p(0)=0)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Thrust')
    plt.ylabel('Force')
    plt.title('5th Order Polynomial Fit with p(0)=0')
    plt.legend()
    plt.grid(True)
    plt.show()
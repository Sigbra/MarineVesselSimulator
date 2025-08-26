import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def quintic_fit_casadi(thrust, force):
    """
    Fit a quintic polynomial p(x) = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x
    to data using CasADi least-squares solver.
    """
    # Decision variables: [a, b, c, d, e]
    coeffs = ca.MX.sym('coeffs', 5)
    
    # Construct polynomial for all data points
    thrust_mx = ca.MX(thrust)
    force_mx = ca.MX(force)
    p = (coeffs[0]*thrust_mx**5 + coeffs[1]*thrust_mx**4 +
         coeffs[2]*thrust_mx**3 + coeffs[3]*thrust_mx**2 + coeffs[4]*thrust_mx)
    
    # Least-squares cost
    cost = ca.sumsqr(p - force_mx)
    
    # Solve as NLP
    nlp = {'x': coeffs, 'f': cost}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=[0,0,0,0,0])
    return np.array(sol['x']).flatten()

def quintic_eval_casadi(x, coeffs):
    a, b, c, d, e = coeffs
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x

if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    data = np.loadtxt("../data/bollard_pull_data.csv", delimiter=',', skiprows=1)
    thrust = data[:, 0]
    force = data[:, 1]

    # Use only data in [-0.8, 0.8]
    mask = (thrust >= -0.8) & (thrust <= 0.8)
    thrust_filt = thrust[mask]
    force_filt = force[mask]

    # -----------------------------
    # Fit quintic model using CasADi
    # -----------------------------
    coeffs = quintic_fit_casadi(thrust_filt, force_filt)
    print("Quintic coefficients [a, b, c, d, e]:", coeffs)

    # -----------------------------
    # Evaluate fit
    # -----------------------------
    x_fit = np.linspace(-1, 1, 500)
    y_fit = quintic_eval_casadi(x_fit, coeffs)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8,6))
    plt.scatter(thrust_filt, force_filt, color='blue', s=15, label='Data')
    plt.plot(x_fit, y_fit, 'r-', label='Quintic fit (CasADi, p(0)=0)')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Thrust (normalized)')
    plt.ylabel('Force (kg)')
    plt.title('Quintic Polynomial Least-Squares Fit using CasADi')
    plt.legend()
    plt.grid(True)
    plt.show()

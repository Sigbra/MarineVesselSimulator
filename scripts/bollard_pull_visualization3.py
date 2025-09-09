import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def spline_fit_casadi(thrust, force):
    """
    Return a CasADi MX function representing a cubic spline through the data.
    """
    thrust = list(thrust)          # convert to Python list of floats
    force = ca.MX(force)           # CasADi MX
    kind = "cubic"                 # interpolation type
    extrapolate = True              # allow values outside the data range

    # Create the interpolant
    spline = ca.interp1d(thrust, force, kind, extrapolate)
    return spline

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
    # Fit spline model using CasADi
    # -----------------------------
    spline_func = spline_fit_casadi(thrust_filt, force_filt)

    # -----------------------------
    # Evaluate fit
    # -----------------------------
    x_fit = np.linspace(-0.8, 0.8, 500)
    y_fit = np.array(spline_func(ca.MX(x_fit))).flatten()  # symbolic evaluation

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8,6))
    plt.scatter(thrust_filt, force_filt, color='blue', s=15, label='Data')
    plt.plot(x_fit, y_fit, 'r-', label='Spline fit (CasADi)')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Thrust (normalized)')
    plt.ylabel('Force (kg)')
    plt.title('Cubic Spline Fit using CasADi')
    plt.legend()
    plt.grid(True)
    plt.show()

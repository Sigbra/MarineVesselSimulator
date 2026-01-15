import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("../data/bollard_pull_data.csv", delimiter=',', skiprows=1)
    thrust = data[:, 0]
    force = data[:, 1]

    # Filter to -1 to 1 range
    mask = (thrust >= -1) & (thrust <= 1)
    thrust_filt = thrust[mask]
    force_filt = force[mask]

    # Plot only the data
    plt.scatter(thrust_filt, force_filt, color='blue', label='Data points')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Relative Thrust ' + r'$n \in [-1, 1]$')
    plt.ylabel('Pull in kg')
    plt.title('Bollard Pull Data')
    plt.legend()
    plt.grid(True)
    plt.show()

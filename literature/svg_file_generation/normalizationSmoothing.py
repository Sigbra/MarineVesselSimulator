import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define the range of a values (zoomed around zero)
    a = np.linspace(-0.1, 0.1, 10000)
    epsilon = 1e-4

    # Compute |a| and |â a|
    abs_a = np.sqrt(a**2)
    abs_hat = np.sqrt(a**2 + epsilon)

    # Plot both curves
    plt.figure()
    plt.plot(a, abs_a, label='$|a|$')
    plt.plot(a, abs_hat, label='$|\\hat{a}|$')
    plt.xlabel('a')
    plt.ylabel('Value')
    plt.title('Comparison of $|a|$ vs. $|\\hat{a}|$')
    plt.legend()
    #plt.grid(True)
    # Zoom in on the origin
    plt.xlim(-0.075, 0.075)
    plt.ylim(0, 0.05)
    plt.tight_layout()
    plt.savefig('normalizationSmoothing.svg', format='svg')
    plt.show()

if __name__ == '__main__':
    main()

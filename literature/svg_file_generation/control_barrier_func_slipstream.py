import numpy as np
import matplotlib.pyplot as plt

# Define the alpha range from -90° to 90° in radians
alpha = np.linspace(-np.pi/2, np.pi/2, 400)

# Define the two functions
f1 = np.exp(-((alpha + np.pi/2)**2) / 0.2)
f2 = np.exp(-((alpha - np.pi/2)**2) / 0.2)

# Create the plot
plt.figure()  
plt.plot(alpha, f1, label='$f_1(\\alpha)$')     
plt.plot(alpha, f2, label='$f_2(\\alpha)$') 
plt.xlabel(r'$\alpha$ [radians]') 
plt.ylabel(r'f$(\alpha)$')  
plt.legend()
plt.tight_layout()

# Save as SVG
file_path = 'control_barrier_func_slipstream.svg'
plt.savefig(file_path, format='svg')

file_path

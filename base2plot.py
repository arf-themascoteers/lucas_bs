import numpy as np
import matplotlib.pyplot as plt

x = np.array([8,16,32,65,131,262,525,1050,2100,4200])
y = np.array([0.81,0.81,0.82,0.83,0.83,0.84,0.84,0.85,0.9,0.91])

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-', label='y = log2(x)')
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xlabel('x (log scale, base 2)')
plt.ylabel('y (log scale, base 2)')
plt.title('Log-Log Plot of x vs log2(x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

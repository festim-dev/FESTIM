# Made with https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.htm

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from scipy.interpolate import griddata

E = []
n = []
error = []

data = np.genfromtxt('8e+16/cost_function_data.csv', delimiter=',')

E = data[:, 0]
n = data[:, 1]
error = data[:, 2]
error /= max(error)


# Create grid values first.
xi = np.linspace(min(E), max(E), 200)
yi = np.linspace(min(n), max(n), 200)

zi = griddata((E, n), error, (xi[None, :], yi[:, None]), method='linear')

plt.figure(1)
plt.contourf(xi, yi, zi, levels=10)
plt.colorbar(label='Rationalised error')
# plt.plot(E, n, 'ko', ms=3)
plt.plot(0.87, 1.3e-3, '*', ms=10, label="Global minimum")
plt.xlabel(r"$E_1$ (eV)")
plt.ylabel(r"$n_1$ (at.fr.)")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend()

plt.figure(2)
ax = plt.axes(projection='3d')
xv, yv = np.meshgrid(xi, yi)
ax.plot_surface(xv, yv, zi, cmap='viridis', edgecolor='none')
plt.xlabel(r"$E_1$ (eV)")
plt.ylabel(r"$n_1$ (at.fr.)")
ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax.set_zlabel(r"Rationalised error")

plt.show()

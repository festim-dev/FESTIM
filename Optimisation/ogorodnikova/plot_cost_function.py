# Made with https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.htm

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from scipy.interpolate import griddata
import matplotlib.animation as animation

E = []
n = []
error = []

data = np.genfromtxt('cost_function_pondered_average/cost_function_data_very_refined.csv', delimiter=',')
# data = np.genfromtxt('cost_function_unpondered_average/cost_function_data_very_refined.csv', delimiter=',')
# data = np.genfromtxt('fit_ogorodnikova/cost_function_data.csv', delimiter=',')
N = 2000
E = data[:N, 0]
n = data[:N, 1]
error = data[:N, 2]
# error = (error - min(error))/(max(error) - min(error))


# Create grid values first.
xi = np.linspace(min(E), max(E), 200)
yi = np.linspace(min(n), max(n), 200)

zi = griddata((E, n), error, (xi[None, :], yi[:, None]), method='cubic')
zi_min = zi.min()
zi_max = zi.max()
zi = (zi - zi_min)/(zi_max - zi_min)

plt.figure(1)
CF = plt.contourf(xi, yi, zi, levels=100)
plt.colorbar(CF, label='Rationalised error')

CS = plt.contour(xi, yi, zi, levels=10, colors="black", linewidths=0.75)
plt.clabel(CS, inline=1, fontsize=10)
# plt.plot(E, n, 'ko', ms=1, alpha=0.5)
for c in CF.collections:  # for avoiding white lines in pdf
    c.set_edgecolor("face")

min_coordinates = np.unravel_index(zi.argmin(), zi.shape)


plt.plot(xi[min_coordinates[1]], yi[min_coordinates[0]], '*', ms=10, label="Global minimum")
plt.xlabel(r"$E_1$ (eV)")
plt.ylabel(r"$n_1$ (at.fr.)")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(loc='lower left')


fig = plt.figure(2)
ax = plt.axes(projection='3d')
ax.view_init(elev=35, azim=-121)
xv, yv = np.meshgrid(xi, yi)
ax.plot_surface(xv, yv, zi, cmap='viridis', edgecolor='none')
plt.xlabel(r"$E_1$ (eV)")
plt.ylabel(r"$n_1$ (at.fr.)")
ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax.set_zlabel(r"Rationalised error")

plt.show()

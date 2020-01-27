import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from scipy.interpolate import griddata
import matplotlib.animation as animation

data = np.genfromtxt('cost_function_pondered_average/cost_function_data_very_refined.csv', delimiter=',')
E = data[:, 0]
n = data[:, 1]
error = data[:, 2]
# error = (error - min(error))/(max(error) - min(error))


# Create grid values first.
xi = np.linspace(min(E), max(E), 200)
yi = np.linspace(min(n), max(n), 200)

zi = griddata((E, n), error, (xi[None, :], yi[:, None]), method='cubic')
zi_min = zi.min()
zi_max = zi.max()
zi = (zi - zi_min)/(zi_max - zi_min)

fig = plt.figure()


ax = plt.axes(projection='3d')
xv, yv = np.meshgrid(xi, yi)
ax.plot_surface(xv, yv, zi, cmap='viridis', edgecolor='none')
plt.xlabel(r"$E_1$ (eV)")
plt.ylabel(r"$n_1$ (at.fr.)")
ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax.set_zlabel(r"Rationalised error")


T = 10
fps = 10


def init():
    # Plot the surface.
    ax.plot_surface(xv, yv, zi, cmap='viridis', edgecolor='none')
    return fig,


def animate(i):
    # azimuth angle : 0 deg to 360 deg
    ax.view_init(elev=10, azim=i*360/T/fps)
    return fig,


ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=T*fps, interval=200, blit=True)
fn = 'cost_function'
ani.save(fn+'.gif', writer='imagemagick', fps=fps)
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
# plt.show()

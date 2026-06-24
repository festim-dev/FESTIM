from mpi4py import MPI

import numpy as np
import pyvista
from dolfinx import plot
from dolfinx.mesh import create_unit_square

import festim as F

my_model = F.HydrogenTransportProblemDiscontinuous()

D = 0.1

mat1 = F.Material(D_0=D, E_D=0, K_S_0=1, E_K_S=0)
mat2 = F.Material(D_0=D, E_D=0, K_S_0=2, E_K_S=0)

vol1 = F.VolumeSubdomain1D(id=1, material=mat1, borders=[0, 0.5])
vol2 = F.VolumeSubdomain1D(id=2, material=mat2, borders=[0.5, 1])

left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))

my_model.subdomains = [vol1, vol2, left, right]

# dolfinx_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
# my_model.mesh = F.Mesh(dolfinx_mesh)

my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 101))


A = F.Species("A", subdomains=[vol1, vol2])

my_model.species = [A]


my_model.interfaces = [
    F.InterfaceFlux(id=1, subdomains=[vol1, vol2], k_plus=2, k_minus=1),
    # F.Interface(
    #     id=2,
    #     subdomains=[vol1, vol2],
    #     penalty_term=50,
    #     method=F.InterfaceMethod.nitsche,
    # ),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(species=A, subdomain=left, value=1),
    F.FixedConcentrationBC(species=A, subdomain=right, value=0),
]

my_model.temperature = 300

my_model.settings = F.Settings(final_time=3, atol=1e-9, rtol=1e-9, stepsize=0.1)
# my_model.settings = F.Settings(transient=False, atol=1e-10, rtol=1e-10)

my_model.exports = [
    F.Profile1DExport(field=A, subdomain=vol1),
    F.Profile1DExport(field=A, subdomain=vol2),
]

from dolfinx.log import set_log_level, LogLevel

set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()


import matplotlib.pyplot as plt

for e in my_model.exports:
    for idx, data in enumerate(e.data):
        plt.plot(
            e.x,
            e.data[idx],
            label=f"Subdomain {e.subdomain.id}",
        )
plt.show()

# def make_ugrid(solution):
#     topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
#     u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#     u_grid.point_data["c"] = solution.x.array.real
#     u_grid.set_active_scalars("c")
#     return u_grid


# u_plotter = pyvista.Plotter()

# # fun = A.post_processing_solution
# fun = A.subdomain_to_post_processing_solution[vol1]
# u_grid_left = make_ugrid(fun)
# u_plotter.add_mesh(u_grid_left, show_edges=True)

# fun = A.subdomain_to_post_processing_solution[vol2]
# u_grid_right = make_ugrid(fun)
# u_plotter.add_mesh(u_grid_right, show_edges=True)
# u_plotter.view_xy()
# u_plotter.add_text("Hydrogen concentration in multi-material problem", font_size=12)

# if not pyvista.OFF_SCREEN:
#     u_plotter.show()
# else:
#     figure = u_plotter.screenshot("concentration.png")

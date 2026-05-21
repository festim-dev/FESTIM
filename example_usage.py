from mpi4py import MPI

import numpy as np
from dolfinx.mesh import create_unit_square
from dolfinx import plot
import festim as F
import pyvista

my_model = F.HydrogenTransportProblemDiscontinuous()

mat = F.Material(D_0=0.1, E_D=0, K_S_0=1, E_K_S=0)

vol1 = F.VolumeSubdomain(id=1, material=mat, locator=lambda x: x[0] < 0.6)
vol2 = F.VolumeSubdomain(id=2, material=mat, locator=lambda x: x[0] >= 0.5)

left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))

my_model.subdomains = [vol1, vol2, left, right]

dolfinx_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
my_model.mesh = F.Mesh(dolfinx_mesh)


A = F.Species("A", subdomains=[vol1, vol2])

my_model.species = [A]


my_model.interfaces = [
    F.InterfaceFlux(id=1, subdomains=[vol1, vol2], k_plus=1, k_minus=1)
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(species=A, subdomain=left, value=1),
    F.FixedConcentrationBC(species=A, subdomain=right, value=0),
]

my_model.temperature = 300

# my_model.settings = F.Settings(final_time=10, atol=1e-9, rtol=1e-9, stepsize=1)
my_model.settings = F.Settings(transient=False, atol=1e-9, rtol=1e-9)

my_model.initialise()
my_model.run()


def make_ugrid(solution):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid


u_plotter = pyvista.Plotter()

# fun = A.post_processing_solution
fun = A.subdomain_to_post_processing_solution[vol1]
u_grid_left = make_ugrid(fun)
u_plotter.add_mesh(u_grid_left, show_edges=True)

fun = A.subdomain_to_post_processing_solution[vol2]
u_grid_right = make_ugrid(fun)
u_plotter.add_mesh(u_grid_right, show_edges=True)
u_plotter.view_xy()
u_plotter.add_text("Hydrogen concentration in multi-material problem", font_size=12)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")

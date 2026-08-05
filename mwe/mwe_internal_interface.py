"""
Modified from https://gist.github.com/RemDelaporteMathurin/d1a678b6b7439e339c8471e97cd31a39
Demonstrates a FESTIM codimension-1 internal interface with diffusion and coupling.
"""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np

import festim as F

# Parameters
L = 4.0
x_int = L / 2
D_left = 2.0
D_right = 1.5
D_int = 1.0
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0
c_int_max = 1.0

# Transient settings
dt = 1e-2
T = 10.0

mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, 1.0])],
    [20, 10],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)

eps = 1e-12
left = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_left, E_D=0.0),
    locator=lambda x: x[0] <= x_int + eps,
    name="left",
)
right = F.VolumeSubdomain(
    id=2,
    material=F.Material(D_0=D_right, E_D=0.0),
    locator=lambda x: x[0] >= x_int - eps,
    name="right",
)
gamma = F.VolumeSubdomain(
    id=3,
    material=F.Material(D_0=D_int, E_D=0.0),
    dim=mesh.topology.dim - 1,
    locator=lambda x: np.isclose(x[0], x_int),
    name="interface",
)

left_boundary = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 0.0))
right_boundary = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], L))

H_left = F.Species("c_left", subdomains=[left])
H_right = F.Species("c_right", subdomains=[right])
H_int = F.Species("c_int", subdomains=[gamma])

# Flux from the external left boundary into the left bulk.
left_boundary_flux = F.ParticleFluxBC(
    subdomain=left_boundary,
    species=H_left,
    value=0.5,
)

# Interface coupling: bulk fluxes into the codim-1 interface, plus matching sources
# in the interface equation.
J_left = lambda c_int, c_left: k1 * c_left * (1.0 - c_int / c_int_max) - k2 * c_int
J_right = lambda c_int, c_right: k3 * c_right * (1.0 - c_int / c_int_max) - k4 * c_int

interface_sources = [
    F.ParticleSource(
        value=J_left,
        species=H_int,
        volume=gamma,
        species_dependent_value={"c_int": H_int, "c_left": H_left},
    ),
    F.ParticleSource(
        value=J_right,
        species=H_int,
        volume=gamma,
        species_dependent_value={"c_int": H_int, "c_right": H_right},
    ),
]

interface_fluxes = [
    F.ParticleFluxBC(
        subdomain=gamma,
        species=H_left,
        value=lambda c_int, c_left: -J_left(c_int, c_left),
        species_dependent_value={"c_int": H_int, "c_left": H_left},
    ),
    F.ParticleFluxBC(
        subdomain=gamma,
        species=H_right,
        value=lambda c_int, c_right: -J_right(c_int, c_right),
        species_dependent_value={"c_int": H_int, "c_right": H_right},
    ),
]

right_dirichlet = F.FixedConcentrationBC(
    subdomain=right_boundary,
    value=0.0,
    species=H_right,
)

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    species=[H_left, H_right, H_int],
    subdomains=[left, right, gamma, left_boundary, right_boundary],
    sources=[*interface_sources],
    boundary_conditions=[left_boundary_flux, *interface_fluxes, right_dirichlet],
    temperature=500,
    settings=F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=True,
        final_time=T,
        stepsize=dt,
    ),
    exports=[
        F.VTXSpeciesExport(filename="results/c_left.bp", field=H_left, subdomain=left),
        F.VTXSpeciesExport(
            filename="results/c_right.bp", field=H_right, subdomain=right
        ),
        F.VTXSpeciesExport(filename="results/c_int.bp", field=H_int, subdomain=gamma),
    ],
)

model.initialise()
model.run()

c_left = H_left.subdomain_to_post_processing_solution[left].x.array
c_right = H_right.subdomain_to_post_processing_solution[right].x.array
c_int = H_int.subdomain_to_post_processing_solution[gamma].x.array

print("c_left range", c_left.min(), c_left.max())
print("c_right range", c_right.min(), c_right.max())
print("c_int range", c_int.min(), c_int.max())

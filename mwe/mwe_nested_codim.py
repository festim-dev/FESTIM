"""Nested codimensional coupling: bulk -> Gamma -> Lambda.

The 3D analogue of ``mwe_internal_interfaces_traps.py``, with one more level of nesting:

    Omega_left, Omega_right   codim 0   the two halves of the box, x < 2 and x > 2
    Gamma                     codim 1   the plane x = 2, a mobile species of its own,
                                        exchanging with both bulks
    Lambda                    codim 2   the line x = 2, y = 0.5 inside Gamma, a mobile
                                        species of its own, exchanging with Gamma

Every exchange crosses exactly one codimension, so each one is an ordinary facet
integral on the mesh of the subdomain being exchanged with: bulk <-> Gamma on the parent
mesh, Gamma <-> Lambda on Gamma's submesh. Nothing ever needs the trace of a field on a
set of codimension 2 of its own mesh, which is the thing that does not exist.

Declaring a nested subdomain takes two extra arguments compared with a manifold:

    dim=mesh_dim - 2      as for a manifold, the topological dimension
    parent=<manifold>     the codim-1 subdomain it lies in. Its ``locator`` is applied
                          to *that subdomain's submesh*, not to the parent mesh.

Lambda's own ends are one codimension further down again: a point in a 3D mesh, ie. a
``SurfaceSubdomain`` with ``dim=mesh_dim - 3``. The branch already locates the boundary
of a manifold on that manifold's submesh (commit ed5cb0e), and the same code works one
level down without change -- the entities wanted are always the *facets* of the mesh
the bounded subdomain lives on. A zero Dirichlet condition is applied at the bottom end
of Lambda (z = 0) below, which is what makes its profile differ visibly from Gamma's
instead of relaxing to nearly the same value.
"""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np

import festim as F

# Parameters
L = 4.0
x_int = L / 2
y_int = 0.5
D_left, D_right, D_int, D_line = 2.0, 1.5, 1.0, 0.8

# bulk <-> Gamma
k1 = k2 = k3 = k4 = 1.0
c_int_max = 1.0
# Gamma <-> Lambda
k5, k6 = 5.0, 1.0
c_line_max = 1.0

dt = 0.1
T_final = 10.0

mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, 1.0, 1.0])],
    [16, 8, 8],
    cell_type=dolfinx.mesh.CellType.hexahedron,
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
# NEW: codim 2. The locator runs on gamma's submesh, so it selects a line of that
# surface -- there is no need (and no way) to find it in the parent mesh's tags.
lam = F.VolumeSubdomain(
    id=4,
    material=F.Material(D_0=D_line, E_D=0.0),
    dim=mesh.topology.dim - 2,
    parent=gamma,
    locator=lambda x: np.isclose(x[1], y_int),
    name="line",
)

left_boundary = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 0.0))
right_boundary = F.SurfaceSubdomain(id=6, locator=lambda x: np.isclose(x[0], L))
# NEW: codim 3, the bottom end of Lambda. Gamma is the plane x = 2 and Lambda the line
# y = 0.5 inside it, so Lambda runs along z and its ends are at z = 0 and z = 1. The
# locator is applied to *Lambda's* submesh, and which subdomain the surface bounds is
# read off the species of the boundary condition using it -- so this one object could
# be reused on another nested subdomain with a different species.
line_bottom = F.SurfaceSubdomain(
    id=7,
    dim=mesh.topology.dim - 3,
    locator=lambda x: np.isclose(x[2], 0.0),
)

H_left = F.Species("c_left", subdomains=[left])
H_right = F.Species("c_right", subdomains=[right])
H_int = F.Species("c_int", subdomains=[gamma])
H_line = F.Species("c_line", subdomains=[lam])

left_boundary_flux = F.ParticleFluxBC(
    subdomain=left_boundary, species=H_left, value=0.5
)
right_dirichlet = F.FixedConcentrationBC(
    subdomain=right_boundary, value=0.0, species=H_right
)
# a sink at one end of the line: Lambda now has somewhere to drain to, so it develops a
# gradient along z instead of equilibrating with Gamma almost everywhere
line_dirichlet = F.FixedConcentrationBC(
    subdomain=line_bottom, value=0.0, species=H_line
)


# --- bulk <-> Gamma (unchanged, codim-1 coupling on the parent mesh) ----------
def J_left(c_int, c_left):
    return k1 * c_left * (1.0 - c_int / c_int_max) - k2 * c_int


def J_right(c_int, c_right):
    return k3 * c_right * (1.0 - c_int / c_int_max) - k4 * c_int


# --- Gamma <-> Lambda (new, codim-2 coupling on Gamma's submesh) -------------
def J_line(c_line, c_int):
    """The rate at which Lambda gains particles from Gamma, per unit length."""
    return k5 * c_int * (1.0 - c_line / c_line_max) - k6 * c_line


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
    # Lambda gains what Gamma loses
    F.ParticleSource(
        value=J_line,
        species=H_line,
        volume=lam,
        species_dependent_value={"c_line": H_line, "c_int": H_int},
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
    # ... and Gamma loses it. A codim-2 subdomain is used wherever a surface is
    # expected, exactly as a codim-1 one already is
    F.ParticleFluxBC(
        subdomain=lam,
        species=H_int,
        value=lambda c_line, c_int: -J_line(c_line, c_int),
        species_dependent_value={"c_line": H_line, "c_int": H_int},
    ),
]

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    species=[H_left, H_right, H_int, H_line],
    subdomains=[
        left,
        right,
        gamma,
        lam,
        left_boundary,
        right_boundary,
        line_bottom,
    ],
    # note: a flat list. `reactions=[decay]` where `decay` is itself a list is what
    # produced the 'list' object has no attribute 'volume' error
    reactions=[
        F.Reaction(reactant=[H_int], k_0=1.0, E_k=0.0, volume=gamma),
        F.Reaction(reactant=[H_line], k_0=0.5, E_k=0.0, volume=lam),
    ],
    sources=interface_sources,
    boundary_conditions=[
        left_boundary_flux,
        *interface_fluxes,
        right_dirichlet,
        line_dirichlet,
    ],
    temperature=500,
    settings=F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=True,
        final_time=T_final,
        stepsize=dt,
    ),
    exports=[
        F.VTXSpeciesExport(filename="results/c_left.bp", field=H_left, subdomain=left),
        F.VTXSpeciesExport(
            filename="results/c_right.bp", field=H_right, subdomain=right
        ),
        F.VTXSpeciesExport(filename="results/c_int.bp", field=H_int, subdomain=gamma),
        F.VTXSpeciesExport(filename="results/c_line.bp", field=H_line, subdomain=lam),
    ],
)

model.initialise()
model.run()

for name, species, subdomain in [
    ("c_left", H_left, left),
    ("c_right", H_right, right),
    ("c_int", H_int, gamma),
    ("c_line", H_line, lam),
]:
    values = species.subdomain_to_post_processing_solution[subdomain].x.array
    print(f"{name} range", values.min(), values.max())

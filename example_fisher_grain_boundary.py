"""Fisher grain-boundary diffusion, as a codimension-1 subdomain.

A short-circuit diffusion problem: a specimen is held at a constant surface
concentration, and a grain boundary perpendicular to that surface carries hydrogen far
deeper than lattice diffusion alone would, leaking sideways into the grain as it goes.
This is the classical Fisher (1951) model; Whipple and Le Claire solved it analytically,
and the script checks the simulation against that solution at the end.

Geometry -- the problem is symmetric about the grain boundary plane, so only half of it
is modelled. ``x`` runs across the grain, ``y`` runs into the depth::

      y=0   ┌───────────────┐   surface, c = C0
            │Γ              │
            │Γ    grain     │   Γ = the grain boundary, a codim-1 subdomain at x=0
            │Γ    (D_B)     │       carrying its own equation with D_GB >> D_B
            │Γ              │
      y=LY  └───────────────┘
           x=0            x=LX  <- mid-grain symmetry plane, zero flux

Requires the codim-1 machinery *and* Dirichlet conditions on the boundary of a manifold
(the ``dim`` argument of ``SurfaceSubdomain``), which is what pins the grain boundary to
the surface concentration at its mouth.

UNITS -- the one thing to get right
-----------------------------------
The grain boundary is physically a slab of width ``DELTA`` that has been collapsed onto
a surface. ``c_gb`` is the concentration *inside* that slab (H/m3, the same unit as the
bulk), so the two halves of the exchange are not the same number:

* the bulk loses a **flux** ``J`` (H/m2/s) through the grain boundary plane;
* the slab gains a **volumetric rate** ``2 J / DELTA`` (H/m3/s).

``1 / DELTA`` converts a per-area flux into a per-volume rate inside the slab, and the
factor ``2`` is because the full slab collects from both of its faces. It comes out the
same in this symmetric half-cell: the half-slab of width ``DELTA / 2`` collects from one
face only, ``J / (DELTA / 2)``. Matching some other convention -- Fisher's own source
term is ``2 D_B dc/dx``, a flux per face -- is therefore a matter of changing this one
coefficient, not of restructuring the coupling. What must stay paired is that both
halves are written from the *same* ``J``: that is what makes the exchange conservative.
"""

from mpi4py import MPI

import dolfinx
import numpy as np

import festim as F

# ---------------------------------------------------------------- parameters
D_B = 1e-4  # lattice (bulk) diffusivity
D_GB = 1.0  # grain-boundary diffusivity, 1e4 x faster
DELTA = 1e-3  # grain-boundary width
K_EX = 1.0  # bulk <-> grain-boundary exchange coefficient (see below)
C0 = 1.0  # surface concentration

LX, LY = 0.1, 1.5  # half grain width, depth
NX, NY = 100, 150
T_END, DT = 1.0, 0.02

# Fisher's model assumes *local equilibrium* between the grain boundary and the lattice
# in contact with it, rather than a finite exchange rate. FESTIM's coupling is kinetic,
# so equilibrium is approached by making K_EX large compared with the bulk transport it
# competes with, sqrt(D_B / T_END). The script reports how well that holds.

# ---------------------------------------------------------------- model
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([LX, LY])], [NX, NY]
)

grain = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_B, E_D=0.0),
    locator=lambda x: np.full_like(x[0], True, dtype=bool),
)
# the grain boundary: a line inside a 2D mesh, with its own transport equation
gb = F.VolumeSubdomain(
    id=2,
    material=F.Material(D_0=D_GB, E_D=0.0),
    dim=1,
    locator=lambda x: np.isclose(x[0], 0.0),
)
surface = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], 0.0))
# where the grain boundary meets the surface: the boundary of a codim-1 subdomain, so
# dim = mesh dimension - 2. Its locator is evaluated on the grain boundary itself.
gb_mouth = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[1], 0.0))

c_b = F.Species("c_b", subdomains=[grain])
c_gb = F.Species("c_gb", subdomains=[gb])

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    species=[c_b, c_gb],
    subdomains=[grain, gb, surface, gb_mouth],
    sources=[
        # ... and the slab gains it, as a volumetric rate (see UNITS above)
        F.ParticleSource(
            value=lambda cb, cg: (2.0 / DELTA) * K_EX * (cb - cg),
            species=c_gb,
            volume=gb,
            species_dependent_value={"cb": c_b, "cg": c_gb},
        )
    ],
    boundary_conditions=[
        # the bulk loses the flux J = K_EX (c_b - c_gb) through the grain boundary.
        # FESTIM's flux convention is the influx, hence the reversed sign
        F.ParticleFluxBC(
            subdomain=gb,
            species=c_b,
            value=lambda cb, cg: K_EX * (cg - cb),
            species_dependent_value={"cb": c_b, "cg": c_gb},
        ),
        F.FixedConcentrationBC(subdomain=surface, value=C0, species=c_b),
        F.FixedConcentrationBC(subdomain=gb_mouth, value=C0, species=c_gb),
    ],
    temperature=500,
    settings=F.Settings(
        atol=1e-14,
        rtol=1e-12,
        transient=True,
        final_time=T_END,
        stepsize=F.Stepsize(initial_value=DT),
    ),
    exports=[
        F.VTXSpeciesExport("fisher_grain.bp", field=c_b, subdomain=grain),
        F.VTXSpeciesExport("fisher_gb.bp", field=c_gb, subdomain=gb),
    ],
)
model.initialise()
model.run()


# ---------------------------------------------------------------- analysis
def eval_on(fn, points):
    """Evaluate a fenics Function at a list of (x, y, 0) points."""
    fn_mesh = fn.function_space.mesh
    tree = dolfinx.geometry.bb_tree(fn_mesh, fn_mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(fn_mesh, candidates, points)
    cells = [colliding.links(i)[0] for i in range(len(points))]
    return fn.eval(points, cells).reshape(-1)


cg_fn = c_gb.subdomain_to_post_processing_solution[gb]
cb_fn = c_b.subdomain_to_post_processing_solution[grain]
order = np.argsort(cg_fn.function_space.tabulate_dof_coordinates()[:, 1])
y_gb = cg_fn.function_space.tabulate_dof_coordinates()[order, 1]
c_gb_vals = cg_fn.x.array[order]

bulk_only = 2 * np.sqrt(D_B * T_END)
print(f"lattice diffusion alone would reach ~{bulk_only:.3g}")
print("grain boundary profile:")
for y in (0.0, 0.25, 0.5, 0.75, 1.0):
    print(f"   y = {y:.2f}   c_gb = {np.interp(y, y_gb, c_gb_vals):.4e}")

# how close the kinetic exchange gets to Fisher's local-equilibrium assumption
probe = np.linspace(0.05, 1.0, 20)
pts = np.column_stack([np.zeros_like(probe), probe, np.zeros_like(probe)])
ratio = eval_on(cb_fn, pts) / np.interp(probe, y_gb, c_gb_vals)
print(f"\nlocal equilibrium c_b(0,y)/c_gb(y): {ratio.min():.4f} .. {ratio.max():.4f}")

# Whipple / Le Claire: in type-B kinetics the section-averaged concentration obeys
#     ln(cbar) linear in y**(6/5),  and  delta*D_gb = 1.322 sqrt(D_B/t) (-slope)**(-5/3)
depths = np.linspace(0.15, 1.0, 25)
xs = np.linspace(0.0, LX, 200)
cbar = np.array(
    [
        (
            np.trapezoid(
                eval_on(
                    cb_fn,
                    np.column_stack([xs, np.full_like(xs, y), np.zeros_like(xs)]),
                ),
                xs,
            )
            + (DELTA / 2) * np.interp(y, y_gb, c_gb_vals)
        )
        / (LX + DELTA / 2)
        for y in depths
    ]
)

slope, intercept = np.polyfit(depths**1.2, np.log(cbar), 1)
residual = np.abs(np.log(cbar) - (slope * depths**1.2 + intercept)).max()
recovered = 1.322 * np.sqrt(D_B / T_END) * (-slope) ** (-5.0 / 3.0)

alpha = DELTA / (2 * np.sqrt(D_B * T_END))
beta = DELTA * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * T_END))
print(
    f"\nLe Claire analysis (valid for alpha << 1, beta >> 1: "
    f"alpha = {alpha:.3f}, beta = {beta:.0f})"
)
print(f"   d ln(cbar) / d y**(6/5) = {slope:.4f}  (max fit residual {residual:.3f})")
print(f"   recovered delta*D_gb    = {recovered:.4e}")
print(f"   input     delta*D_gb    = {DELTA * D_GB:.4e}")
print(f"   error                   = {100 * abs(recovered / (DELTA * D_GB) - 1):.1f} %")

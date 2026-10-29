"""Strong boundary conditions on the boundary of a manifold subdomain.

A manifold (``VolumeSubdomain(dim=mesh_dim - 1)``) has a boundary of its own -- the
endpoints of a line in a 2D mesh, the rim of a surface in a 3D mesh. That boundary is
codim-1 *relative to the manifold*, so it is located directly on the manifold's submesh
and needs no codim-2 entity of the parent mesh and no meshtag. It is declared as a
``SurfaceSubdomain`` with ``dim = mesh_dim - 2``.

The motivating case is a pipe: the fluid bulk is a 1D advection-diffusion domain running
along a 2D wall, exchanging with it through a mass transfer coefficient, and the inlet
concentration is a Dirichlet condition at one end of the 1D domain.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

WALL_ID, FLUID_ID, INLET_ID, OUTLET_ID, OUTER_ID = 1, 2, 3, 4, 5


def velocity(mesh, v_x):
    vel = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,))))
    vel.interpolate(
        lambda x: np.vstack([np.full(x.shape[1], v_x), np.zeros(x.shape[1])])
    )
    return vel


def advection_diffusion_along_gamma(n, length=1.0, height=1.0, D=1.0, v_x=1.0):
    """Pure advection-diffusion along Γ with a Dirichlet condition at *both* ends.

    Γ is uncoupled from the bulk here, so it is an ordinary 1D two-point boundary value
    problem with a closed-form solution -- which is what makes it a sharp test of the
    endpoint conditions.
    """
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([length, height])],
        [n, max(2, n // 4)],
    )
    wall = F.VolumeSubdomain(
        id=WALL_ID,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gamma = F.VolumeSubdomain(
        id=FLUID_ID,
        material=F.Material(D_0=D, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[1], height),
    )
    # the two ends of Γ: codim 2, ie. dim = mesh_dim - 2 = 0
    inlet = F.SurfaceSubdomain(
        id=INLET_ID, dim=0, locator=lambda x: np.isclose(x[0], 0)
    )
    outlet = F.SurfaceSubdomain(
        id=OUTLET_ID, dim=0, locator=lambda x: np.isclose(x[0], length)
    )
    outer = F.SurfaceSubdomain(id=OUTER_ID, locator=lambda x: np.isclose(x[1], 0.0))

    c_wall = F.Species("c_wall", subdomains=[wall])
    c_gamma = F.Species("c_gamma", subdomains=[gamma])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_wall, c_gamma],
        subdomains=[wall, gamma, inlet, outlet, outer],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=c_gamma),
            F.FixedConcentrationBC(subdomain=outlet, value=0.0, species=c_gamma),
            F.FixedConcentrationBC(subdomain=outer, value=0.0, species=c_wall),
        ],
        drift_terms=[
            F.AdvectionTerm(
                velocity=velocity(mesh, v_x), subdomain=gamma, species=c_gamma
            )
        ],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()

    c = c_gamma.subdomain_to_post_processing_solution[gamma]
    x = c.function_space.tabulate_dof_coordinates()[:, 0]
    # -D c'' + v c' = 0, c(0) = 1, c(length) = 0
    peclet = v_x * length / D
    exact = (np.exp(peclet) - np.exp(v_x * x / D)) / (np.exp(peclet) - 1)
    return np.max(np.abs(c.x.array - exact))


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_dirichlet_on_manifold_ends_matches_analytical():
    """Both endpoint values are attained and the profile between them converges."""
    refinements = [10, 20, 40]
    errors = [advection_diffusion_along_gamma(n) for n in refinements]
    rates = [
        np.log(e0 / e1) / np.log(n1 / n0)
        for e0, e1, n0, n1 in zip(
            errors, errors[1:], refinements, refinements[1:], strict=False
        )
    ]
    assert all(r > 1.9 for r in rates), (rates, errors)
    assert errors[-1] < 1e-5, errors


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only for now")
def test_pipe_inlet_concentration():
    """The motivating case: a 1D fluid on a 2D pipe wall, fed at the inlet.

    ``J = k (c_inf - c_wall)`` leaves the fluid and enters the wall, so the fluid
    concentration must decay downstream of the inlet, and everything the wall takes up
    must leave through its outer boundary.
    """
    length, height = 4.0, 1.0
    d_wall, d_fluid, k, v_x, c_in = 0.05, 1.0, 0.8, 2.0, 1.0

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([length, height])],
        [80, 20],
    )
    wall = F.VolumeSubdomain(
        id=WALL_ID,
        material=F.Material(D_0=d_wall, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    fluid = F.VolumeSubdomain(
        id=FLUID_ID,
        material=F.Material(D_0=d_fluid, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[1], height),
    )
    inlet = F.SurfaceSubdomain(
        id=INLET_ID, dim=0, locator=lambda x: np.isclose(x[0], 0)
    )
    outlet = F.SurfaceSubdomain(
        id=OUTLET_ID, dim=0, locator=lambda x: np.isclose(x[0], length)
    )
    outer = F.SurfaceSubdomain(id=OUTER_ID, locator=lambda x: np.isclose(x[1], 0.0))

    c_wall = F.Species("c_wall", subdomains=[wall])
    c_inf = F.Species("c_inf", subdomains=[fluid])

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_wall, c_inf],
        subdomains=[wall, fluid, inlet, outlet, outer],
        sources=[
            F.ParticleSource(
                value=lambda c_f, c_w: -k * (c_f - c_w),
                species=c_inf,
                volume=fluid,
                species_dependent_value={"c_f": c_inf, "c_w": c_wall},
            ),
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=fluid,
                species=c_wall,
                value=lambda c_f, c_w: k * (c_f - c_w),
                species_dependent_value={"c_f": c_inf, "c_w": c_wall},
            ),
            F.FixedConcentrationBC(subdomain=inlet, value=c_in, species=c_inf),
            F.FixedConcentrationBC(subdomain=outer, value=0.0, species=c_wall),
            # the fluid leaves at the far end. Without this the divergence form makes
            # the outlet a closed end, and the fluid backs up instead of draining
            F.OutflowBC(subdomain=outlet, species=c_inf),
        ],
        drift_terms=[
            F.AdvectionTerm(
                velocity=velocity(mesh, v_x), subdomain=fluid, species=c_inf
            )
        ],
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()

    c = c_inf.subdomain_to_post_processing_solution[fluid]
    order = np.argsort(c.function_space.tabulate_dof_coordinates()[:, 0])
    profile = c.x.array[order]

    assert np.isclose(profile[0], c_in, atol=1e-12), (
        "the inlet value is imposed exactly"
    )
    assert np.all(np.diff(profile) < 0), "the fluid gives up tritium to the wall"

    # what the wall takes up along Γ leaves through its outer boundary
    def assemble(form):
        return mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    form, entity_maps=[sd.cell_map for sd in model.volume_subdomains]
                )
            ),
            op=MPI.SUM,
        )

    into_wall = assemble(
        k
        * (c_inf.subdomain_to_solution[fluid] - c_wall.subdomain_to_solution[wall])
        * model.ds(FLUID_ID)
    )
    n = ufl.FacetNormal(mesh)
    out_of_wall = -assemble(
        d_wall
        * ufl.dot(ufl.grad(c_wall.subdomain_to_solution[wall]), n)
        * model.ds(OUTER_ID)
    )
    # the outflux is recovered by differentiating the solution on a Dirichlet boundary,
    # which is only first order: the gap halves under refinement (8.2e-4, 4.1e-4, 2.1e-4
    # at 40x10, 80x20, 160x40). The point of the check is a wiring or sign error, which
    # would show up as a discrepancy of order one, not of order h
    assert np.isclose(into_wall, out_of_wall, rtol=1e-3), (into_wall, out_of_wall)

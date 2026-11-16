import numpy as np
import pytest
from dolfinx import fem

import festim as F

from .tools import error_L2


@pytest.mark.parametrize(
    "porosity, tortuosity",
    [
        (1.0, 1.0),  # No porous corrections
        (0.4, 1.0),  # Porosity correction only
        (1.0, 2.0),  # Tortuosity correction only
        (0.4, 2.0),  # Combined corrections
    ],
)
def test_porous_transient_mms_with_advection(porosity, tortuosity):
    """Verify transient transport in a porous material using MMS.

    Governing equation:
        porosity * dc/dt
        - div(porosity/tortuosity * D * grad(c))
        + div(u_D * c) = S

    Here, porosity is the void-volume fraction, tortuosity
    is the diffusion-path correction factor, and u_D is the
    superficial (Darcy) velocity.

    The manufactured solution is quadratic in space and linear
    in time.

    The test checks second-order spatial convergence with P1
    elements and is sensitive to incorrect porosity scaling
    of the transient and diffusion terms.
    """

    L, t0, tf, c0 = 1.0, 1.0, 0.4, 1.0
    D, u_D = 0.1, 0.2

    # Manufactured exact solution
    def exact(x, t):
        xi = x[0] / L
        return c0 * (1 + t / t0 + xi + xi**2)

    # Independently derived source term
    def manufactured_source(x):
        xi = x[0] / L
        return c0 * (
            porosity / t0
            - 2 * porosity * D / (tortuosity * L**2)
            + u_D / L * (1 + 2 * xi)
        )

    errors = []

    for n in (20, 40):
        mesh = F.Mesh1D(np.linspace(0, L, n + 1))

        material = F.PorousMaterial(
            D_0=D,
            E_D=0.0,
            porosity=porosity,
            tortuosity=tortuosity,
        )

        volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
        left = F.SurfaceSubdomain1D(id=1, x=0.0)
        right = F.SurfaceSubdomain1D(id=2, x=L)
        species = F.Species("H")

        # Constant Darcy velocity
        V_velocity = fem.functionspace(mesh.mesh, ("Lagrange", 1, (1,)))
        velocity = fem.Function(V_velocity)
        velocity.interpolate(lambda x: np.full((1, x.shape[1]), u_D))

        model = F.HydrogenTransportProblem(
            mesh=mesh,
            subdomains=[volume, left, right],
            species=[species],
            temperature=500.0,
            initial_conditions=[
                F.InitialConcentration(
                    value=lambda x: exact(x, 0.0),
                    volume=volume,
                    species=species,
                )
            ],
            boundary_conditions=[
                F.FixedConcentrationBC(
                    subdomain=left,
                    value=exact,
                    species=species,
                ),
                F.FixedConcentrationBC(
                    subdomain=right,
                    value=exact,
                    species=species,
                ),
            ],
            sources=[
                F.ParticleSource(
                    value=manufactured_source,
                    volume=volume,
                    species=species,
                )
            ],
            drift_terms=[
                F.AdvectionTerm(
                    velocity=velocity,
                    subdomain=volume,
                    species=species,
                )
            ],
            settings=F.Settings(
                atol=1e-12,
                rtol=1e-12,
                transient=True,
                final_time=tf,
            ),
        )
        model.settings.stepsize = 0.1

        model.initialise()
        model.run()

        errors.append(
            error_L2(
                species.post_processing_solution,
                lambda x: exact(x, tf),
            )
        )

    # Expected second-order spatial convergence
    rate = np.log(errors[0] / errors[1]) / np.log(2)

    assert errors[-1] < 1e-3
    assert rate > 1.8, (errors, rate)


def test_porous_advection_warns_about_superficial_velocity():
    """Warn once per porous volume when advection is specified."""

    mesh = F.Mesh1D(np.linspace(0, 1, 6))

    material = F.PorousMaterial(
        D_0=1.0,
        E_D=0.0,
        porosity=0.4,
        tortuosity=2.0,
    )
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0)

    H = F.Species("H")

    V = fem.functionspace(mesh.mesh, ("Lagrange", 1, (1,)))
    velocity = fem.Function(V)
    velocity.x.array[:] = 0.2

    model = F.HydrogenTransportProblem(
        mesh=mesh,
        subdomains=[volume, left],
        species=[H],
        temperature=500.0,
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left, value=0.0, species=H)
        ],
        drift_terms=[
            F.AdvectionTerm(
                velocity=velocity,
                subdomain=volume,
                species=H,
            )
        ],
        settings=F.Settings(transient=False, atol=1e-9, rtol=1e-9),
    )

    with pytest.warns(UserWarning, match="expects superficial") as caught:
        model.initialise()

    matching_warnings = [w for w in caught if "expects superficial" in str(w.message)]

    assert len(matching_warnings) == 1


@pytest.mark.parametrize(
    "porosity, tortuosity",
    [
        (1.0, 1.0),  # No porous corrections
        (0.4, 1.0),  # Porosity correction only
        (1.0, 2.0),  # Tortuosity correction only
        (0.4, 2.0),  # Combined corrections
    ],
)
def test_porous_transient_mms_with_advection_discontinuous(porosity, tortuosity):
    """Verify transient transport in a porous material using MMS.

    Governing equation:
        porosity * dc/dt
        - div(porosity/tortuosity * D * grad(c))
        + div(u_D * c) = S

    Here, porosity is the void-volume fraction, tortuosity
    is the diffusion-path correction factor, and u_D is the
    superficial (Darcy) velocity.

    The manufactured solution is quadratic in space and linear
    in time.

    The test checks second-order spatial convergence with P1
    elements and is sensitive to incorrect porosity scaling
    of the transient and diffusion terms.
    """

    L, t0, tf, c0 = 1.0, 1.0, 0.4, 1.0
    D, u_D = 0.1, 0.2

    # Manufactured exact solution
    def exact(x, t):
        xi = x[0] / L
        return c0 * (1 + t / t0 + xi + xi**2)

    # Independently derived source term
    def manufactured_source(x):
        xi = x[0] / L
        return c0 * (
            porosity / t0
            - 2 * porosity * D / (tortuosity * L**2)
            + u_D / L * (1 + 2 * xi)
        )

    errors = []

    for n in (20, 40):
        mesh = F.Mesh1D(np.linspace(0, L, n + 1))

        material = F.PorousMaterial(
            D_0=D,
            E_D=0.0,
            porosity=porosity,
            tortuosity=tortuosity,
        )

        volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
        left = F.SurfaceSubdomain1D(id=1, x=0.0)
        right = F.SurfaceSubdomain1D(id=2, x=L)
        species = F.Species("H", subdomains=[volume])

        # Constant Darcy velocity
        V_velocity = fem.functionspace(mesh.mesh, ("Lagrange", 1, (1,)))
        velocity = fem.Function(V_velocity)
        velocity.interpolate(lambda x: np.full((1, x.shape[1]), u_D))

        model = F.HydrogenTransportProblemDiscontinuous(
            mesh=mesh,
            subdomains=[volume, left, right],
            species=[species],
            temperature=500.0,
            initial_conditions=[
                F.InitialConcentration(
                    value=lambda x: exact(x, 0.0),
                    volume=volume,
                    species=species,
                )
            ],
            boundary_conditions=[
                F.FixedConcentrationBC(
                    subdomain=left,
                    value=exact,
                    species=species,
                ),
                F.FixedConcentrationBC(
                    subdomain=right,
                    value=exact,
                    species=species,
                ),
            ],
            sources=[
                F.ParticleSource(
                    value=manufactured_source,
                    volume=volume,
                    species=species,
                )
            ],
            drift_terms=[
                F.AdvectionTerm(
                    velocity=velocity,
                    subdomain=volume,
                    species=species,
                )
            ],
            settings=F.Settings(
                atol=1e-12,
                rtol=1e-12,
                transient=True,
                final_time=tf,
            ),
        )
        model.settings.stepsize = 0.1

        model.initialise()
        model.run()

        errors.append(
            error_L2(
                species.subdomain_to_post_processing_solution[volume],
                lambda x: exact(x, tf),
            )
        )

    # Expected second-order spatial convergence
    rate = np.log(errors[0] / errors[1]) / np.log(2)

    assert errors[-1] < 1e-3
    assert rate > 1.8, (errors, rate)


def test_porous_advection_warns_about_superficial_velocity_discontinuous():
    """Warn once per porous volume when advection is specified."""

    mesh = F.Mesh1D(np.linspace(0, 1, 6))

    material = F.PorousMaterial(
        D_0=1.0,
        E_D=0.0,
        porosity=0.4,
        tortuosity=2.0,
    )
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0)

    H = F.Species("H", subdomains=[volume])

    V = fem.functionspace(mesh.mesh, ("Lagrange", 1, (1,)))
    velocity = fem.Function(V)
    velocity.x.array[:] = 0.2

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=mesh,
        subdomains=[volume, left],
        species=[H],
        temperature=500.0,
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left, value=0.0, species=H)
        ],
        drift_terms=[
            F.AdvectionTerm(
                velocity=velocity,
                subdomain=volume,
                species=H,
            )
        ],
        settings=F.Settings(transient=False, atol=1e-9, rtol=1e-9),
    )

    with pytest.warns(UserWarning, match="expects superficial") as caught:
        model.initialise()

    matching_warnings = [w for w in caught if "expects superficial" in str(w.message)]

    assert len(matching_warnings) == 1

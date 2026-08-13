"""Drift terms: advection, Soret and electromigration.

The physical tests here assert closed forms rather than regression baselines. Two of
them are specifically about the divergence form ``-div(c v)``:

- ``test_..._mms_with_a_non_solenoidal_velocity`` is the only test in the suite that
  distinguishes it from ``v . grad(c)``. Every pre-existing advection test uses a
  divergence-free velocity with Dirichlet data on the whole boundary, so all of them
  pass under either form and none of them would have caught a regression here.
- the equilibrium and inventory tests exercise the boundary term the divergence form
  leaves behind, which is what makes a boundary with no condition on it genuinely
  no-flux.
"""

from mpi4py import MPI

import basix
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import create_unit_square

import festim as F

from .tools import error_L2


def _velocity(mesh, expression, degree=2):
    """A velocity field interpolated into an ambient vector space."""
    element = basix.ufl.element(
        "Lagrange", mesh.topology.cell_name(), degree, shape=(mesh.geometry.dim,)
    )
    velocity = fem.Function(fem.functionspace(mesh, element))
    velocity.interpolate(expression)
    return velocity


def _profile(species):
    """The 1D solution as ``(x, values)`` sorted by x.

    Read from the function's own dof coordinates rather than the mesh vertices: the
    change-of-variable problem post-processes into DG1, which has two dofs per vertex,
    so vertex ordering would silently pick the wrong entries.
    """
    function = species.post_processing_solution
    x = function.function_space.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(x)
    return x[order], function.x.array[order]


def _soret_equilibrium_model(final_time=20.0, exports=None, n=400):
    """A slab with a temperature gradient and no boundary conditions at all.

    Nothing constrains the flux at either end, so the only thing stopping the species
    from leaving is the boundary term of the conservative drift form.
    """
    length, Q_star, T_cold, T_hot, D_0 = 1.0, 0.1, 400.0, 600.0, 1.0

    material = F.Material(D_0=D_0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    H = F.Species("H", mobile=True)

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0.0, length, n + 1)),
        subdomains=[volume, left, right],
        species=[H],
        temperature=lambda x: T_cold + (T_hot - T_cold) * x[0] / length,
        initial_conditions=[
            F.InitialConcentration(value=1.0, volume=volume, species=H)
        ],
        drift_terms=[F.SoretTerm(species=H, Q_star=Q_star, subdomain=volume)],
        exports=exports or [],
        settings=F.Settings(
            atol=1e-12, rtol=1e-12, transient=True, final_time=final_time
        ),
    )
    model.settings.stepsize = F.Stepsize(0.1)
    return model, H, volume, right, (length, Q_star, T_cold, T_hot)


def test_soret_equilibrium_matches_the_closed_form():
    r"""No-flux slab in a temperature gradient relaxes to ``c = A exp(Q*/(k_B T))``.

    Setting ``J = -D grad(c) - D Q* c / (k_B T^2) grad(T)`` to zero gives
    ``grad(c)/c = grad(Q*/(k_B T))``, so the equilibrium profile is exact and ``A`` is
    fixed by the conserved inventory. For a positive heat of transport hydrogen piles up
    at the cold end.
    """
    model, H, *_rest = _soret_equilibrium_model()
    length, Q_star, T_cold, T_hot = _rest[-1]

    model.initialise()
    model.run()

    x, computed = _profile(H)

    temperature = T_cold + (T_hot - T_cold) * x / length
    shape = np.exp(Q_star / (F.k_B * temperature))
    # the inventory is conserved, and started at 1.0 everywhere
    exact = shape * length / np.trapezoid(shape, x)

    assert np.max(np.abs(computed - exact)) / np.max(exact) < 1e-5
    # and the effect is not a rounding error: the cold end holds 2.6x the hot end
    assert np.isclose(computed[0] / computed[-1], exact[0] / exact[-1], rtol=1e-4)


def test_inventory_is_conserved_under_drift():
    """Nothing leaves a domain whose boundaries carry no flux condition.

    The drift term's boundary contribution is what makes this true -- under the
    non-conservative form the species is pushed out through the ends instead.
    """
    model, H, volume, _right, _params = _soret_equilibrium_model()
    inventory = F.TotalVolume(field=H, volume=volume)
    model.exports = [inventory]

    model.initialise()
    model.run()

    assert np.isclose(inventory.data[-1], 1.0, rtol=1e-9)
    assert np.isclose(min(inventory.data), max(inventory.data), rtol=1e-9)


def test_surface_flux_accounts_for_drift():
    """At equilibrium the total flux is zero, while the diffusive part is not.

    The sharpest available check that the export reports ``-D grad(c) . n + c v . n``
    rather than the diffusive term alone: the two contributions cancel, so a
    diffusive-only export could not come out anywhere near zero. What is left is the
    O(h) error of the P1 gradient at the boundary, so the residual is asserted to be
    both small against the diffusive part and first-order convergent -- a fixed
    tolerance would say nothing about which of the two is being reported.
    """
    residuals, diffusives = [], []
    for n in (200, 400):
        model, H, _volume, right, _params = _soret_equilibrium_model(n=n)
        flux = F.SurfaceFlux(field=H, surface=right)
        model.exports = [flux]

        model.initialise()
        model.run()

        assert flux.drift_velocity is not None

        # the diffusive part alone, on the same converged solution
        normal = ufl.FacetNormal(model.mesh.mesh)
        diffusives.append(
            fem.assemble_scalar(
                fem.form(
                    -flux.D
                    * ufl.dot(ufl.grad(H.post_processing_solution), normal)
                    * model.ds(right.id)
                )
            )
        )
        residuals.append(abs(flux.data[-1]))

    # the diffusive flux is O(0.4) and does not vanish with h
    assert all(abs(d) > 0.1 for d in diffusives)
    # the total flux is three orders of magnitude smaller, and halves with the mesh
    assert residuals[-1] < 1e-2 * abs(diffusives[-1])
    rate = np.log(residuals[0] / residuals[1]) / np.log(2)
    assert rate > 0.9, (residuals, rate)


def test_advection_mms_with_a_non_solenoidal_velocity():
    r"""MMS whose source is built from ``div(c v)``, with ``div(v) = 1``.

    ``-div(D grad(c)) + div(c v) = f`` is what FESTIM solves. The form it assembled
    before, ``-div(D grad(c)) + v . grad(c) = f``, differs by ``c div(v)``, so this
    manufactured solution is only recovered by the divergence form -- under the old one
    the error stalls around 1e-1 instead of converging. Dirichlet data on the whole
    boundary keeps the boundary term out of it, so the volume form is the only thing
    under test; ``test_inventory_is_conserved_under_drift`` covers the boundary term.
    """
    D_0 = 1.5
    errors = []
    for n in (20, 40):
        mesh = create_unit_square(MPI.COMM_WORLD, n, n)
        x = ufl.SpatialCoordinate(mesh)

        # div(v) = 1, so the divergence form and v . grad(c) disagree
        velocity = _velocity(mesh, lambda x: np.vstack([x[0], np.zeros_like(x[1])]))

        material = F.Material(D_0=D_0, E_D=0.0)
        volume = F.VolumeSubdomain(id=1, material=material)
        boundary = F.SurfaceSubdomain(id=1)
        H = F.Species("H", mobile=True)

        def exact(x):
            return 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2

        flux = -D_0 * ufl.grad(exact(x)) + exact(x) * velocity
        source = ufl.div(flux)

        model = F.HydrogenTransportProblem(
            mesh=F.Mesh(mesh),
            subdomains=[volume, boundary],
            species=[H],
            temperature=500.0,
            boundary_conditions=[
                F.FixedConcentrationBC(subdomain=boundary, value=exact, species=H)
            ],
            sources=[F.ParticleSource(value=source, volume=volume, species=H)],
            drift_terms=[
                F.AdvectionTerm(velocity=velocity, subdomain=volume, species=H)
            ],
            settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
        )

        model.initialise()
        model.run()
        errors.append(error_L2(H.post_processing_solution, exact))

    assert errors[-1] < 1e-3
    rate = np.log(errors[0] / errors[1]) / np.log(2)
    assert rate > 1.8, rate


@pytest.mark.parametrize("outflow", [False, True])
def test_outflow_bc_decides_what_an_unlabelled_boundary_means(outflow):
    r"""Wall or outlet, both against a closed form.

    1D, ``D = 1``, ``v = 2``, ``c(0) = 1``, nothing else imposed. The interior equation
    is the same either way since ``div(v) = 0``, so this isolates the boundary term::

        -c'' + 2c' = 0   =>   c = A + B e^(2x)

    Without :class:`festim.OutflowBC` the natural condition at ``x = L`` is zero
    **total** flux, ``-c'(L) + 2c(L) = 0``, giving ``A = 0`` and ``c = e^(2x)``: a
    closed end, where the drift piles the species up against the wall. With it the
    condition is zero **diffusive** flux, ``c'(L) = 0``, giving ``B = 0`` and ``c = 1``:
    the species is carried out at the rate the flow delivers it.
    """
    length, D_0, v_x, n = 1.0, 1.0, 2.0, 400

    material = F.Material(D_0=D_0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    H = F.Species("H", mobile=True)

    festim_mesh = F.Mesh1D(np.linspace(0.0, length, n + 1))
    velocity = _velocity(
        festim_mesh.mesh, lambda x: np.vstack([np.full_like(x[0], v_x)])
    )

    boundary_conditions = [F.FixedConcentrationBC(subdomain=left, value=1.0, species=H)]
    if outflow:
        boundary_conditions.append(F.OutflowBC(subdomain=right, species=H))

    model = F.HydrogenTransportProblem(
        mesh=festim_mesh,
        subdomains=[volume, left, right],
        species=[H],
        temperature=500.0,
        boundary_conditions=boundary_conditions,
        drift_terms=[F.AdvectionTerm(velocity=velocity, subdomain=volume, species=H)],
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
    )
    model.initialise()
    model.run()

    x, computed = _profile(H)
    exact = np.ones_like(x) if outflow else np.exp(v_x * x / D_0)

    assert np.max(np.abs(computed - exact)) < 1e-4, (computed[-1], exact[-1])


def test_outflow_bc_is_a_no_op_without_drift():
    """It cancels a drift boundary term, so with no drift there is nothing to cancel."""
    length, n = 1.0, 50
    material = F.Material(D_0=1.0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)

    profiles = []
    for extra in ([], None):
        H = F.Species("H", mobile=True)
        bcs = [F.FixedConcentrationBC(subdomain=left, value=1.0, species=H)]
        if extra is None:
            bcs.append(F.OutflowBC(subdomain=right, species=H))
        model = F.HydrogenTransportProblem(
            mesh=F.Mesh1D(np.linspace(0.0, length, n + 1)),
            subdomains=[volume, left, right],
            species=[H],
            temperature=500.0,
            boundary_conditions=bcs,
            settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
        )
        model.initialise()
        model.run()
        profiles.append(_profile(H)[1])

    assert np.allclose(profiles[0], profiles[1])


def test_electromigration_equilibrium_is_boltzmann():
    r"""No-flux slab in a potential gradient relaxes to ``c = A exp(-z phi / (k_B T))``.

    The Nernst-Planck equilibrium. Isothermal, so the only drift is electromigration.
    """
    length, charge, phi_0, temperature, n = 1.0, 2, 0.05, 500.0, 400

    material = F.Material(D_0=1.0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    H = F.Species("H", mobile=True)

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0.0, length, n + 1)),
        subdomains=[volume, left, right],
        species=[H],
        temperature=temperature,
        initial_conditions=[
            F.InitialConcentration(value=1.0, volume=volume, species=H)
        ],
        drift_terms=[
            F.ElectromigrationTerm(
                species=H,
                charge=charge,
                potential=lambda x: phi_0 * (1.0 - x[0] / length),
                subdomain=volume,
            )
        ],
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=True, final_time=20.0),
    )
    model.settings.stepsize = F.Stepsize(0.1)

    model.initialise()
    model.run()

    x, computed = _profile(H)

    potential = phi_0 * (1.0 - x / length)
    shape = np.exp(-charge * potential / (F.k_B * temperature))
    exact = shape * length / np.trapezoid(shape, x)

    assert np.max(np.abs(computed - exact)) / np.max(exact) < 1e-5
    # positive charge accumulates where the potential is low, i.e. at x = L
    assert computed[-1] > computed[0]


@pytest.mark.parametrize(
    "coordinate_system, power", [("cylindrical", 1), ("spherical", 2)]
)
def test_conservative_drift_mms_in_curvilinear_coordinates(coordinate_system, power):
    r"""The conservative form needs the metric factor; the other one does not.

    FESTIM multiplies the equation by the metric factor ``m`` (``r`` cylindrical,
    ``r^2`` spherical) and uses ``w / m`` as the test function, so the divergence form
    picks up ``-m c v . grad(w/m)`` while ``v . grad(c)`` is metric-free -- the factors
    cancel there. Manufacturing against ``div(m J) / m`` is what pins that down.
    """
    D_0 = 1.2
    inner_radius, outer_radius = 1.0, 2.0

    errors = []
    for n in (40, 80):
        # away from the axis, where the metric weight is singular
        vertices = np.linspace(inner_radius, outer_radius, n + 1)
        festim_mesh = F.Mesh1D(vertices, coordinate_system=coordinate_system)
        mesh = festim_mesh.mesh
        r = ufl.SpatialCoordinate(mesh)[0]

        velocity = _velocity(mesh, lambda x: np.vstack([x[0]]))

        material = F.Material(D_0=D_0, E_D=0.0)
        volume = F.VolumeSubdomain1D(
            id=1, borders=[inner_radius, outer_radius], material=material
        )
        left = F.SurfaceSubdomain1D(id=1, x=inner_radius)
        right = F.SurfaceSubdomain1D(id=2, x=outer_radius)
        H = F.Species("H", mobile=True)

        def exact(x):
            return 1.0 + x[0] ** 2

        flux = -D_0 * ufl.grad(exact([r])) + exact([r]) * velocity
        source = ufl.div(r**power * flux) / r**power

        model = F.HydrogenTransportProblem(
            mesh=festim_mesh,
            subdomains=[volume, left, right],
            species=[H],
            temperature=500.0,
            boundary_conditions=[
                F.FixedConcentrationBC(subdomain=left, value=exact, species=H),
                F.FixedConcentrationBC(subdomain=right, value=exact, species=H),
            ],
            sources=[F.ParticleSource(value=source, volume=volume, species=H)],
            drift_terms=[
                F.AdvectionTerm(velocity=velocity, subdomain=volume, species=H)
            ],
            settings=F.Settings(atol=1e-12, rtol=1e-12, transient=False),
        )

        model.initialise()
        model.run()
        errors.append(error_L2(H.post_processing_solution, exact))

    assert errors[-1] < 1e-4
    rate = np.log(errors[0] / errors[1]) / np.log(2)
    assert rate > 1.8, rate


def test_drift_is_applied_in_the_change_of_variable_problem():
    """A drift term used to be silently ignored by the change-of-variable problem.

    With a uniform solubility the change of variable is a rescaling, so the Soret
    equilibrium of ``test_soret_equilibrium_matches_the_closed_form`` still holds for
    the concentration ``u * K_S``.
    """
    length, Q_star, T_cold, T_hot, K_S = 1.0, 0.1, 400.0, 600.0, 3.0
    n = 300

    material = F.Material(D_0=1.0, E_D=0.0, K_S_0=K_S, E_K_S=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    H = F.Species("H", mobile=True)

    model = F.HydrogenTransportProblemDiscontinuousChangeVar(
        mesh=F.Mesh1D(np.linspace(0.0, length, n + 1)),
        subdomains=[volume, left, right],
        species=[H],
        temperature=lambda x: T_cold + (T_hot - T_cold) * x[0] / length,
        initial_conditions=[
            F.InitialConcentration(value=1.0 / K_S, volume=volume, species=H)
        ],
        drift_terms=[F.SoretTerm(species=H, Q_star=Q_star, subdomain=volume)],
        settings=F.Settings(atol=1e-12, rtol=1e-12, transient=True, final_time=20.0),
    )
    model.settings.stepsize = F.Stepsize(0.1)

    model.initialise()
    model.run()

    x, concentration = _profile(H)

    temperature = T_cold + (T_hot - T_cold) * x / length
    shape = np.exp(Q_star / (F.k_B * temperature))
    exact = shape * length / np.trapezoid(shape, x)

    assert np.max(np.abs(concentration - exact)) / np.max(exact) < 1e-4

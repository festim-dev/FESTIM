import warnings

import numpy as np
import pytest
import ufl
from dolfinx import fem

import festim as F
from festim.drift import drift_form, is_zero_velocity
from festim.mesh import CoordinateSystem

test_mesh_1d = F.Mesh1D(vertices=np.linspace(0, 1, 20))
test_functionspace = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))


@pytest.mark.parametrize("cls", [F.SoretTerm, F.ElectromigrationTerm])
@pytest.mark.parametrize("value", ["coucou", 1.0, 1, F.SurfaceSubdomain(id=1)])
def test_subdomain_setter(cls, value):
    """The base class validation applies to every drift term."""
    kwargs = {"Q_star": 0.1} if cls is F.SoretTerm else {"charge": 1, "potential": 0.0}
    with pytest.raises(
        TypeError,
        match=f"Subdomain must be a festim.Subdomain object, not {type(value)}",
    ):
        cls(species=F.Species("H"), subdomain=value, **kwargs)


@pytest.mark.parametrize("cls", [F.SoretTerm, F.ElectromigrationTerm])
def test_species_setter(cls):
    kwargs = {"Q_star": 0.1} if cls is F.SoretTerm else {"charge": 1, "potential": 0.0}
    with pytest.raises(
        TypeError, match=r"elements of species must be of type festim\.Species"
    ):
        cls(species="H", subdomain=F.VolumeSubdomain(id=1, material="m"), **kwargs)


@pytest.mark.parametrize("cls", [F.SoretTerm, F.ElectromigrationTerm])
def test_species_is_wrapped_in_a_list(cls):
    kwargs = {"Q_star": 0.1} if cls is F.SoretTerm else {"charge": 1, "potential": 0.0}
    species = F.Species("H")
    term = cls(
        species=species, subdomain=F.VolumeSubdomain(id=1, material="m"), **kwargs
    )
    assert term.species == [species]


def test_every_drift_term_shares_the_base_class():
    """One assembly for all of them, so none can drift out of divergence form."""
    vol = F.VolumeSubdomain(id=1, material="m")
    terms = [
        F.SoretTerm(species=F.Species("H"), Q_star=0.1, subdomain=vol),
        F.ElectromigrationTerm(
            species=F.Species("H"), charge=1, potential=0.0, subdomain=vol
        ),
        F.AdvectionTerm(velocity=None, subdomain=vol, species=F.Species("H")),
    ]
    assert all(isinstance(term, F.DriftTermBase) for term in terms)
    # the form is not selectable: there is no per-term switch to get the old one back
    assert not any(hasattr(term, "conservative") for term in terms)


def test_uniform_temperature_gives_no_soret_velocity():
    """grad of a spatially constant field is a ufl Zero, so the velocity is reported as
    identically zero -- but the term is still assembled, and contributes nothing."""
    T = fem.Constant(test_mesh_1d.mesh, 500.0)
    D = fem.Constant(test_mesh_1d.mesh, 1.0)
    term = F.SoretTerm(
        species=F.Species("H"),
        Q_star=0.1,
        subdomain=F.VolumeSubdomain(id=1, material="m"),
    )
    term.convert_inputs(function_space=test_functionspace, temperature=T)

    assert is_zero_velocity(term.drift_velocity(D=D, temperature=T))

    c = fem.Function(test_functionspace)
    c.x.array[:] = 1.5
    v = ufl.TestFunction(test_functionspace)
    dx = ufl.Measure("dx", domain=test_mesh_1d.mesh)
    drift = drift_form(
        concentration=c,
        test_function=v,
        velocity=term.drift_velocity(D=D, temperature=T),
        dx=dx,
        coordinate_system=CoordinateSystem.CARTESIAN,
        mesh=test_mesh_1d.mesh,
    )
    assert drift is not None

    # the zero folds into the integrand it shares a measure with, so a formulation
    # carrying the term assembles to exactly the one without it
    diffusion = ufl.dot(ufl.grad(c), ufl.grad(v)) * dx
    without = fem.assemble_vector(fem.form(diffusion))
    with_drift = fem.assemble_vector(fem.form(diffusion + drift))
    assert np.array_equal(without.array, with_drift.array)


@pytest.mark.parametrize(
    "temperature", [500.0, lambda t: 500.0 + t], ids=["float", "callable_of_t"]
)
def test_uniform_temperature_warns_that_soret_does_nothing(temperature):
    """A uniform temperature silently disables Soret, so say so.

    ``lambda t: 500 + t`` is the trap this exists for: it looks like it varies, and it
    does -- in time, not in space -- so grad(T) is still zero.
    """
    material = F.Material(D_0=1.0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    H = F.Species("H", mobile=True)

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0.0, 1.0, 20)),
        subdomains=[volume, F.SurfaceSubdomain1D(id=1, x=0.0)],
        species=[H],
        temperature=temperature,
        drift_terms=[F.SoretTerm(species=H, Q_star=0.1, subdomain=volume)],
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )

    with pytest.warns(UserWarning, match="SoretTerm on volume subdomain 1"):
        model.initialise()


def test_uniform_potential_warns_that_electromigration_does_nothing():
    material = F.Material(D_0=1.0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    H = F.Species("H", mobile=True)

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0.0, 1.0, 20)),
        subdomains=[volume, F.SurfaceSubdomain1D(id=1, x=0.0)],
        species=[H],
        temperature=500.0,
        drift_terms=[
            F.ElectromigrationTerm(species=H, charge=1, potential=2.0, subdomain=volume)
        ],
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )

    with pytest.warns(UserWarning, match="the potential is spatially uniform"):
        model.initialise()


def test_no_warning_when_the_drift_actually_acts():
    material = F.Material(D_0=1.0, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    H = F.Species("H", mobile=True)

    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0.0, 1.0, 20)),
        subdomains=[volume, F.SurfaceSubdomain1D(id=1, x=0.0)],
        species=[H],
        temperature=lambda x: 400.0 + 200.0 * x[0],
        drift_terms=[F.SoretTerm(species=H, Q_star=0.1, subdomain=volume)],
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.initialise()


def test_uniform_potential_gives_no_electromigration_velocity():
    T = fem.Constant(test_mesh_1d.mesh, 500.0)
    D = fem.Constant(test_mesh_1d.mesh, 1.0)
    term = F.ElectromigrationTerm(
        species=F.Species("H"),
        charge=1,
        potential=2.0,
        subdomain=F.VolumeSubdomain(id=1, material="m"),
    )
    term.convert_inputs(function_space=test_functionspace, temperature=T)

    assert is_zero_velocity(term.drift_velocity(D=D, temperature=T))


def test_soret_velocity_expression():
    """-D Q* / (k_B T^2) grad(T), the term of eq. (Soret) in the theory guide."""
    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Function(V)
    T.interpolate(lambda x: 300.0 + 100.0 * x[0])
    D = fem.Constant(test_mesh_1d.mesh, 2.0)

    term = F.SoretTerm(
        species=F.Species("H"),
        Q_star=0.3,
        subdomain=F.VolumeSubdomain(id=1, material="m"),
    )
    term.convert_inputs(function_space=V, temperature=T)
    velocity = term.drift_velocity(D=D, temperature=T)

    expected = -D * 0.3 / (F.k_B * T**2) * ufl.grad(T)
    difference = fem.assemble_scalar(
        fem.form(ufl.dot(velocity - expected, velocity - expected) * ufl.dx)
    )
    assert np.isclose(difference, 0.0)


def test_electromigration_velocity_sign_follows_the_charge():
    """A positive species drifts down the potential gradient, a negative one up it."""
    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Constant(test_mesh_1d.mesh, 500.0)
    phi = fem.Function(V)
    phi.interpolate(lambda x: 1.0 - x[0])  # grad(phi) = -1, so drift is +x for z > 0
    D = fem.Constant(test_mesh_1d.mesh, 1.0)

    velocities = {}
    for charge in (+1, -1):
        term = F.ElectromigrationTerm(
            species=F.Species("H"),
            charge=charge,
            potential=phi,
            subdomain=F.VolumeSubdomain(id=1, material="m"),
        )
        term.convert_inputs(function_space=V, temperature=T)
        velocity = term.drift_velocity(D=D, temperature=T)
        velocities[charge] = fem.assemble_scalar(
            fem.form(velocity[0] * ufl.dx(domain=test_mesh_1d.mesh))
        )

    assert velocities[+1] > 0
    assert np.isclose(velocities[+1], -velocities[-1])


def _terms():
    vol = F.VolumeSubdomain(id=1, material="m")
    return (
        F.AdvectionTerm(velocity=None, subdomain=vol, species=F.Species("H")),
        F.SoretTerm(species=F.Species("H"), Q_star=0.1, subdomain=vol),
    )


def test_advection_terms_is_deprecated():
    advection, _ = _terms()
    with pytest.warns(DeprecationWarning, match="use drift_terms instead"):
        model = F.HydrogenTransportProblem(advection_terms=[advection])
    assert model.drift_terms == [advection]

    with pytest.warns(DeprecationWarning, match="use drift_terms instead"):
        model.advection_terms


def test_drift_terms_alone_is_not_deprecated():
    advection, soret = _terms()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model = F.HydrogenTransportProblem(drift_terms=[soret, advection])
    assert model.drift_terms == [soret, advection]


def test_advection_terms_is_appended_not_substituted():
    """The reason it is not a plain alias.

    Assigning the old attribute must not take the whole drift_terms list with it: a
    Soret term set separately would be silently dropped, and the model would quietly
    solve different physics.
    """
    advection, soret = _terms()

    with pytest.warns(DeprecationWarning):
        both = F.HydrogenTransportProblem(
            drift_terms=[soret], advection_terms=[advection]
        )
    assert both.drift_terms == [soret, advection]

    model = F.HydrogenTransportProblem(drift_terms=[soret])
    with pytest.warns(DeprecationWarning):
        model.advection_terms = [advection]
    assert model.drift_terms == [soret, advection]

    # and it still reads as assignment: setting it twice must not duplicate
    with pytest.warns(DeprecationWarning):
        model.advection_terms = [advection]
    assert model.drift_terms == [soret, advection]

    # the getter is the advection subset, not the whole list
    with pytest.warns(DeprecationWarning):
        assert model.advection_terms == [advection]


def test_drift_form_is_the_divergence_form():
    """``-c v . grad(w)``, not ``(v . grad(c)) w``.

    A direct guard on the assembled form, since the two are no longer selectable at
    runtime and so cannot be compared by solving the same problem twice. With a
    non-solenoidal velocity they differ by ``c div(v)``, which is what this catches.
    """
    V = test_functionspace
    c = fem.Function(V)
    c.interpolate(lambda x: 1.0 + x[0] ** 2)
    w = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=test_mesh_1d.mesh)

    # div(v) = 1
    velocity = ufl.as_vector([ufl.SpatialCoordinate(test_mesh_1d.mesh)[0]])

    assembled = fem.assemble_vector(
        fem.form(
            drift_form(
                concentration=c,
                test_function=w,
                velocity=velocity,
                dx=dx,
                coordinate_system=CoordinateSystem.CARTESIAN,
                mesh=test_mesh_1d.mesh,
            )
        )
    ).array
    divergence = fem.assemble_vector(
        fem.form(-c * ufl.dot(velocity, ufl.grad(w)) * dx)
    ).array
    non_conservative = fem.assemble_vector(
        fem.form(ufl.inner(ufl.dot(ufl.grad(c), velocity), w) * dx)
    ).array

    assert np.allclose(assembled, divergence)
    assert not np.allclose(assembled, non_conservative)


def test_unknown_coordinate_system_raises():
    with pytest.raises(NotImplementedError, match="Unknown coordinate system"):
        drift_form(
            concentration=fem.Function(test_functionspace),
            test_function=ufl.TestFunction(test_functionspace),
            velocity=ufl.as_vector([fem.Constant(test_mesh_1d.mesh, 1.0)]),
            dx=ufl.Measure("dx", domain=test_mesh_1d.mesh),
            coordinate_system="not a coordinate system",
            mesh=test_mesh_1d.mesh,
        )


@pytest.mark.parametrize("value", ["H", 1.0, F.VolumeSubdomain(id=1, material="m")])
def test_outflow_bc_species_setter(value):
    with pytest.raises(TypeError, match=r"species must be of type festim\.Species"):
        F.OutflowBC(subdomain=F.SurfaceSubdomain(id=1), species=value)


def test_base_class_cannot_be_instantiated_without_a_drift_velocity():
    with pytest.raises(TypeError, match="abstract"):
        F.DriftTermBase(
            subdomain=F.VolumeSubdomain(id=1, material="m"), species=F.Species("H")
        )

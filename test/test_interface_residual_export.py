import pathlib
from itertools import pairwise

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import festim as F
from festim import k_B
from festim.material import SolubilityLaw


def generate_two_material_mesh(n=10):
    """Unit square split in half at y=0.5, with the interface tagged 5.

    Returns:
        the mesh, the facet meshtags and the cell meshtags
    """
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle
    )
    tdim = mesh.topology.dim
    fdim = tdim - 1

    top_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[1], 1.0)
    )
    bottom_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    num_facets_local = (
        mesh.topology.index_map(fdim).size_local
        + mesh.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[top_facets] = 1
    values[bottom_facets] = 2

    bottom_cells = dolfinx.mesh.locate_entities(
        mesh, tdim, lambda x: x[1] <= 0.5 + 1e-14
    )
    num_cells_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    cells = np.full(num_cells_local, 4, dtype=np.int32)
    cells[bottom_cells] = 3
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
    )

    all_b_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(3), tdim, fdim
    )
    all_t_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(4), tdim, fdim
    )
    values[np.intersect1d(all_b_facets, all_t_facets)] = 5

    mt = dolfinx.mesh.meshtags(mesh, fdim, facets, values)
    return mesh, mt, ct


def build_initialised_model(
    law_0="sievert",
    law_1="sievert",
    K_S_0_0=3.0,
    E_K_S_0=0.0,
    K_S_0_1=6.0,
    E_K_S_1=0.0,
    temperature=500,
    tmpdir=None,
    n=10,
    final_time=None,
    stepsize=0.5,
):
    """Build and initialise (but do not run) a two-material discontinuous model.

    Subdomain 0 of the interface is the bottom half, subdomain 1 the top half.

    Args:
        final_time: if given, the model is transient up to this time, otherwise
            it is steady state
        stepsize: the time step of a transient model

    Returns:
        the initialised model, the species and the residual export
    """
    mesh, mt, ct = generate_two_material_mesh(n)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_bottom = F.Material(
        D_0=5.0, E_D=0, K_S_0=K_S_0_0, E_K_S=E_K_S_0, solubility_law=law_0
    )
    material_top = F.Material(
        D_0=2.0, E_D=0, K_S_0=K_S_0_1, E_K_S=E_K_S_1, solubility_law=law_1
    )

    bottom_domain = F.VolumeSubdomain(3, material=material_bottom)
    top_domain = F.VolumeSubdomain(4, material=material_top)
    top_surface = F.SurfaceSubdomain(id=1)
    bottom_surface = F.SurfaceSubdomain(id=2)
    my_model.subdomains = [bottom_domain, top_domain, top_surface, bottom_surface]

    interface = F.Interface(5, (bottom_domain, top_domain))
    my_model.interfaces = [interface]

    H = F.Species("H", mobile=True)
    H.subdomains = [bottom_domain, top_domain]
    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(top_surface, value=1, species=H),
        F.FixedConcentrationBC(bottom_surface, value=1, species=H),
    ]
    my_model.temperature = temperature
    if final_time is None:
        my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
    else:
        my_model.settings = F.Settings(
            atol=1e-10, rtol=1e-10, transient=True, final_time=final_time
        )
        my_model.settings.stepsize = stepsize
        my_model.show_progress_bar = False

    export = F.VTXInterfaceResidualExport(
        field=H, filename=str(tmpdir.join("residual.bp")), interface=interface
    )
    my_model.exports = [export]

    my_model.initialise()

    return my_model, H, export


def impose_concentrations(species, interface, c_0_np, c_1_np):
    """Overwrite the post-processing solutions on both sides of the interface."""
    subdomain_0, subdomain_1 = interface.subdomains
    species.subdomain_to_post_processing_solution[subdomain_0].interpolate(c_0_np)
    species.subdomain_to_post_processing_solution[subdomain_1].interpolate(c_1_np)


def expected_side_term(c, K_S, law, other_law):
    """The exact interface condition term of one side, evaluated with numpy."""
    if law != other_law and law == SolubilityLaw.SIEVERT:
        return (c / K_S) ** 2
    return c / K_S


# a deliberately non-trivial concentration field on each side of the interface
def c_0_np(x):
    return 3 + np.sin(2 * np.pi * x[0]) + 0.5 * np.cos(np.pi * x[1])


def c_1_np(x):
    return 7 - 2 * np.sin(3 * np.pi * x[0]) + 0.25 * x[0] ** 2


def T_np(x):
    return 400 + 100 * x[0]


LAW_COMBINATIONS = [
    ("sievert", "sievert"),
    ("henry", "henry"),
    ("henry", "sievert"),
    ("sievert", "henry"),
]


def exact_residual_at_dofs(export, law_0, law_1, K_S_0_0, E_K_S_0, K_S_0_1, E_K_S_1, T):
    """Compute |f_1 - f_0| analytically at the dofs of the interface submesh.

    The interface dofs are vertices of the parent mesh, where the CG1
    concentration fields take their exact nodal values, so this is the exact
    residual the export should produce, up to machine precision.
    """
    x = export.function.function_space.tabulate_dof_coordinates().T
    T_at_dofs = T(x) if callable(T) else T

    K_S_0 = K_S_0_0 * np.exp(-E_K_S_0 / (k_B * T_at_dofs))
    K_S_1 = K_S_0_1 * np.exp(-E_K_S_1 / (k_B * T_at_dofs))

    law_0, law_1 = SolubilityLaw.from_string(law_0), SolubilityLaw.from_string(law_1)
    f_0 = expected_side_term(c_0_np(x), K_S_0, law_0, law_1)
    f_1 = expected_side_term(c_1_np(x), K_S_1, law_1, law_0)

    return f_0, f_1, np.abs(f_1 - f_0)


@pytest.mark.parametrize("law_0, law_1", LAW_COMBINATIONS)
def test_residual_matches_analytical_value(tmpdir, law_0, law_1):
    """The exported residual matches |f_1 - f_0| computed analytically."""
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    _, H, export = build_initialised_model(
        law_0=law_0, law_1=law_1, K_S_0_0=K_S_0_0, K_S_0_1=K_S_0_1, tmpdir=tmpdir
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)

    # RUN
    export.update()

    # TEST
    _, _, expected = exact_residual_at_dofs(
        export, law_0, law_1, K_S_0_0, 0.0, K_S_0_1, 0.0, T=500
    )
    assert np.allclose(export.function.x.array, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("law_0, law_1", LAW_COMBINATIONS)
def test_residual_matches_analytical_value_temperature_dependent(tmpdir, law_0, law_1):
    """The residual matches the analytical value for a space-dependent K_S."""
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    E_K_S_0, E_K_S_1 = 0.2, 0.4
    _, H, export = build_initialised_model(
        law_0=law_0,
        law_1=law_1,
        K_S_0_0=K_S_0_0,
        E_K_S_0=E_K_S_0,
        K_S_0_1=K_S_0_1,
        E_K_S_1=E_K_S_1,
        temperature=T_np,
        tmpdir=tmpdir,
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)

    # RUN
    export.update()

    # TEST
    _, _, expected = exact_residual_at_dofs(
        export, law_0, law_1, K_S_0_0, E_K_S_0, K_S_0_1, E_K_S_1, T=T_np
    )
    assert np.allclose(export.function.x.array, expected, rtol=1e-12, atol=1e-12)


def test_f_0_matches_analytical_value(tmpdir):
    """The exported f_0 field matches the analytical term of side 0."""
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    _, H, export = build_initialised_model(
        law_0="sievert", law_1="henry", K_S_0_0=K_S_0_0, K_S_0_1=K_S_0_1, tmpdir=tmpdir
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)

    # RUN
    export.update()

    # TEST
    expected_f_0, _, _ = exact_residual_at_dofs(
        export, "sievert", "henry", K_S_0_0, 0.0, K_S_0_1, 0.0, T=500
    )
    assert np.allclose(
        export._f_0_interface.x.array, expected_f_0, rtol=1e-12, atol=1e-12
    )


def test_f_1_matches_analytical_value(tmpdir):
    """The exported f_1 field matches the analytical term of side 1."""
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    _, H, export = build_initialised_model(
        law_0="sievert", law_1="henry", K_S_0_0=K_S_0_0, K_S_0_1=K_S_0_1, tmpdir=tmpdir
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)

    # RUN
    export.update()

    # TEST
    _, expected_f_1, _ = exact_residual_at_dofs(
        export, "sievert", "henry", K_S_0_0, 0.0, K_S_0_1, 0.0, T=500
    )
    assert np.allclose(
        export._f_1_interface.x.array, expected_f_1, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("law_0, law_1", LAW_COMBINATIONS)
def test_residual_is_zero_when_condition_is_satisfied(tmpdir, law_0, law_1):
    """The residual vanishes when both sides satisfy the interface condition."""
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    _, H, export = build_initialised_model(
        law_0=law_0, law_1=law_1, K_S_0_0=K_S_0_0, K_S_0_1=K_S_0_1, tmpdir=tmpdir
    )

    # build c_1 from c_0 so that f_1 == f_0 exactly
    _law_0 = SolubilityLaw.from_string(law_0)
    _law_1 = SolubilityLaw.from_string(law_1)

    def c_1_matching(x):
        f_0 = expected_side_term(c_0_np(x), K_S_0_0, _law_0, _law_1)
        if _law_0 != _law_1 and _law_1 == SolubilityLaw.SIEVERT:
            return K_S_0_1 * np.sqrt(f_0)
        return K_S_0_1 * f_0

    impose_concentrations(H, export.interface, c_0_np, c_1_matching)

    # RUN
    export.update()

    # TEST
    assert np.allclose(export.function.x.array, 0.0, atol=1e-12)


def test_residual_is_non_negative(tmpdir):
    """The residual is an absolute value and is therefore never negative."""
    # BUILD
    _, H, export = build_initialised_model(
        law_0="henry", law_1="sievert", tmpdir=tmpdir
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)

    # RUN
    export.update()

    # TEST
    assert np.all(export.function.x.array >= 0.0)


def test_residual_export_not_supported_by_continuous_problem(tmpdir):
    """A clear error is raised if the export is used without interfaces."""
    # BUILD
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, num=10))
    mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    my_model.subdomains = [vol]
    H = F.Species("H")
    my_model.species = [H]
    my_model.temperature = 500
    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
    my_model.exports = [
        F.VTXInterfaceResidualExport(
            field=H,
            filename=str(tmpdir.join("residual.bp")),
            interface=F.Interface(5, (vol, vol)),
        )
    ]

    # RUN + TEST
    with pytest.raises(NotImplementedError, match="Discontinuous"):
        my_model.initialise()


def dummy_interface():
    """An interface between two subdomains, enough to build an export."""
    mat = F.Material(D_0=1, E_D=0)
    return F.Interface(
        id=5,
        subdomains=(
            F.VolumeSubdomain(id=1, material=mat),
            F.VolumeSubdomain(id=2, material=mat),
        ),
    )


def test_export_stores_field(tmpdir):
    """The export stores the species it was given."""
    species = F.Species("H")
    my_export = F.VTXInterfaceResidualExport(
        field=species,
        filename=str(tmpdir.join("residual.bp")),
        interface=dummy_interface(),
    )
    assert my_export.field is species


def test_export_stores_interface(tmpdir):
    """The export stores the interface it was given."""
    interface = dummy_interface()
    my_export = F.VTXInterfaceResidualExport(
        field=F.Species("H"),
        filename=str(tmpdir.join("residual.bp")),
        interface=interface,
    )
    assert my_export.interface is interface


def test_export_suffix_converter(tmpdir):
    """A non-.bp filename is corrected to a .bp suffix."""
    my_export = F.VTXInterfaceResidualExport(
        field=F.Species("H"),
        filename=str(tmpdir.join("residual.txt")),
        interface=dummy_interface(),
    )
    assert my_export.filename.suffix == ".bp"


@pytest.mark.parametrize("value", [1, "H", None])
def test_field_setter_raises_type_error(tmpdir, value):
    """field must be a festim.Species."""
    with pytest.raises(TypeError, match=r"field must be a festim\.Species"):
        F.VTXInterfaceResidualExport(
            field=value,
            filename=str(tmpdir.join("residual.bp")),
            interface=dummy_interface(),
        )


@pytest.mark.parametrize("value", [1, "interface", None])
def test_interface_setter_raises_type_error(tmpdir, value):
    """interface must be a festim.Interface."""
    with pytest.raises(TypeError, match=r"interface must be a festim\.Interface"):
        F.VTXInterfaceResidualExport(
            field=F.Species("H"),
            filename=str(tmpdir.join("residual.bp")),
            interface=value,
        )


@pytest.mark.parametrize(
    "temperature, T_final",
    [
        (lambda t: 500 + 100 * t, lambda x: 600.0),
        (lambda x, t: 400 + 100 * x[0] + 50 * t, lambda x: 450 + 100 * x[0]),
    ],
    ids=["uniform", "space-dependent"],
)
def test_residual_uses_the_temperature_of_the_final_timestep(
    tmpdir, temperature, T_final
):
    """After a transient run the residual reflects the temperature at that time.

    The expressions are built once, during initialise, and reference the
    temperature that the problem then updates in place at every step. If they
    captured the initial temperature instead, the residual would silently be
    evaluated at t=0 for the whole simulation. E_K_S is non-zero on both sides so
    that the temperature actually reaches the residual through K_S.
    """
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    E_K_S_0, E_K_S_1 = 0.2, 0.4
    my_model, H, export = build_initialised_model(
        law_0="sievert",
        law_1="henry",
        K_S_0_0=K_S_0_0,
        E_K_S_0=E_K_S_0,
        K_S_0_1=K_S_0_1,
        E_K_S_1=E_K_S_1,
        temperature=temperature,
        tmpdir=tmpdir,
        final_time=1.0,
    )

    # RUN
    my_model.run()
    impose_concentrations(H, export.interface, c_0_np, c_1_np)
    export.update()

    # TEST
    _, _, expected = exact_residual_at_dofs(
        export, "sievert", "henry", K_S_0_0, E_K_S_0, K_S_0_1, E_K_S_1, T=T_final
    )
    assert np.allclose(export.function.x.array, expected, rtol=1e-12, atol=1e-12)


def test_residual_is_written_at_every_timestep(tmpdir):
    """A transient run drives the export once per step without error."""
    # BUILD
    my_model, _, export = build_initialised_model(
        law_0="sievert",
        law_1="henry",
        temperature=lambda t: 500 + 100 * t,
        tmpdir=tmpdir,
        final_time=1.0,
        stepsize=0.25,
    )
    written = []
    original_write = export.writer.write
    export.writer.write = lambda t: written.append(t) or original_write(t)

    # RUN
    my_model.run()

    # TEST
    assert written == [0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("law_0, law_1", LAW_COMBINATIONS)
def test_residual_matches_analytical_value_uniform_temperature_with_E_K_S(
    tmpdir, law_0, law_1
):
    """The residual is correct for a uniform temperature and a non-zero E_K_S.

    Regression test. A uniform temperature is a fem.Constant carrying the parent
    mesh as its domain, which cannot appear in an expression alongside the
    interface-submesh concentrations. When E_K_S is zero the temperature cancels
    out of K_S and the clash never shows, so this combination, which is the common
    one in practice, is the only thing that exposes it.
    """
    # BUILD
    K_S_0_0, K_S_0_1 = 3.0, 6.0
    E_K_S_0, E_K_S_1 = 0.2, 0.4
    _, H, export = build_initialised_model(
        law_0=law_0,
        law_1=law_1,
        K_S_0_0=K_S_0_0,
        E_K_S_0=E_K_S_0,
        K_S_0_1=K_S_0_1,
        E_K_S_1=E_K_S_1,
        temperature=500,
        tmpdir=tmpdir,
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)

    # RUN
    export.update()

    # TEST
    _, _, expected = exact_residual_at_dofs(
        export, law_0, law_1, K_S_0_0, E_K_S_0, K_S_0_1, E_K_S_1, T=500
    )
    assert np.allclose(export.function.x.array, expected, rtol=1e-12, atol=1e-12)


@pytest.fixture
def restore_log_level():
    """Set the dolfinx log level for one test and put it back afterwards."""
    original = dolfinx.log.get_log_level()
    yield dolfinx.log.set_log_level
    dolfinx.log.set_log_level(original)


def build_export_with_known_residual(tmpdir):
    """An initialised export whose residual comes from the imposed concentrations."""
    _, H, export = build_initialised_model(
        law_0="sievert", law_1="henry", K_S_0_0=3.0, K_S_0_1=6.0, tmpdir=tmpdir
    )
    impose_concentrations(H, export.interface, c_0_np, c_1_np)
    export.update()
    f_0, f_1, residual = exact_residual_at_dofs(
        export, "sievert", "henry", 3.0, 0.0, 6.0, 0.0, T=500
    )
    return export, f_0, f_1, residual


def test_statistics_max_matches_the_analytical_residual(tmpdir):
    """The reported max is the largest analytical residual on the interface."""
    export, _, _, residual = build_export_with_known_residual(tmpdir)

    assert np.isclose(export.compute_statistics()["max"], np.max(residual), rtol=1e-12)


def test_statistics_mean_matches_the_analytical_residual(tmpdir):
    """The reported mean is the average analytical residual on the interface."""
    export, _, _, residual = build_export_with_known_residual(tmpdir)

    assert np.isclose(
        export.compute_statistics()["mean"], np.mean(residual), rtol=1e-12
    )


def test_statistics_max_relative_is_scaled_by_the_side_terms(tmpdir):
    """The relative max divides the largest mismatch by the largest side term."""
    export, f_0, f_1, residual = build_export_with_known_residual(tmpdir)

    expected = np.max(residual) / max(np.max(np.abs(f_0)), np.max(np.abs(f_1)))
    assert np.isclose(export.compute_statistics()["max_relative"], expected, rtol=1e-12)


def test_statistics_are_zero_when_the_condition_is_satisfied(tmpdir):
    """A satisfied interface condition reports a zero maximum residual."""
    _, H, export = build_initialised_model(
        law_0="henry", law_1="henry", K_S_0_0=3.0, K_S_0_1=6.0, tmpdir=tmpdir
    )
    # same law on both sides, so c_1 = (K_S_1 / K_S_0) * c_0 satisfies it exactly
    impose_concentrations(H, export.interface, c_0_np, lambda x: 6.0 / 3.0 * c_0_np(x))
    export.update()

    assert export.compute_statistics()["max"] < 1e-12


@pytest.mark.parametrize(
    "level", [dolfinx.log.LogLevel.WARNING, dolfinx.log.LogLevel.ERROR]
)
def test_log_statistics_is_silent_above_info(
    tmpdir, monkeypatch, restore_log_level, level
):
    """Nothing is logged, and nothing is computed, unless INFO is selected."""
    export, _, _, _ = build_export_with_known_residual(tmpdir)
    restore_log_level(level)

    lines = []
    monkeypatch.setattr(dolfinx.log, "log", lambda lvl, msg: lines.append(msg))
    monkeypatch.setattr(
        export,
        "compute_statistics",
        lambda: pytest.fail("statistics computed while logging is off"),
    )
    export.log_statistics(0.5)

    assert lines == []


@pytest.mark.parametrize(
    "level", [dolfinx.log.LogLevel.INFO, dolfinx.log.LogLevel.DEBUG]
)
def test_log_statistics_reports_the_interface_at_info(
    tmpdir, monkeypatch, restore_log_level, level
):
    """At INFO the line names the interface and carries the three statistics."""
    export, _, _, residual = build_export_with_known_residual(tmpdir)
    restore_log_level(level)

    lines = []
    monkeypatch.setattr(dolfinx.log, "log", lambda lvl, msg: lines.append(msg))
    export.log_statistics(0.5)

    assert lines == [
        f"Interface 5 residual (H) t=5.00000e-01 ; max={np.max(residual):.5e} ; "
        f"avg={np.mean(residual):.5e} ; "
        f"max_relative={export.compute_statistics()['max_relative']:.5e}"
    ]


def build_1d_model(
    law_0="sievert",
    law_1="henry",
    K_S_0_0=2.0,
    K_S_0_1=5.0,
    interface_positions=(1.0,),
    penalty_terms=None,
    filename=None,
):
    """Build and initialise a 1D discontinuous model with one export per interface.

    In 1D the interface is a single point, so the residual lives on a mesh of
    dimension 0 and its function space is DG0 rather than CG1.

    Args:
        interface_positions: where the subdomains meet; one interface per position
        penalty_terms: the penalty term of each interface, defaulting to 1e4

    Returns:
        the initialised model, the species and the list of exports
    """
    edges = [0.0, *interface_positions, interface_positions[-1] + 1.0]
    vertices = np.concatenate(
        [
            np.linspace(a, b, num=10, endpoint=(b == edges[-1]))
            for a, b in pairwise(edges)
        ]
    )

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh1D(vertices=vertices)
    my_model.show_progress_bar = False

    laws = [law_0, law_1] * len(edges)
    solubilities = [K_S_0_0, K_S_0_1] * len(edges)
    volumes = [
        F.VolumeSubdomain1D(
            id=i + 1,
            borders=[a, b],
            material=F.Material(
                D_0=1, E_D=0, K_S_0=solubilities[i], E_K_S=0, solubility_law=laws[i]
            ),
        )
        for i, (a, b) in enumerate(pairwise(edges))
    ]
    left = F.SurfaceSubdomain1D(id=90, x=edges[0])
    right = F.SurfaceSubdomain1D(id=91, x=edges[-1])
    my_model.subdomains = [*volumes, left, right]

    penalties = penalty_terms or [1e4] * (len(volumes) - 1)
    interfaces = [
        F.Interface(50 + i, (volumes[i], volumes[i + 1]), penalty_term=penalties[i])
        for i in range(len(volumes) - 1)
    ]
    my_model.interfaces = interfaces

    H = F.Species("H", mobile=True, subdomains=volumes)
    my_model.species = [H]
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(left, value=3, species=H),
        F.FixedConcentrationBC(right, value=1, species=H),
    ]
    my_model.temperature = 500
    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)

    exports = [
        F.VTXInterfaceResidualExport(
            field=H,
            interface=interface,
            filename=None if filename is None else f"{filename}_{interface.id}.bp",
        )
        for interface in interfaces
    ]
    my_model.exports = exports

    my_model.initialise()
    # solve: without this every field is zero and the residual is trivially zero
    my_model.run()

    return my_model, H, exports


def residual_at_a_point(species, interface, position, law_0, law_1, K_0, K_1):
    """The analytical |f_1 - f_0| at an interface point, from the solved fields."""
    values = []
    for subdomain, K, law, other in (
        (interface.subdomains[0], K_0, law_0, law_1),
        (interface.subdomains[1], K_1, law_1, law_0),
    ):
        c = species.subdomain_to_post_processing_solution[subdomain]
        x = c.function_space.tabulate_dof_coordinates()
        at_point = c.x.array[np.isclose(x[:, 0], position)][0]
        values.append(
            expected_side_term(
                at_point,
                K,
                SolubilityLaw.from_string(law),
                SolubilityLaw.from_string(other),
            )
        )
    return abs(values[1] - values[0])


@pytest.mark.parametrize("law_0, law_1", LAW_COMBINATIONS)
def test_residual_in_1d_matches_analytical_value(law_0, law_1):
    """In 1D the residual is the analytical mismatch at the single interface point."""
    # BUILD
    K_S_0_0, K_S_0_1 = 2.0, 5.0
    _, H, exports = build_1d_model(
        law_0=law_0, law_1=law_1, K_S_0_0=K_S_0_0, K_S_0_1=K_S_0_1
    )

    # RUN
    export = exports[0]

    # TEST
    expected = residual_at_a_point(
        H, export.interface, 1.0, law_0, law_1, K_S_0_0, K_S_0_1
    )
    assert np.isclose(export.function.x.array[0], expected, rtol=1e-10)


def test_residual_in_1d_lives_on_a_single_dof():
    """The 1D interface submesh is a point, so the residual has exactly one value."""
    _, _, exports = build_1d_model()

    assert exports[0].function.x.array.size == 1


def test_statistics_in_1d_agree_with_the_single_value():
    """With one dof the max and the mean are the same number."""
    _, _, exports = build_1d_model()

    stats = exports[0].compute_statistics()
    assert np.isclose(stats["max"], stats["mean"], rtol=1e-12)


def test_each_interface_reports_its_own_residual():
    """With several interfaces, each export reports the residual of its own."""
    # BUILD
    K_S_0_0, K_S_0_1 = 2.0, 5.0
    _, H, exports = build_1d_model(
        law_0="sievert",
        law_1="henry",
        K_S_0_0=K_S_0_0,
        K_S_0_1=K_S_0_1,
        interface_positions=(1.0, 2.0),
    )

    # TEST
    assert len(exports) == 2
    for export, position, K_0, K_1 in (
        (exports[0], 1.0, K_S_0_0, K_S_0_1),
        (exports[1], 2.0, K_S_0_1, K_S_0_0),
    ):
        law_0, law_1 = (
            export.interface.subdomains[0].material.solubility_law,
            export.interface.subdomains[1].material.solubility_law,
        )
        expected = residual_at_a_point(
            H,
            export.interface,
            position,
            str(law_0.name).lower(),
            str(law_1.name).lower(),
            K_0,
            K_1,
        )
        assert np.isclose(export.function.x.array[0], expected, rtol=1e-10), (
            f"interface {export.interface.id} at x={position}"
        )


def test_each_export_follows_its_own_interfaces_penalty():
    """An export reports the residual of its own interface, not a neighbour's.

    The two interfaces are otherwise symmetric and give identical residuals, so
    they are given different penalty terms: the weakly enforced one must then show
    the larger mismatch. Swapping the two exports would invert this.
    """
    _, _, exports = build_1d_model(
        interface_positions=(1.0, 2.0), penalty_terms=[1e2, 1e5]
    )

    weak, strong = exports[0].function.x.array[0], exports[1].function.x.array[0]
    assert weak > 100 * strong, f"{weak=} is not much larger than {strong=}"


def test_no_filename_means_no_writer():
    """Without a filename the export has no writer and writes nothing."""
    _, _, exports = build_1d_model(filename=None)

    assert exports[0].writer is None


def test_no_filename_still_computes_the_residual():
    """The residual is still available when no file is written."""
    _, _, exports = build_1d_model(filename=None)

    assert exports[0].compute_statistics()["max"] > 0


def test_no_filename_writes_no_file(tmpdir):
    """Nothing is created on disk when no filename is given."""
    # the model has to stay referenced: __del__ closes the writers
    _model, _, _ = build_1d_model(filename=None)

    assert list(pathlib.Path(str(tmpdir)).iterdir()) == []


def test_filename_creates_one_file_per_interface(tmpdir):
    """A filename still produces a .bp file per interface, written during run()."""
    _model, _, _ = build_1d_model(
        interface_positions=(1.0, 2.0), filename=str(tmpdir.join("residual"))
    )

    written = sorted(p.name for p in pathlib.Path(str(tmpdir)).iterdir())
    assert written == ["residual_50.bp", "residual_51.bp"]


def test_filename_defaults_to_none(tmpdir):
    """filename is optional; the export is usable without one."""
    my_export = F.VTXInterfaceResidualExport(
        field=F.Species("H"), interface=dummy_interface()
    )
    assert my_export.filename is None

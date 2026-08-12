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
):
    """Build and initialise (but do not run) a two-material discontinuous model.

    Subdomain 0 of the interface is the bottom half, subdomain 1 the top half.

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
    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

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

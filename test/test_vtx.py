from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import festim as F

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def test_vtx_export_one_function(tmpdir):
    """Test can add one function to a vtx export."""
    u = dolfinx.fem.Function(V)
    sp = F.Species("H")
    sp.post_processing_solution = u
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(filename, field=sp)

    functions = my_export.get_functions()
    assert len(functions) == 1
    assert functions[0] == u


def test_vtx_export_two_functions(tmpdir):
    """Test can add two functions to a vtx export."""
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    sp1 = F.Species("1")
    sp2 = F.Species("2")
    sp1.post_processing_solution = u
    sp2.post_processing_solution = v
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(filename, field=[sp1, sp2])

    functions = my_export.get_functions()
    assert len(functions) == 2
    assert functions[0] == u
    assert functions[1] == v


@pytest.mark.skip(reason="Not implemented")
def test_vtx_export_subdomain():
    """Test that given multiple subdomains in problem, only correct functions are
    extracted from species."""
    pass


def test_vtx_suffix_converter(tmpdir):
    filename = str(tmpdir.join("my_export.txt"))
    my_export = F.VTXSpeciesExport(filename, field=[])
    assert my_export.filename.suffix == ".bp"


def test_vtx_DG(tmpdir):
    """Test VTX export setup for DG formulation."""
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    my_mat = F.Material(D_0=3, E_D=2, K_S_0=1, E_K_S=0, name="mat")

    s0 = F.VolumeSubdomain1D(1, borders=[0.0, 2], material=my_mat)
    s1 = F.VolumeSubdomain1D(2, borders=[2, 4], material=my_mat)
    l0 = F.SurfaceSubdomain1D(1, x=0.0)
    l1 = F.SurfaceSubdomain1D(2, x=4.0)
    my_model.interfaces = [F.Interface(6, (s0, s1))]

    my_model.temperature = 55
    my_model.subdomains = [s0, s1, l0, l1]
    # NOTE: Ask Remi why `H` has to live in both s0 and s1
    my_model.species = [
        F.Species("H", subdomains=[s0, s1]),
        F.Species("T", subdomains=[s0, s1], mobile=False),
    ]

    filename = str(tmpdir.join("my_export.txt"))
    my_export = F.VTXSpeciesExport(filename, field=my_model.species, subdomain=s0)
    assert my_export.filename.suffix == ".bp"
    my_model.exports = [my_export]
    my_model.settings = F.Settings(atol=1, rtol=0.1)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    my_model.initialise()
    assert len(my_export.get_functions()) == 2

    all_vtx = [e for e in my_model.exports if isinstance(e, F.ExportBaseClass)]

    assert len(all_vtx) == 1


@pytest.mark.parametrize("checkpoint", [True, False])
def test_vtx_integration_with_h_transport_problem(tmpdir, checkpoint):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    my_mat = F.Material(D_0=1, E_D=0, name="mat")
    my_model.subdomains = [
        F.VolumeSubdomain1D(1, borders=[0.0, 4.0], material=my_mat),
        F.SurfaceSubdomain1D(1, x=0.0),
        F.SurfaceSubdomain1D(2, x=4.0),
    ]
    my_model.species = [F.Species("H")]
    my_model.temperature = lambda t: 500 + t

    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(
        filename, field=my_model.species[0], checkpoint=checkpoint
    )
    my_model.exports = [my_export]
    my_model.settings = F.Settings(atol=1, rtol=0.1)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    my_model.initialise()
    assert len(my_export.get_functions()) == 1

    all_vtx = [e for e in my_model.exports if isinstance(e, F.ExportBaseClass)]

    assert len(all_vtx) == 1


@pytest.mark.parametrize(
    "T",
    [
        500,
        lambda t: 500 + t,
        lambda x: 500 + 200 * x[0],
        lambda x, t: 500 + 200 * x[0] + t,
    ],
)
def test_vtx_temperature(T, tmpdir):
    """Tests that VTX temperature exports work with HydrogenTransportProblem."""
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    my_mat = F.Material(D_0=1, E_D=0, name="mat")
    my_model.subdomains = [
        F.VolumeSubdomain1D(1, borders=[0.0, 4.0], material=my_mat),
        F.SurfaceSubdomain1D(1, x=0.0),
        F.SurfaceSubdomain1D(2, x=4.0),
    ]
    my_model.species = [F.Species("H")]
    my_model.temperature = T

    filename = str(tmpdir.join("my_export.bp"))

    my_export = F.VTXTemperatureExport(filename)
    my_model.exports = [my_export]
    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    all_vtx = [e for e in my_model.exports if isinstance(e, F.ExportBaseClass)]

    assert len(all_vtx) == 1


def test_field_attribute_is_always_list():
    """Test that the field attribute is always a list."""
    my_export = F.VTXSpeciesExport("my_export.bp", field=F.Species("H"))
    assert isinstance(my_export.field, list)

    my_export = F.VTXSpeciesExport("my_export.bp", field=[F.Species("H")])
    assert isinstance(my_export.field, list)


@pytest.mark.parametrize("field", [["H", 1], 1, [F.Species("H"), 1]])
def test_field_attribute_raises_error_when_invalid_type(field):
    """Test that the field attribute raises an error if the type is not festim.Species
    or list."""
    with pytest.raises(TypeError):
        F.VTXSpeciesExport("my_export.bp", field=field)


def test_filename_raises_error_when_wrong_type():
    """Test that the filename attribute raises an error if the extension is not .bp."""
    with pytest.raises(TypeError):
        F.VTXSpeciesExport(1, field=[F.Species("H")])


def test_filename_temp_raises_error_when_wrong_type():
    """Test that the filename attribute for VTXTemperature export raises an error if the
    extension is not .bp."""
    with pytest.raises(TypeError):
        F.VTXTemperatureExport(1)


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: x[0] + x[1] * 2,
        lambda T: T + 1,
        lambda c_A, c_B: c_A + c_B,
        lambda c_A, c_B, x: c_A * c_B + x[0],
        lambda c_A, T, x: c_A * T + x[0],
    ],
)
def test_custom_field(tmp_path, expression):
    """
    Test custom field export functionality.
    This test checks that a custom field can be created with various types of
    expressions.
    """

    my_model = F.HydrogenTransportProblem()

    mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

    vol = F.VolumeSubdomain(id=1, material=mat)

    top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
    bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))
    left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))

    my_model.subdomains = [vol, top, bottom, left, right]

    dolfinx_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    my_model.mesh = F.Mesh(dolfinx_mesh)

    A = F.Species("A")
    B = F.Species("B")
    C = F.Species("C")
    D = F.Species("D")

    my_model.species = [A, B, C, D]

    my_model.boundary_conditions = (
        [
            F.FixedConcentrationBC(species=A, subdomain=top, value=1),
            F.FixedConcentrationBC(species=B, subdomain=left, value=1),
        ]
        + [
            F.FixedConcentrationBC(species=C, subdomain=surf, value=0)
            for surf in [top, bottom, left, right]
        ]
        + [
            F.FixedConcentrationBC(species=D, subdomain=surf, value=0)
            for surf in [top, bottom, left, right]
        ]
    )

    my_model.reactions = [
        F.Reaction(
            reactant=[A, B], product=[C], k_0=1, E_k=0, p_0=0, E_p=0, volume=vol
        ),
        F.Reaction(reactant=[C], product=[D], k_0=0.1, E_k=0, p_0=0, E_p=0, volume=vol),
    ]

    my_model.temperature = 300

    my_model.settings = F.Settings(transient=False, atol=1e-9, rtol=1e-9)

    custom_field = F.CustomFieldExport(
        filename=tmp_path / "custom_field.bp",
        expression=expression,
        species_dependent_value={"c_A": A, "c_B": B},
    )

    my_model.exports = [
        custom_field,
    ]

    my_model.initialise()

    my_model.run()


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: x[0] + x[1] * 2,
        lambda T: T + 1,
        lambda c_A, c_B: c_A + c_B,
        lambda c_A, T: c_A * T,
        lambda c_A, c_B, x: c_A * c_B + x[0],
        lambda c_A, T, x: c_A * T + x[0],
        lambda T, x: T + x[0],
    ],
)
def test_custom_field_discontinuous(tmp_path, expression):
    """
    Test custom field export functionality.
    This test checks that a custom field can be created with various types of
    expressions.
    """

    my_model = F.HydrogenTransportProblemDiscontinuous()

    mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

    vol = F.VolumeSubdomain(id=1, material=mat)

    top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
    bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))
    left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))

    my_model.subdomains = [vol, top, bottom, left, right]

    my_model.surface_to_volume = {top: vol, bottom: vol, left: vol, right: vol}

    dolfinx_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    my_model.mesh = F.Mesh(dolfinx_mesh)

    A = F.Species("A", subdomains=my_model.volume_subdomains)
    B = F.Species("B", subdomains=my_model.volume_subdomains)
    C = F.Species("C", subdomains=my_model.volume_subdomains)
    D = F.Species("D", subdomains=my_model.volume_subdomains)

    my_model.species = [A, B, C, D]

    my_model.boundary_conditions = (
        [
            F.FixedConcentrationBC(species=A, subdomain=top, value=1),
            F.FixedConcentrationBC(species=B, subdomain=left, value=1),
        ]
        + [
            F.FixedConcentrationBC(species=C, subdomain=surf, value=0)
            for surf in [top, bottom, left, right]
        ]
        + [
            F.FixedConcentrationBC(species=D, subdomain=surf, value=0)
            for surf in [top, bottom, left, right]
        ]
    )

    my_model.temperature = lambda x: 300 + 100 * x[0]

    my_model.settings = F.Settings(transient=False, atol=1e-9, rtol=1e-9)

    custom_field = F.CustomFieldExport(
        filename=tmp_path / "custom_field.bp",
        expression=expression,
        species_dependent_value={"c_A": A, "c_B": B},
        subdomain=vol,
    )

    my_model.exports = [
        custom_field,
    ]

    my_model.initialise()

    my_model.run()


@pytest.mark.parametrize(
    "expression",
    [lambda c_A, t: c_A * t, lambda t: t],
)
def test_custom_field_not_implemented_error(expression):
    my_model = F.HydrogenTransportProblemDiscontinuous()

    mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

    vol = F.VolumeSubdomain(id=1, material=mat)

    top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
    bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))
    left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))

    my_model.subdomains = [vol, top, bottom, left, right]

    my_model.surface_to_volume = {top: vol, bottom: vol, left: vol, right: vol}

    dolfinx_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    my_model.mesh = F.Mesh(dolfinx_mesh)

    A = F.Species("A", subdomains=my_model.volume_subdomains)
    B = F.Species("B", subdomains=my_model.volume_subdomains)
    C = F.Species("C", subdomains=my_model.volume_subdomains)
    D = F.Species("D", subdomains=my_model.volume_subdomains)

    my_model.species = [A, B, C, D]

    my_model.boundary_conditions = (
        [
            F.FixedConcentrationBC(species=A, subdomain=top, value=1),
            F.FixedConcentrationBC(species=B, subdomain=left, value=1),
        ]
        + [
            F.FixedConcentrationBC(species=C, subdomain=surf, value=0)
            for surf in [top, bottom, left, right]
        ]
        + [
            F.FixedConcentrationBC(species=D, subdomain=surf, value=0)
            for surf in [top, bottom, left, right]
        ]
    )

    my_model.temperature = lambda x: 300 + 100 * x[0]

    my_model.settings = F.Settings(transient=False, atol=1e-9, rtol=1e-9)

    custom_field = F.CustomFieldExport(
        filename="custom_field.bp",
        expression=expression,
        species_dependent_value={"c_A": A, "c_B": B},
        subdomain=vol,
    )

    my_model.exports = [
        custom_field,
    ]

    with pytest.raises(NotImplementedError):
        my_model.initialise()


@pytest.mark.parametrize("direction", ["both", "forward", "backward"])
@pytest.mark.parametrize("product_type", ["list", "single"])
@pytest.mark.parametrize("p_0, E_p", [(0.01, 0.05), (0.01, 0.0), (0.0, 0.0)])
def test_reaction_rate_export(tmp_path, direction, product_type, p_0, E_p):
    """
    Test ReactionRateExport export functionality for different directions, product formats,
    and reaction configurations.
    """
    if p_0 == 0.0 and direction == "backward":
        pytest.skip(
            "Backward direction export not supported when backward reaction is disabled"
        )
    my_model = F.HydrogenTransportProblem()
    mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    vol = F.VolumeSubdomain(id=1, material=mat)
    top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
    bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))
    left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
    right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1))

    my_model.subdomains = [vol, top, bottom, left, right]

    dolfinx_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    my_model.mesh = F.Mesh(dolfinx_mesh)

    A = F.Species("A")
    B = F.Species("B")
    C = F.Species("C")

    my_model.species = [A, B, C]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(species=A, subdomain=top, value=1),
        F.FixedConcentrationBC(species=B, subdomain=left, value=1),
        F.FixedConcentrationBC(species=C, subdomain=bottom, value=0),
    ]

    reaction = F.Reaction(
        reactant=[A, B],
        product=[C] if product_type == "list" else C,
        k_0=1,
        E_k=0.1,
        p_0=p_0,
        E_p=E_p,
        volume=vol,
    )

    my_model.reactions = [reaction]

    my_model.temperature = 300

    my_model.settings = F.Settings(transient=False, atol=1e-9, rtol=1e-9)

    reaction_rate_export = F.ReactionRateExport(
        filename=tmp_path / f"reaction_rate_{direction}.bp",
        reaction=reaction,
        direction=direction,
    )

    my_model.exports = [reaction_rate_export]

    my_model.initialise()
    my_model.run()


def test_reaction_rate_override_signature():
    """
    Test that ReactionRateExport signature override correctly updates signatures.
    """
    mat = F.Material(D_0=1, E_D=0)
    vol = F.VolumeSubdomain(id=1, material=mat)
    A = F.Species("A")
    B = F.Species("B")
    reaction = F.Reaction(
        reactant=[A], product=[B], k_0=1, E_k=0, p_0=0, E_p=0, volume=vol
    )

    rr = F.ReactionRateExport(reaction=reaction, filename="dummy.bp")

    def my_expression(**kwargs):
        return kwargs.get("x", 0) + kwargs.get("y", 0)

    rr.override_signature(my_expression, ["A"], ["B"])
    import inspect

    sig = inspect.signature(my_expression)
    assert set(sig.parameters.keys()) == {"T", "A", "B"}


def test_export_base_class_times_and_extension(tmp_path):
    """
    Test that ExportBaseClass sorts times and warns when wrong extension is given.
    """
    with pytest.warns(UserWarning, match="does not have .bp extension"):
        export = F.ExportBaseClass(
            filename=tmp_path / "wrong_extension.txt", ext=".bp", times=[3.0, 1.0, 2.0]
        )

    assert export.filename.suffix == ".bp"
    assert export.times == [1.0, 2.0, 3.0]


def test_export_base_class_no_times(tmp_path):
    export = F.ExportBaseClass(filename=tmp_path / "correct.bp", ext=".bp", times=None)
    assert export.times is None

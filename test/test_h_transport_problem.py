import festim as F
import tqdm.autonotebook
import mpi4py.MPI as MPI
import dolfinx.mesh
from dolfinx import fem, nls
import ufl
import numpy as np
import pytest

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)
dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")


# TODO test all the methods in the class
@pytest.mark.parametrize(
    "value", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou", lambda x: 2 * x[0]]
)
def test_temperature_setter_type(value):
    """Test that the temperature type is correctly set"""
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
    )

    if not isinstance(value, (fem.Constant, int, float)):
        if callable(value):
            my_model.temperature = value
        else:
            with pytest.raises(TypeError):
                my_model.temperature = value


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (1, False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), True),
    ],
)
def test_time_dependent_temperature_attribute(input, expected_value):
    """Test that the temperature_time_dependent attribute is correctly set"""

    my_model = F.HydrogenTransportProblem()
    my_model.temperature = input

    assert my_model.temperature_time_dependent is expected_value


def test_define_temperature_value_error_raised():
    """Test that a ValueError is raised when the temperature is None"""

    # BUILD
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)

    my_model.temperature = None

    # TEST
    with pytest.raises(
        ValueError, match="the temperature attribute needs to be defined"
    ):
        my_model.define_temperature()


@pytest.mark.parametrize(
    "input, expected_type",
    [
        (1.0, fem.Constant),
        (1, fem.Constant),
        (fem.Constant(test_mesh.mesh, 1.0), fem.Constant),
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), fem.Function),
        (lambda t: 100.0 if t < 1 else 0.0, fem.Constant),
    ],
)
def test_define_temperature(input, expected_type):
    """Test that the define_temperature method correctly sets the
    temperature_fenics attribute to either a fem.Constant or a
    fem.Function depending on the type of input"""

    # BUILD
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)
    my_model.t = fem.Constant(test_mesh.mesh, 0.0)

    my_model.temperature = input

    # RUN
    my_model.define_temperature()

    # TEST
    assert isinstance(my_model.temperature_fenics, expected_type)


@pytest.mark.parametrize(
    "input",
    [
        lambda t: ufl.conditional(ufl.lt(t, 1.0), 1, 2),
        lambda t: 1 + ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
        lambda t: 2 * ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
        lambda t: 2 / ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
    ],
)
def test_define_temperature_error_if_ufl_conditional_t_only(input):
    """Test that a ValueError is raised when the temperature attribute is a callable
    of t only and contains a ufl conditional"""
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)
    my_model.t = fem.Constant(test_mesh.mesh, 0.0)

    my_model.temperature = input

    with pytest.raises(
        ValueError, match="self.temperature should return a float or an int, not "
    ):
        my_model.define_temperature()


def test_iterate():
    """Test that the iterate method updates the solution and time correctly"""
    # BUILD
    my_model = F.HydrogenTransportProblem()

    my_model.settings = F.Settings(atol=1e-6, rtol=1e-6, final_time=10)

    my_model.progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    V = fem.FunctionSpace(mesh, ("CG", 1))
    my_model.u = fem.Function(V)
    my_model.u_n = fem.Function(V)
    my_model.dt = fem.Constant(mesh, 2.0)
    v = ufl.TestFunction(V)

    source_value = 2.0
    form = (
        my_model.u - my_model.u_n
    ) / my_model.dt * v * ufl.dx - source_value * v * ufl.dx

    problem = fem.petsc.NonlinearProblem(form, my_model.u, bcs=[])
    my_model.solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    my_model.t = fem.Constant(mesh, 0.0)

    for i in range(10):
        # RUN
        my_model.iterate()

        # TEST

        # check that t evolves
        expected_t_value = (i + 1) * float(my_model.dt)
        assert np.isclose(float(my_model.t), expected_t_value)

        # check that u and u_n are updated
        expected_u_value = (i + 1) * float(my_model.dt) * source_value
        assert np.all(np.isclose(my_model.u.x.array, expected_u_value))


@pytest.mark.parametrize(
    "T_function, expected_values",
    [
        (lambda t: t, [1.0, 2.0, 3.0]),
        (lambda t: 1.0 + t, [2.0, 3.0, 4.0]),
        (lambda x, t: 1.0 + x[0] + t, [6.0, 7.0, 8.0]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.5), 100.0 + x[0], 0.0),
            [104.0, 0.0, 0.0],
        ),
    ],
)
def test_update_time_dependent_values_temperature(T_function, expected_values):
    """Test that different time-dependent callable functions for the
    temperature are updated at each time step and match an expected value"""

    # BUILD
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_model.temperature = T_function

    my_model.define_temperature()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.temperature_fenics, fem.Constant):
            computed_value = float(my_model.temperature_fenics)
            print(computed_value)
        else:
            computed_value = my_model.temperature_fenics.vector.array[-1]
            print(computed_value)
        assert np.isclose(computed_value, expected_values[i])


def test_initialise_exports_find_species_with_one_field():
    """Test that a species can be found from the model species if given as a string"""

    # BUILD
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0])),
        species=[F.Species("H"), F.Species("D")],
        temperature=1,
    )
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    my_model.exports = [F.SurfaceFlux(field="J", surface=surf)]
    my_model.define_function_spaces()

    # TEST
    with pytest.raises(ValueError, match="Species J not found in list of species"):
        my_model.initialise_exports()


def test_define_D_global_different_temperatures():
    """Test that the D_global object is correctly defined when the temperature
    is different in the volume subdomains"""
    D_0, E_D = 1.5, 0.1
    my_mat = F.Material(D_0=D_0, E_D=E_D, name="my_mat")
    surf = F.SurfaceSubdomain1D(id=1, x=0)
    H = F.Species("H")

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 4, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 2], material=my_mat),
            F.VolumeSubdomain1D(id=2, borders=[2, 4], material=my_mat),
        ],
        species=[H],
        temperature=lambda x: 100.0 * x[0] + 50,
    )

    my_model.define_function_spaces()
    my_model.define_markers_and_measures()
    my_model.define_temperature()

    D_computed, D_expr = my_model.define_D_global(H)

    computed_values = [D_computed.x.array[0], D_computed.x.array[-1]]

    D_analytical_left = D_0 * np.exp(-E_D / (F.k_B * 50))
    D_analytical_right = D_0 * np.exp(-E_D / (F.k_B * 450))

    expected_values = [D_analytical_left, D_analytical_right]

    assert np.isclose(computed_values, expected_values).all()


def test_define_D_global_different_materials():
    """Test that the D_global object is correctly defined when the material
    is different in the volume subdomains"""
    D_0_left, E_D_left = 1.0, 0.1
    D_0_right, E_D_right = 2.0, 0.2
    my_mat_L = F.Material(D_0=D_0_left, E_D=E_D_left, name="my_mat_L")
    my_mat_R = F.Material(D_0=D_0_right, E_D=E_D_right, name="my_mat_R")
    H = F.Species("H")

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 4, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 2], material=my_mat_L),
            F.VolumeSubdomain1D(id=2, borders=[2, 4], material=my_mat_R),
        ],
        species=[H],
        temperature=500,
    )

    my_model.define_function_spaces()
    my_model.define_markers_and_measures()
    my_model.define_temperature()

    D_computed, D_expr = my_model.define_D_global(H)

    computed_values = [D_computed.x.array[0], D_computed.x.array[-1]]

    D_expected_left = D_0_left * np.exp(-E_D_left / (F.k_B * my_model.temperature))
    D_expected_right = D_0_right * np.exp(-E_D_right / (F.k_B * my_model.temperature))

    expected_values = [D_expected_left, D_expected_right]

    assert np.isclose(computed_values, expected_values).all()


def test_initialise_exports_multiple_exports_same_species():
    """Test that the diffusion coefficient within the D_global object function is the same
    for multiple exports of the same species, and that D_global object is only
    created once per species"""

    D_0, E_D = 1.5, 0.1
    my_mat = F.Material(D_0=D_0, E_D=E_D, name="my_mat")
    surf_1 = F.SurfaceSubdomain1D(id=1, x=0)
    surf_2 = F.SurfaceSubdomain1D(id=1, x=4)
    H = F.Species("H")

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 4, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 2], material=my_mat),
            F.VolumeSubdomain1D(id=2, borders=[2, 4], material=my_mat),
        ],
        species=[H],
        temperature=500,
        exports=[
            F.SurfaceFlux(
                field=H,
                surface=surf_1,
            ),
            F.SurfaceFlux(
                field=H,
                surface=surf_2,
            ),
        ],
    )

    my_model.define_function_spaces()
    my_model.define_markers_and_measures()
    my_model.define_temperature()
    my_model.initialise_exports()

    Ds = [export.D for export in my_model.exports]

    assert Ds[0].x.array[0] == Ds[1].x.array[0]


def test_define_D_global_multispecies():
    """Test that the D_global object is correctly defined when there are multiple
    species in one subdomain"""
    A = F.Species("A")
    B = F.Species("B")

    D_0_A, D_0_B = 1.0, 2.0
    E_D_A, E_D_B = 0.1, 0.2

    my_mat = F.Material(
        D_0={A: D_0_A, B: D_0_B}, E_D={A: E_D_A, B: E_D_B}, name="my_mat"
    )
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 1, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat),
        ],
        species=[F.Species("A"), F.Species("B")],
        temperature=500,
    )

    my_model.define_function_spaces()
    my_model.define_markers_and_measures()
    my_model.define_temperature()

    D_A_computed, D_A_expr = my_model.define_D_global(A)
    D_B_computed, D_B_expr = my_model.define_D_global(B)

    computed_values = [D_A_computed.x.array[-1], D_B_computed.x.array[-1]]

    D_expected_A = D_0_A * np.exp(-E_D_A / (F.k_B * my_model.temperature))
    D_expected_B = D_0_B * np.exp(-E_D_B / (F.k_B * my_model.temperature))

    expected_values = [D_expected_A, D_expected_B]

    assert np.isclose(computed_values, expected_values).all()


def test_post_processing_update_D_global():
    """Test that the D_global object is updated at each time
    step when temperture is time dependent"""
    my_mesh = F.Mesh1D(np.linspace(0, 1, num=11))
    my_mat = F.Material(D_0=1.5, E_D=0.1, name="my_mat")
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    # create species and interpolate solution
    H = F.Species("H")
    V = fem.FunctionSpace(my_mesh.mesh, ("CG", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: 2 * x[0] ** 2 + 1)
    H.solution = u

    my_export = F.SurfaceFlux(
        field=H,
        surface=surf,
    )

    # Build the model
    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh,
        subdomains=[F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat), surf],
        species=[H],
        temperature=lambda t: 500 * t,
        exports=[my_export],
    )

    my_model.define_function_spaces()
    my_model.define_markers_and_measures()
    my_model.t = fem.Constant(my_model.mesh.mesh, 1.0)
    my_model.define_temperature()
    my_model.initialise_exports()

    # RUN
    my_model.post_processing()
    value_t_1 = my_export.D.x.array[-1]

    my_model.t = fem.Constant(my_model.mesh.mesh, 2.0)
    my_model.update_time_dependent_values()
    my_model.post_processing()
    value_t_2 = my_export.D.x.array[-1]

    # TEST
    assert value_t_1 != value_t_2


def test_all_species_correct_type():
    """Test that the D_global object is updated at each time
    step when temperture is time dependent"""

    my_model = F.HydrogenTransportProblem()
    C = 1
    my_species = [F.Species("A"), F.Species("B"), C]

    with pytest.raises(
        TypeError,
        match="elements of species must be of type festim.Species not <class 'int'>",
    ):
        my_model.species = my_species


def test_create_formulation_with_reactions():
    A, B, C = F.Species("A"), F.Species("B"), F.Species("C")
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 1, num=11)),
        temperature=500,
        species=[A, B, C],
        subdomains=[F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)],
    )

    my_reaction = F.Reaction(
        reactant1=A,
        reactant2=B,
        product=C,
        k_0=2.0,
        E_k=0.2,
        p_0=1.0,
        E_p=0.1,
    )
    my_model.reactions = [my_reaction]

    my_model.define_function_spaces()
    my_model.define_markers_and_measures()
    my_model.assign_functions_to_species()
    my_model.t = fem.Constant(my_model.mesh.mesh, 1.0)
    my_model.dt = fem.Constant(my_model.mesh.mesh, 0.1)
    my_model.define_temperature()
    my_model.create_formulation()

    # remove all white spaces between characters
    computed_formulation = str(my_model.formulation).translate(
        str.maketrans("", "", " \n\t\r")
    )

    expected_formulation = """{ ({ A | A_{i_8} = (grad(f[0]))[i_8] * c_47 * exp(-1 * c_48 / 8.6173303e-05 / c_46) }) . (grad(v_0[0])) } * dx(<Mesh #8>[1], {})
        +  { v_0[0] * (f[0] + -1 * f[0]) / c_45 } * dx(<Mesh #8>[1], {})
        +  { ({ A | A_{i_9} = (grad(f[1]))[i_9] * c_49 * exp(-1 * c_50 / 8.6173303e-05 / c_46) }) . (grad(v_0[1])) } * dx(<Mesh #8>[1], {})
        +  { v_0[1] * (f[1] + -1 * f[1]) / c_45 } * dx(<Mesh #8>[1], {})
        +  { ({ A | A_{i_{10}} = (grad(f[2]))[i_{10}] * c_51 * exp(-1 * c_52 / 8.6173303e-05 / c_46) }) . (grad(v_0[2])) } * dx(<Mesh #8>[1], {})
        +  { v_0[2] * (f[2] + -1 * f[2]) / c_45 } * dx(<Mesh #8>[1], {})
        +  { v_0[0] * (f[1] * f[0] * 2.0 * exp(-0.2 / 8.6173303e-05 * c_46) + -1 * f[2] * exp(-0.1 / 8.6173303e-05 * c_46)) } * dx(<Mesh #8>[everywhere], {})
        +  { v_0[1] * (f[1] * f[0] * 2.0 * exp(-0.2 / 8.6173303e-05 * c_46) + -1 * f[2] * exp(-0.1 / 8.6173303e-05 * c_46)) } * dx(<Mesh #8>[everywhere], {})
        +  { v_0[2] * -1 * (f[1] * f[0] * 2.0 * exp(-0.2 / 8.6173303e-05 * c_46) + -1 * f[2] * exp(-0.1 / 8.6173303e-05 * c_46)) } * dx(<Mesh #8>[everywhere], {})
        """
    # remove all white spaces between characters
    expected_formulation = expected_formulation.translate(
        str.maketrans("", "", " \n\t\r")
    )

    assert computed_formulation == expected_formulation

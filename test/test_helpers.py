from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem

import festim as F

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)

test_function_space = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
test_function = fem.Function(test_function_space)


@pytest.mark.parametrize(
    "value",
    [
        1,
        fem.Constant(test_mesh.mesh, default_scalar_type(1.0)),
        1.0,
        "coucou",
        2 * x[0],
    ],
)
def test_temperature_type_and_processing(value):
    """Test that the temperature type is correctly set."""

    if not isinstance(value, fem.Constant | int | float):
        with pytest.raises(TypeError):
            F.as_fenics_constant(value, test_mesh.mesh)
    else:
        assert isinstance(F.as_fenics_constant(value, test_mesh.mesh), fem.Constant)


@pytest.mark.parametrize(
    "input_value, expected_output_type", [(1.0, fem.Constant), (3, fem.Constant)]
)
def test_value_convert_float_int_inputs(input_value, expected_output_type):
    """Test that float and  value is correctly converted."""

    test_value = F.Value(input_value)

    test_value.convert_input_value(function_space=test_function_space)

    assert isinstance(test_value.fenics_object, expected_output_type)


@pytest.mark.parametrize(
    "input_value, expected_output_type",
    [
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], ufl.core.expr.Expr),
        (lambda x, t: 1.0 + x[0] + t, ufl.core.expr.Expr),
        (lambda x, t, T: 1.0 + x[0] + t + T, ufl.core.expr.Expr),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            ufl.core.expr.Expr,
        ),
    ],
)
def test_value_convert_up_to_ufl_inputs(input_value, expected_output_type):
    """Test that float and  value is correctly converted."""

    my_mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = fem.functionspace(my_mesh, ("Lagrange", 1))
    my_t = fem.Constant(my_mesh, default_scalar_type(10))
    my_T = fem.Constant(my_mesh, default_scalar_type(3))

    test_value = F.Value(input_value)

    test_value.convert_input_value(
        function_space=V,
        t=my_t,
        temperature=my_T,
        up_to_ufl_expr=True,
    )

    assert isinstance(test_value.fenics_object, expected_output_type)


@pytest.mark.parametrize(
    "input_value, expected_output_type",
    [
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (lambda x, t, T: 1.0 + x[0] + t + T, fem.Function),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            ufl.core.expr.Expr,
        ),
    ],
)
def test_value_convert_callable_inputs(input_value, expected_output_type):
    """Test that float and  value is correctly converted."""

    my_mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 12)
    my_t = fem.Constant(my_mesh, default_scalar_type(8))
    my_T = fem.Constant(my_mesh, default_scalar_type(5))

    my_function_space = fem.functionspace(my_mesh, ("Lagrange", 1))

    test_value = F.Value(input_value)

    test_value.convert_input_value(
        function_space=my_function_space,
        t=my_t,
        temperature=my_T,
    )

    assert isinstance(test_value.fenics_object, expected_output_type)


def test_error_raised_wehn_input_value_is_not_accepted():
    """Test that an error is raised when the input value is not accepted."""

    with pytest.raises(
        TypeError,
        match=(
            r"Value must be a float, int, fem.Constant, fem.Expression, "
            r"ufl.core.expr.Expr, fem.Function, or callable not coucou"
        ),
    ):
        F.Value("coucou")


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(test_mesh.mesh, default_scalar_type(1.0)), False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            True,
        ),
    ],
)
def test_time_dependent_values(input_value, expected_output):
    """Test that the time_dependent attribute is correctly set."""

    test_value = F.Value(input_value)

    assert test_value.explicit_time_dependent == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(test_mesh.mesh, default_scalar_type(1.0)), False),
        (lambda t: t, False),
        (lambda T: 1.0 + T, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, False),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (
            lambda T, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + T[0], 0.0),
            True,
        ),
    ],
)
def test_temperature_dependent_values(input_value, expected_output):
    """Test that the time_dependent attribute is correctly set."""

    test_value = F.Value(input_value)

    assert test_value.temperature_dependent == expected_output


@pytest.mark.parametrize(
    "value",
    [
        fem.Constant(test_mesh.mesh, default_scalar_type(1.0)),
        test_function,
    ],
)
def test_input_values_of_constants_and_functions_are_accepted(value):
    """Test that the input values of constants and functions are accepted."""

    test_value = F.Value(value)

    test_value.convert_input_value()

    assert test_value.fenics_object == value


def test_input_values_of_expressions_are_accepted():
    """Test that the input values of constants and functions are accepted."""

    def my_func(x):
        return 1.0 + x[0]

    kwargs = {}
    kwargs["x"] = x
    mapped_func = my_func(**kwargs)

    test_expression = fem.Expression(
        mapped_func,
        test_function_space.element.interpolation_points,
    )
    test_value = F.Value(input_value=test_expression)

    test_value.convert_input_value()

    assert test_value.fenics_interpolation_expression == test_expression


def test_ValueError_raised_when_callable_returns_wrong_type():
    """The create_value_fenics method should raise a ValueError when the callable
    returns an object which is not a float or int."""

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    test_value = F.Value(my_value)

    T = fem.Constant(test_mesh.mesh, 550.0)
    t = fem.Constant(test_mesh.mesh, 0.0)

    with pytest.raises(
        ValueError,
        match=(
            r"self.value should return a float or an int, not <class "
            r"'ufl.conditional.Conditional'"
        ),
    ):
        test_value.convert_input_value(
            function_space=test_function_space, temperature=T, t=t
        )


@pytest.mark.parametrize(
    "value",
    [
        1,
        1.0,
        lambda t: t,
        lambda T: 1.0 + T,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t, T: 1.0 + x[0] + t + T,
        lambda T, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + T[0], 0.0),
    ],
)
def test_value_representation(value):
    """Test that the representation of the value is correct."""

    test_value = F.Value(value)

    assert repr(test_value) == f"{value}"


@pytest.mark.parametrize(
    "value",
    [
        lambda T: 1.0 + T,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t, T: 1.0 + x[0] + t + T,
        lambda T, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + T[0], 0.0),
    ],
)
def test_velocity_field_convert_input_error_when_t_not_only_arg(value):
    """Test when an input value of type callable is converted that a Type error is
    rasied when t is not the only arg."""

    test_value = F.VelocityField(value)
    t = F.as_fenics_constant(value=1.0, mesh=test_mesh.mesh)

    with pytest.raises(
        TypeError, match="velocity function can only be a function of time arg t"
    ):
        test_value.convert_input_value(function_space=test_function_space, t=t)


def test_velocity_field_convert_input_error_when_callable_doesnt_return_fem_func():
    """Test when an input value of type callable is converted that a Type error is
    rasied when t is not the only arg."""

    def example_func(t):
        return 2 * t

    test_value = F.VelocityField(input_value=lambda t: example_func(t))
    t = F.as_fenics_constant(value=1.0, mesh=test_mesh.mesh)

    with pytest.raises(
        ValueError,
        match=(
            r"A time dependent advection field should return an fem.Function, not a "
            r"<class 'ufl.algebra.Product'>"
        ),
    ):
        test_value.convert_input_value(function_space=test_function_space, t=t)


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (2, True),
        (3, True),
        (1, True),
        (-1, False),
        (0, False),
        (5, False),
        (1.5, False),
    ],
)
def test_is_it_time_to_export(input, expected_output):
    times = [1, 2, 3]
    assert (
        F.helpers.is_it_time_to_export(current_time=input, times=times)
        == expected_output
    )


def test_is_it_time_to_export_when_times_not_given():
    times = None
    assert F.helpers.is_it_time_to_export(current_time=1.0, times=times)


@pytest.mark.parametrize(
    "input_dict, expected",
    [
        (None, {}),
        ({}, {}),
        ({"c1": F.Species("c1")}, None),  # None here means "same as input_dict"
    ],
)
def test_value_stores_species_dependent_value(input_dict, expected):
    """Test that the species_dependent_value passed to Value is stored, and that a
    None input is normalised to an empty dict."""

    test_value = F.Value(1.0, species_dependent_value=input_dict)

    assert test_value.species_dependent_value == (
        input_dict if expected is None else expected
    )


def test_value_species_dependent_value_defaults_to_empty_dict():
    """Test that species_dependent_value defaults to an empty dict when not given."""

    test_value = F.Value(1.0)

    assert test_value.species_dependent_value == {}


def test_as_mapped_function_uses_species_concentration():
    """Test that as_mapped_function injects the species concentration (continuous case,
    where species.concentration is set) as the matching callable argument."""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    species = F.Species("c1")
    species.solution = fem.Function(V)  # concentration is not None
    species.solution.x.array[:] = 3.0

    mapped = F.as_mapped_function(
        value=lambda c1: 2 * c1,
        function_space=V,
        species_dependent_value={"c1": species},
    )

    # the mapped expression should evaluate to 2 * concentration = 6.0 everywhere
    result = fem.Function(V)
    result.interpolate(fem.Expression(mapped, V.element.interpolation_points))
    assert np.allclose(result.x.array, 6.0)


def test_as_mapped_function_uses_subdomain_solution_when_concentration_is_none():
    """Test that as_mapped_function falls back to
    species.subdomain_to_solution[subdomain] (discontinuous case, where
    species.concentration is None) for the matching argument."""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    subdomain = F.VolumeSubdomain1D(
        id=1, borders=[0, 1], material=F.Material(D_0=1, E_D=1)
    )

    species = F.Species("c1")  # solution is None -> concentration is None
    sub_solution = fem.Function(V)
    sub_solution.x.array[:] = 4.0
    species.subdomain_to_solution = {subdomain: sub_solution}

    mapped = F.as_mapped_function(
        value=lambda c1: 2 * c1,
        function_space=V,
        species_dependent_value={"c1": species},
        subdomain=subdomain,
    )

    result = fem.Function(V)
    result.interpolate(fem.Expression(mapped, V.element.interpolation_points))
    assert np.allclose(result.x.array, 8.0)


def test_as_mapped_function_ignores_species_the_callable_does_not_declare():
    """Test that a species_dependent_value entry whose name is not an argument of
    the callable is ignored (as done for t/x/T), rather than passed and raising."""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    used = F.Species("c1")
    used.solution = fem.Function(V)
    used.solution.x.array[:] = 3.0
    unused = F.Species("c2")  # not an argument of the callable below

    # the callable only declares c1; c2 must be silently dropped
    mapped = F.as_mapped_function(
        value=lambda c1: 2 * c1,
        function_space=V,
        species_dependent_value={"c1": used, "c2": unused},
    )

    result = fem.Function(V)
    result.interpolate(fem.Expression(mapped, V.element.interpolation_points))
    assert np.allclose(result.x.array, 6.0)


def test_convert_input_value_species_dependent_up_to_ufl_expr():
    """Test that convert_input_value threads species_dependent_value through to the
    mapped ufl expression when up_to_ufl_expr is True."""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    species = F.Species("c1")
    species.solution = fem.Function(V)
    species.solution.x.array[:] = 5.0

    test_value = F.Value(lambda c1: 2 * c1, species_dependent_value={"c1": species})
    test_value.convert_input_value(function_space=V, up_to_ufl_expr=True)

    result = fem.Function(V)
    result.interpolate(
        fem.Expression(test_value.fenics_object, V.element.interpolation_points)
    )
    assert np.allclose(result.x.array, 10.0)


def test_convert_input_value_species_dependent_interpolated_function():
    """Test that convert_input_value uses species_dependent_value in the interpolation
    path (callable of x and a species) and produces the expected fenics Function."""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    species = F.Species("c1")
    species.solution = fem.Function(V)
    species.solution.x.array[:] = 2.0

    test_value = F.Value(
        lambda x, c1: c1 + x[0], species_dependent_value={"c1": species}
    )
    test_value.convert_input_value(function_space=V)

    assert isinstance(test_value.fenics_object, fem.Function)
    # fenics_object should equal concentration + x = 2.0 + x
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    assert np.allclose(test_value.fenics_object.x.array, 2.0 + x_coords)


@pytest.mark.parametrize(
    "input_value, species_dependent_value, expected",
    [
        (lambda c1: 2 * c1, {"c1": F.Species("c1")}, True),
        (lambda t: 2 * t, {}, False),
        (lambda t: 2 * t, None, False),
        # a non-callable value cannot depend on a species
        (1.0, {"c1": F.Species("c1")}, False),
    ],
)
def test_value_species_dependent(input_value, species_dependent_value, expected):
    """Test the species_dependent property of Value."""

    test_value = F.Value(input_value, species_dependent_value=species_dependent_value)

    assert test_value.species_dependent is expected


def test_convert_input_value_time_and_species_dependent():
    """Regression test: a value depending on both time and a species must not take the
    t-only constant fast path (which would call the callable without the species
    argument). It should map to a UFL expression that tracks both the time and the
    species concentration as they change."""

    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    species = F.Species("B")
    species.solution = fem.Function(V)  # continuous case: concentration is set
    species.solution.x.array[:] = 2.0

    t = fem.Constant(test_mesh.mesh, 3.0)

    test_value = F.Value(lambda t, B: t * B, species_dependent_value={"B": species})
    # up_to_ufl_expr=True is the path used for sources
    test_value.convert_input_value(function_space=V, t=t, up_to_ufl_expr=True)

    # the result is a UFL expression; evaluate it by interpolating into V
    result = fem.Function(V)
    expr = fem.Expression(test_value.fenics_object, V.element.interpolation_points)
    result.interpolate(expr)
    assert np.allclose(result.x.array, 3.0 * 2.0)  # t * B

    # the expression tracks later changes to both the time and the concentration
    t.value = 5.0
    species.solution.x.array[:] = 4.0
    result.interpolate(expr)
    assert np.allclose(result.x.array, 5.0 * 4.0)

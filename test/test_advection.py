from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pytest
from dolfinx import fem

import festim as F

test_mesh_1d = F.Mesh1D(vertices=np.linspace(0, 1, 100))
test_functionspace = fem.functionspace(test_mesh_1d.mesh, ("CG", 1))


@pytest.mark.parametrize(
    "value", ["coucou", 1.0, 1, fem.Constant(test_mesh_1d.mesh, 1.0)]
)
def test_Typeerror_raised_when_wrong_object_given_to_Advection(value):
    "test"

    my_species = F.Species("H")
    my_subdomain = F.VolumeSubdomain(id=1, material="dummy_mat")

    with pytest.raises(
        TypeError,
        match=f"velocity must be a fem.Function, or callable not {type(value)}",
    ):
        F.AdvectionTerm(velocity=value, subdomain=my_subdomain, species=my_species)


@pytest.mark.parametrize(
    "value",
    ["coucou", 1.0, 1, F.SurfaceSubdomain(id=1)],
)
def test_subdomain_setter(value):
    "test"

    my_species = F.Species("H")

    with pytest.raises(
        TypeError,
        match=f"Subdomain must be a festim.Subdomain object, not {type(value)}",
    ):
        F.AdvectionTerm(velocity=None, subdomain=value, species=my_species)


@pytest.mark.parametrize(
    "value",
    ["coucou", 1.0, 1, F.SurfaceSubdomain(id=1)],
)
def test_species_setter_type_error(value):
    "test"

    my_subdomain = F.VolumeSubdomain(id=1, material="dummy_mat")

    with pytest.raises(
        TypeError,
        match=f"elements of species must be of type festim.Species not {type(value)}",
    ):
        F.AdvectionTerm(velocity=None, subdomain=my_subdomain, species=value)


@pytest.mark.parametrize(
    "value",
    [F.Species("H"), [F.Species("D"), F.Species("test")]],
)
def test_species_setter_changes_input_to_list(value):
    "test"

    my_subdomain = F.VolumeSubdomain(id=1, material="dummy_mat")

    my_advection_term = F.AdvectionTerm(
        velocity=None, subdomain=my_subdomain, species=value
    )

    assert isinstance(my_advection_term.species, list)


@pytest.mark.parametrize(
    "value",
    [lambda x: x[0], lambda t: t[0], lambda x, t: x[0] + 2 * t],
)
def test_velocity_accepts_callable_values(value):
    my_subdomain = F.VolumeSubdomain(id=1, material="dummy_mat")

    F.AdvectionTerm(velocity=value, subdomain=my_subdomain, species=F.Species("H"))


def test_subdomain_accepts_None_value():
    F.AdvectionTerm(velocity=None, subdomain=None, species=F.Species("H"))


def test_velocity_field_update():
    """Test a explicit time dependent input value is updated correctly"""

    test_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(test_mesh, ("Lagrange", 1))

    v_cg = basix.ufl.element(
        "Lagrange", test_mesh.topology.cell_name(), 2, shape=(test_mesh.geometry.dim,)
    )
    V_velocity = fem.functionspace(test_mesh, v_cg)
    u = fem.Function(V_velocity)

    def velocity_func_alt(x, t):
        values = np.zeros((2, x.shape[1]))  # Initialize with zeros

        scalar_value = 2 * t  # Compute the scalar function
        values[0] = scalar_value  # Assign to first component
        values[1] = scalar_value  # Second component remains zero

        return values

    def example_func(t):
        u.interpolate(lambda x: velocity_func_alt(x, t))
        return u

    my_subdomain = F.VolumeSubdomain(id=1, material="dummy_mat")
    my_species = F.Species("H")
    test_value = F.AdvectionTerm(
        velocity=lambda t: example_func(t), subdomain=my_subdomain, species=my_species
    )
    t = F.as_fenics_constant(value=1.0, mesh=test_mesh)

    test_value.velocity.convert_input_value(function_space=V, t=t)

    t_values = [1, 2, 3, 4, 5]

    for t in t_values:
        test_value.velocity.update(t)
        assert np.isclose(np.max(test_value.velocity.fenics_object.x.array), 2 * t)

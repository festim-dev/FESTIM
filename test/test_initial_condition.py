import numpy as np
import festim as F
import ufl
import pytest
import ufl
from dolfinx import fem

dummy_mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
test_mesh = F.Mesh1D(np.linspace(0, 1, 100))


def test_init():
    """Test that the attributes are set correctly"""
    # create a Source object
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)
    value = 1.0
    species = F.Species("test")
    init_cond = F.InitialCondition(volume=volume, value=value, species=species)

    # check that the attributes are set correctly
    assert init_cond.volume == volume
    assert init_cond.value == value
    assert init_cond.species == species


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        1,
        lambda T: 1.0 + T,
        lambda x: 1.0 + x[0],
        lambda x, T: 1.0 + x[0] + T,
    ],
)
def test_create_initial_condition(value):
    """Test that the create initial conditions method produces either a fem.Constant or
    fem.Function depending on the value input"""

    # BUILD
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    # give function to species
    V = fem.FunctionSpace(test_mesh.mesh, ("CG", 1))
    c = fem.Function(V)
    # c.interpolate(lambda x: 2 * x[0] ** 2 + 1)

    my_species = F.Species("test")
    my_species.solution = c

    init_cond = F.InitialCondition(
        volume=vol_subdomain, value=value, species=my_species
    )

    T = fem.Constant(test_mesh.mesh, 10.0)

    # RUN
    init_cond.create_initial_condition(test_mesh.mesh, T)

    # TEST
    print(init_cond.species.solution[0])
    quit()
    assert isinstance(init_cond.species.solution.all(), value.all())

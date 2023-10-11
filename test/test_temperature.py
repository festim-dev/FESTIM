import festim as F
from dolfinx import fem
import numpy as np
import sympy as sp
from pytest import raises


def test_temperature_value():
    # Test that the temperature value is correctly set
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.array([0, 1, 3, 4]))
    my_model.species = [F.Species("H")]
    my_mat = F.Material(1, 1, "1")
    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=my_mat)
    my_model.subdomains = [my_subdomain]

    my_model.temperature = fem.Constant(my_model.mesh.mesh, 23.0)
    my_model.initialise()

    assert float(my_model.temperature) == 23.0


def test_temperature_type():
    # Test that the temperature type is correctly set
    my_mesh = F.Mesh1D(vertices=np.array([0, 1, 3, 4]))
    my_mesh.generate_mesh()
    values = [int(1), fem.Constant(my_mesh.mesh, 1.0), float(1.0)]

    def model(value):
        my_model = F.HydrogenTransportProblem()
        my_model.mesh = my_mesh
        my_model.species = [F.Species("H")]
        my_mat = F.Material(1, 1, "1")
        my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=my_mat)
        my_model.subdomains = [my_subdomain]
        my_model.temperature = value
        my_model.initialise()

        return my_model

    for value in values:
        my_model = model(value)
        assert isinstance(my_model.temperature, fem.Constant)

    with raises(TypeError):
        x = sp.Symbol("x")
        model(sp.sin(sp.pi * x))

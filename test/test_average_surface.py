import festim as F
import numpy as np
from dolfinx import fem
import pytest


def test_average_surface_compute():
    """Test that the average surface export computes the correct value"""

    # BUILD
    L = 6.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=6)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("CG", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: x[0] * 0.5 + 1)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.AverageSurface(field=my_species, surface=dummy_surface)
    my_export.D = D

    # RUN
    my_export.compute()

    # TEST
    expected_value = 2.5
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

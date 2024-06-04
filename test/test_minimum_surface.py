import festim as F
import numpy as np
import ufl
from dolfinx.mesh import meshtags
from dolfinx import fem
import pytest
import os


def test_minimum_surface_export_compute():
    """Test that the minimum surface export computes the correct value"""

    # BUILD
    L = 4.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("CG", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: (x[0] - 2) ** 2)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.MinimumSurface(field=my_species, surface=dummy_surface)
    my_export.D = D

    # RUN
    my_export.compute()

    # TEST
    expected_value = 0
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

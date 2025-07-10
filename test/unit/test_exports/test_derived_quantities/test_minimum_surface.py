from festim import MinimumSurface
import fenics as f
import numpy as np
import pytest
from dolfin import MPI


@pytest.mark.parametrize("field,surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.MinimumSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_min = MinimumSurface(field, surface)
    assert my_min.title == f"Minimum {field} surface {surface} ({my_min.export_unit})"


def test_compute_minimum():
    """Test that the minimum surface export computes the correct value"""

    mesh = f.UnitSquareMesh(10, 10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0] + x[1]", degree=1), V)

    surface_markers = f.MeshFunction("size_t", mesh, 1, 0)

    right = f.CompiledSubDomain("near(x[0], 1) && on_boundary")
    right.mark(surface_markers, 1)

    dx = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_min = MinimumSurface("solute", surface)
    my_min.function = c
    my_min.dx = dx

    produced = my_min.compute(surface_markers)

    expected = 1.0

    assert produced == expected

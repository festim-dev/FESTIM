from this import s
from festim import TotalSurface
import fenics as f
import pytest
from .tools import c_1D, c_2D, c_3D
import pytest


@pytest.mark.parametrize("field,surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.TotalSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_total = TotalSurface(field, surface)
    assert my_total.title == "Total {} surface {}".format(field, surface)


class TestCompute:
    """Test that the total surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    surface_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(surface_markers, 2)

    ds = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_total = TotalSurface("solute", surface)
    my_total.function = c
    my_total.ds = ds

    def test_minimum(self):
        expected = f.assemble(self.c * self.ds(self.surface))

        produced = self.my_total.compute()
        assert produced == expected


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute surface 8 (H m-2)"),
        (c_1D, "T", "Total T surface 8 (K)"),
        (c_2D, "solute", "Total solute surface 8 (H m-1)"),
        (c_2D, "T", "Total T surface 8 (K m)"),
        (c_3D, "solute", "Total solute surface 8 (H)"),
        (c_3D, "T", "Total T surface 8 (K m2)"),
    ],
)
def test_title_with_units(function, field, expected_title):
    my_export = TotalSurface(surface=8, field=field)
    my_export.function = function
    my_export.show_units = True

    assert my_export.title == expected_title

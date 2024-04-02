from festim import AverageSurface
import fenics as f
import pytest


@pytest.mark.parametrize("field, surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.AverageSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_average = AverageSurface(field, surface)
    assert my_average.title == "Average {} surface {}".format(field, surface)


class TestCompute:
    """Test that the average surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    surface_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(surface_markers, 2)

    ds = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_average = AverageSurface("solute", surface)
    my_average.function = c
    my_average.ds = ds

    def test_h_average(self):
        expected = f.assemble(self.c * self.ds(self.surface)) / f.assemble(
            1 * self.ds(self.surface)
        )
        computed = self.my_average.compute()
        assert computed == expected

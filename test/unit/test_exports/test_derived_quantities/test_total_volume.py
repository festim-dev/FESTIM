from festim import TotalVolume
import fenics as f
import pytest


@pytest.mark.parametrize("field,volume", [("solute", 1), ("T", 2)])
def test_title(field, volume):
    """
    A simple test to check that the title is set
    correctly in festim.TotalVolume

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_total = TotalVolume(field, volume)
    assert my_total.title == "Total {} volume {}".format(field, volume)


class TestCompute:
    """Test that the total volume export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    volume_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(volume_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    volume = 1
    my_total = TotalVolume("solute", volume)
    my_total.function = c
    my_total.dx = dx

    def test_minimum(self):
        expected = f.assemble(self.c * self.dx(self.volume))

        produced = self.my_total.compute()
        assert produced == expected

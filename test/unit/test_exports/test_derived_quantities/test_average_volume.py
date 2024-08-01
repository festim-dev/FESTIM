from festim import AverageVolume
import fenics as f
import pytest


@pytest.mark.parametrize("field,volume", [("solute", 1), ("T", 2)])
def test_title(field, volume):
    """
    A simple test to check that the title is set
    correctly in festim.AverageVolume

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
    """

    my_average = AverageVolume(field, volume)
    assert my_average.title == "Average {} volume {}".format(field, volume)


class TestCompute:
    """Test that the average volume export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    volume_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(volume_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    volume = 1
    my_average = AverageVolume("solute", volume)
    my_average.function = c
    my_average.dx = dx

    def test_h_average(self):
        expected = f.assemble(self.c * self.dx(self.volume)) / f.assemble(
            1 * self.dx(self.volume)
        )
        computed = self.my_average.compute()
        assert computed == expected

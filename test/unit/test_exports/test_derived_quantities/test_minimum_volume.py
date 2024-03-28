from festim import MinimumVolume
import fenics as f
import numpy as np
import pytest


@pytest.mark.parametrize("field,volume", [("solute", 1), ("T", 2)])
def test_title(field, volume):
    """
    A simple test to check that the title is set
    correctly in festim.MinimumVolume

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_min = MinimumVolume(field, volume)
    assert my_min.title == "Minimum {} volume {}".format(field, volume)


class TestCompute:
    """Test that the minimum volume export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    volume_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(volume_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    volume = 1
    my_min = MinimumVolume("solute", volume)
    my_min.function = c
    my_min.dx = dx

    def test_minimum(self):
        dm = self.V.dofmap()
        subd_dofs = np.unique(
            np.hstack(
                [
                    dm.cell_dofs(c.index())
                    for c in f.SubsetIterator(self.volume_markers, self.volume)
                ]
            )
        )
        expected = np.min(self.c.vector().get_local()[subd_dofs])

        produced = self.my_min.compute(self.volume_markers)
        assert produced > 0
        assert produced == expected

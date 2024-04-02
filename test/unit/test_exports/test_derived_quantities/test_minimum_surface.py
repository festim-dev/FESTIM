from festim import MinimumSurface
import fenics as f
import numpy as np
import pytest


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
    assert my_min.title == "Minimum {} surface {}".format(field, surface)


class TestCompute:
    """Test that the minimum surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    surface_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(surface_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_min = MinimumSurface("solute", surface)
    my_min.function = c
    my_min.dx = dx

    def test_minimum(self):
        dm = self.V.dofmap()
        subd_dofs = np.unique(
            np.hstack(
                [
                    dm.cell_dofs(c.index())
                    for c in f.SubsetIterator(self.surface_markers, self.surface)
                ]
            )
        )
        expected = np.min(self.c.vector().get_local()[subd_dofs])

        produced = self.my_min.compute(self.surface_markers)
        assert produced == expected

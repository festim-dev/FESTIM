from festim import MaximumSurface
import fenics as f
import numpy as np
import pytest


@pytest.mark.parametrize("field,surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.MaximumSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_max = MaximumSurface(field, surface)
    assert my_max.title == "Maximum {} surface {}".format(field, surface)


class TestCompute:
    """Test that the maximum surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    surface_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(surface_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_max = MaximumSurface("solute", surface)
    my_max.function = c
    my_max.dx = dx

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
        expected = np.max(self.c.vector().get_local()[subd_dofs])

        produced = self.my_max.compute(self.surface_markers)
        assert produced == expected

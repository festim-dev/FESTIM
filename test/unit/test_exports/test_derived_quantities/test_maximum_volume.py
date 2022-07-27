from festim import MaximumVolume
import fenics as f
import numpy as np


def test_title_H():
    volume = 1
    field = "solute"
    my_max = MaximumVolume(field, volume)
    assert my_max.title == "Maximum {} volume {}".format(field, volume)


def test_title_T():
    volume = 2
    field = "T"
    my_max = MaximumVolume(field, volume)
    assert my_max.title == "Maximum {} volume {}".format(field, volume)


class TestCompute:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    volume_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(volume_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    volume = 1
    my_max = MaximumVolume("solute", volume)
    my_max.function = c
    my_max.dx = dx

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
        expected = np.max(self.c.vector().get_local()[subd_dofs])

        produced = self.my_max.compute(self.volume_markers)
        assert produced == expected

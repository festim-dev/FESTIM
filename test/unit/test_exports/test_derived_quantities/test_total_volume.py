from festim import TotalVolume
import fenics as f


def test_title_H():
    volume = 1
    field = "solute"
    my_total = TotalVolume(field, volume)
    assert my_total.title == "Total {} volume {}".format(field, volume)


def test_title_T():
    volume = 2
    field = "T"
    my_total = TotalVolume(field, volume)
    assert my_total.title == "Total {} volume {}".format(field, volume)


class TestCompute:
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

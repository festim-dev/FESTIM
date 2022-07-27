from festim import AverageVolume
import fenics as f


def test_title_H():
    volume = 1
    field = "solute"
    my_average = AverageVolume(field, volume)
    assert my_average.title == "Average {} volume {}".format(field, volume)


def test_title_T():
    volume = 2
    field = "T"
    my_average = AverageVolume(field, volume)
    assert my_average.title == "Average {} volume {}".format(field, volume)


class TestCompute:
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

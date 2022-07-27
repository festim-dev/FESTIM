from festim import AverageSurface
import fenics as f


def test_title_H():
    surface = 1
    field = "solute"
    my_average = AverageSurface(field, surface)
    assert my_average.title == "Average {} surface {}".format(field, surface)


def test_title_T():
    surface = 2
    field = "T"
    my_average = AverageSurface(field, surface)
    assert my_average.title == "Average {} surface {}".format(field, surface)


class TestCompute:
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

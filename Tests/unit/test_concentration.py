import FESTIM
import fenics as f
import sympy as sp
from pathlib import Path


class TestGetComp:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, 'P', 1)

    def test_get_comp_from_expression(self):
        my_conc = FESTIM.Concentration()
        comp = my_conc.get_comp(self.V, 1 + FESTIM.t - FESTIM.x)

        for t in [1, 1.5, 6, 8]:
            comp.t = t
            for x in [2, 5, 0, 6]:
                assert comp(x) == 1 + t - x

    def test_get_comp_from_xdmf(self, tmpdir):

        # build
        value = 1 + FESTIM.t - FESTIM.x

        u = f.Expression(sp.printing.ccode(value), degree=1, t=0)
        u = f.interpolate(u, self.V)

        d = tmpdir.mkdir("folder")
        time = 2
        label = "label"
        filename = str(Path(d.join("u_out.xdmf")))
        with f.XDMFFile(filename) as file:
            file.write_checkpoint(
                u, label, time,
                f.XDMFFile.Encoding.HDF5,
                append=False)

        my_conc = FESTIM.Concentration()

        # run
        comp = my_conc.get_comp(self.V, filename, label=label, time_step=0)

        # test
        for x in [0, 0.1, 0.2, 0.5, 0.6]:
            assert comp(x) == 1 + 0 - x


class TestInitialise:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, 'P', 1)
    u = f.Function(V)
    my_conc = FESTIM.Concentration(previous_solution=u)

    def test_with_expression(self):
        # build
        value = 1 + 2 * FESTIM.x**4
        expected_sol = self.my_conc.get_comp(self.V, value)

        # run
        self.my_conc.initialise(self.V, value)

        # test
        for x in [0, 0.5, 0.3, 0.6]:
            assert self.my_conc.previous_solution(x) == expected_sol(x)

    def test_with_xdmf(self, tmpdir):
        # build
        value = 1 + 2 * FESTIM.x**4

        u = f.Expression(sp.printing.ccode(value), degree=1, t=0)
        u = f.interpolate(u, self.V)

        d = tmpdir.mkdir("folder")
        time = 2
        label = "label"
        filename = str(Path(d.join("u_out.xdmf")))
        with f.XDMFFile(filename) as file:
            file.write_checkpoint(
                u, label, time,
                f.XDMFFile.Encoding.HDF5,
                append=False)

        expected_sol = self.my_conc.get_comp(
            self.V, filename, label=label, time_step=0)

        # run
        self.my_conc.initialise(self.V, value)

        # test
        for x in [0, 0.5, 0.3, 0.6]:
            assert self.my_conc.previous_solution(x) == expected_sol(x)

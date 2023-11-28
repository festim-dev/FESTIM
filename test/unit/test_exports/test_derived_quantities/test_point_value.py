from festim import PointValue
import fenics as f
import pytest


def test_title_H():
    x = 1
    field = "solute"
    my_value = PointValue(field, x)
    assert my_value.title == "{} value at {}".format(field, x)


def test_title_T():
    x = 1
    field = "T"
    my_value = PointValue(field, x)
    assert my_value.title == "{} value at {}".format(field, x)


class TestCompute:
    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    x = 1
    my_value = PointValue("solute", x)
    my_value.function = c

    def test_minimum(self):
        expected = self.c(self.x)

        produced = self.my_value.compute()
        assert produced == expected

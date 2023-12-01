from festim import PointValue
import fenics as f
import pytest


@pytest.mark.parametrize("field", ["solute", "T"])
def test_title(field):
    x = 1
    my_value = PointValue(field, x)
    assert my_value.title == "{} value at [{}]".format(field, x)


@pytest.mark.parametrize(
    "mesh,x", [(f.UnitIntervalMesh(10), 1), (f.UnitCubeMesh(10, 10, 10), (1, 0, 1))]
)
def test_point_compute(mesh, x):
    V = f.FunctionSpace(mesh, "P", 1)
    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    my_value = PointValue("solute", x)
    my_value.function = c

    expected = c(x)
    produced = my_value.compute()
    assert produced == expected

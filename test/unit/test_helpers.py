from festim import (
    kJmol_to_eV,
    k_B,
    R,
    as_constant,
    as_expression,
    as_constant_or_expression,
    t,
)
from fenics import Constant, Expression, UserExpression
import pytest


def test_energy_converter():
    test_values = [2, 30, 20.5, -2, -12.2]
    for energy_value in test_values:
        energy_in_eV = kJmol_to_eV(energy_value)
        expected_value = k_B * energy_value * 1e3 / R

        assert energy_in_eV == expected_value


@pytest.mark.parametrize("constant", [3, 3.0, -2.0, Constant(2.0)])
def test_as_constant(constant):
    assert isinstance(as_constant(constant), Constant)


class CustomExpr(UserExpression):
    def __init__(self):
        super().__init__()

    def eval(self, x, values):
        values[0] = x


@pytest.mark.parametrize(
    "expression,type",
    [
        (3 * t, Expression),
        (Expression("2 + x[0]", degree=2), Expression),
        (CustomExpr(), UserExpression),
    ],
)
def test_as_expression(expression, type):
    assert isinstance(as_expression(expression), type)


@pytest.mark.parametrize(
    "expression,type",
    [
        # constants
        (3, Constant),
        (3.0, Constant),
        (-2.0, Constant),
        (Constant(2.0), Constant),
        # expressions
        (3 * t, Expression),
        (Expression("2 + x[0]", degree=2), Expression),
        (CustomExpr(), UserExpression),
    ],
)
def test_as_constant_or_expression(expression, type):
    assert isinstance(as_constant_or_expression(expression), type)

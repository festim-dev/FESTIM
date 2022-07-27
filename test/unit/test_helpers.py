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


def test_energy_converter():
    test_values = [2, 30, 20.5, -2, -12.2]
    for energy_value in test_values:
        energy_in_eV = kJmol_to_eV(energy_value)
        expected_value = k_B * energy_value * 1e3 / R

        assert energy_in_eV == expected_value


def test_as_constant():
    assert isinstance(as_constant(3), Constant)
    assert isinstance(as_constant(3.0), Constant)
    assert isinstance(as_constant(-2.0), Constant)
    assert isinstance(as_constant(Constant(2.0)), Constant)


def test_as_expression():
    assert isinstance(as_expression(3 * t), Expression)
    assert isinstance(as_expression(Expression("2 + x[0]", degree=2)), Expression)

    class CustomExpr(UserExpression):
        def __init__(self):
            super().__init__()

        def eval(self, x, values):
            values[0] = x

    assert isinstance(as_expression(CustomExpr()), UserExpression)


def test_as_constant_or_expression():

    # constants
    assert isinstance(as_constant_or_expression(3), Constant)
    assert isinstance(as_constant_or_expression(3.0), Constant)
    assert isinstance(as_constant_or_expression(-2.0), Constant)
    assert isinstance(as_constant_or_expression(Constant(2.0)), Constant)

    # expressions
    assert isinstance(as_constant_or_expression(3 * t), Expression)
    assert isinstance(
        as_constant_or_expression(Expression("2 + x[0]", degree=2)), Expression
    )

    class CustomExpr(UserExpression):
        def __init__(self):
            super().__init__()

        def eval(self, x, values):
            values[0] = x

    assert isinstance(as_constant_or_expression(CustomExpr()), UserExpression)

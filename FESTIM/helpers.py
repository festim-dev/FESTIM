import festim
import xml.etree.ElementTree as ET
from fenics import Expression, UserExpression, Constant
import sympy as sp


def update_expressions(expressions, t):
    """Update all FEniCS Expression() in expressions.

    Arguments:
    - expressions: list, contains the fenics Expression
    to be updated.
    - t: float, time.
    """
    for expression in expressions:
        expression.t = t
    return expressions


def as_expression(expr):
    # if expr is already a fenics Expression, use it as is
    if isinstance(expr, (Expression, UserExpression)):
        return expr
    # else assume it's a sympy expression
    else:
        expr_ccode = sp.printing.ccode(expr)
        return Expression(expr_ccode, degree=2, t=0)


def as_constant(constant):
    if isinstance(constant, Constant):
        return constant
    elif isinstance(constant, (int, float)):
        return Constant(constant)


def as_constant_or_expression(val):
    if isinstance(val, (Constant, Expression, UserExpression)):
        return val
    elif isinstance(val, (int, float)):
        return Constant(val)
    else:
        expr_ccode = sp.printing.ccode(val)
        return Expression(expr_ccode, degree=2, t=0)


def kJmol_to_eV(energy):
    """Converts an energy value given in units kJ mol^{-1} to eV

    Args:
        energy (float): Energy in kJ mol^{-1}

    Returns:
        energy (float): Energy in eV
    """
    energy_in_eV = festim.k_B * energy * 1e3 / festim.R

    return energy_in_eV


def extract_xdmf_times(filename):
    """Returns a list of timesteps in an XDMF file

    Args:
        filename (str): the XDMF filename (must end with .xdmf)

    Returns:
        list: the timesteps
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    domains = list(root)
    domain = domains[0]
    grids = list(domain)
    grid = grids[0]

    times = []
    for c in grid:
        for element in c:
            if "Time" in element.tag:
                times.append(float(element.attrib["Value"]))
    return times


def extract_xdmf_labels(filename):
    """Returns a list of labels in an XDMF file

    Args:
        filename (str): the XDMF filename (must end with .xdmf)

    Returns:
        list: the labels
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    domains = list(root)
    domain = domains[0]
    grids = list(domain)
    grid = grids[0]

    labels = []
    for c in grid:
        for element in c:
            if "Attribute" in element.tag:
                labels.append(element.attrib["Name"])

    unique_labels = list(set(labels))
    return unique_labels

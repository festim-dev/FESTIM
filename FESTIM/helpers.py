import FESTIM
import xml.etree.ElementTree as ET


def update_expressions(expressions, t):
    '''Update all FEniCS Expression() in expressions.

    Arguments:
    - expressions: list, contains the fenics Expression
    to be updated.
    - t: float, time.
    '''
    for expression in expressions:
        expression.t = t
    return expressions


def kJmol_to_eV(energy):
    """Converts an energy value given in units kJ mol^{-1} to eV

    Args:
        energy (float): Energy in kJ mol^{-1}

    Returns:
        energy (float): Energy in eV
    """
    energy_in_eV = FESTIM.k_B*energy*1e3/FESTIM.R

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

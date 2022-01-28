import FESTIM


def formulation(simulation):
    """Creates the variational formulation for the H transport problem

    Args:
        simulation (FESTIM.Simulation): main simulation instance

    Returns:
        fenics.Form, list: problem variational formulation, contains
            fenics.Expression() to be updated
    """
    expressions = []
    F = 0

    # diffusion + transient terms

    simulation.mobile.create_form(
        simulation.materials, simulation.dx, simulation.T, simulation.dt,
        traps=simulation.traps,
        chemical_pot=simulation.settings.chemical_pot, soret=simulation.settings.soret)
    F += simulation.mobile.F
    expressions += simulation.mobile.sub_expressions

    # Add traps
    simulation.traps.create_forms(
        simulation.mobile, simulation.materials,
        simulation.T, simulation.dx, simulation.dt,
        simulation.settings.chemical_pot)
    F += simulation.traps.F
    expressions += simulation.traps.sub_expressions

    return F, expressions


def formulation_extrinsic_traps(simulation):
    """Creates a list that contains formulations to be solved during
    time stepping.

    Args:
        simulation (FESTIM.Simulation): the simulation

    Returns:
        list, list: list of fenics.Form and list of fenics.Expression
    """
    formulations = []
    expressions = []
    for trap in simulation.traps.traps:
        if isinstance(trap, FESTIM.ExtrinsicTrap):
            trap.create_form_density(simulation.dx, simulation.dt)
            formulations.append(trap.form_density)
    return formulations, expressions

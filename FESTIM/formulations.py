import FESTIM


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

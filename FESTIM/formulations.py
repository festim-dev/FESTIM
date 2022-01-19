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

    # TODO source term should be like Temperature an argument of Mobile.
    if "source_term" in simulation.parameters:
        source_term = simulation.parameters["source_term"]
    else:
        source_term = []
    simulation.mobile.create_form(
        simulation.materials, simulation.dx, simulation.T, simulation.dt,
        traps=simulation.traps, source_term=source_term,
        chemical_pot=simulation.chemical_pot, soret=simulation.soret)
    F += simulation.mobile.F
    expressions += simulation.mobile.sub_expressions

    # Add traps
    simulation.traps.create_forms(
        simulation.mobile, simulation.materials,
        simulation.T, simulation.dx, simulation.dt,
        simulation.chemical_pot)
    F += simulation.traps.F
    expressions += simulation.traps.sub_expressions

    return F, expressions


def formulation_extrinsic_traps(simulation):
    """Creates a list that contains formulations to be solved during
    time stepping.

    Arguments:


    Returns:
        list -- contains fenics.Form to be solved for extrinsic trap density
        list -- contains fenics.Expression to be updated
    """
    formulations = []
    expressions = []
    for trap in simulation.traps.traps:
        if isinstance(trap, FESTIM.ExtrinsicTrap):
            trap.create_form_density(simulation.dx, simulation.dt)
            formulations.append(trap.form_density)
    return formulations, expressions

class Settings:
    """
    Args:
        absolute_tolerance (float): the absolute tolerance of the newton
            solver
        relative_tolerance (float): the relative tolerance of the newton
            solver
        maximum_iterations (int, optional): maximum iterations allowed for
            the solver to converge. Defaults to 30.
        transient (bool, optional): If set to True, the simulation will be
            transient. Defaults to True.
        final_time (float, optional): final time of the simulation.
            Defaults to None.
        chemical_pot (bool, optional): if True, conservation of chemical
            potential will be assumed. Defaults to False.
        soret (bool, optional): if True, Soret effect will be assumed.
            Defaults to False.
        traps_element_type (str, optional): Finite element used for traps.
            If traps densities are discontinuous (eg. different materials)
            "DG" is recommended. Defaults to "CG".
        update_jacobian (bool, optional): If set to False, the Jacobian of
            the formulation will be computed only once at the beggining.
            Else it will be computed at each time step. Defaults to True.
        linear_solver (str, optional): linear solver method for the newton solver,
            options can be veiwed by print(list_linear_solver_methods()).
            More information can be found at: https://fenicsproject.org/pub/tutorial/html/._ftut1017.html.
            Defaults to None, for the newton solver this is: "umfpack".

    Attributes:
        transient (bool): transient or steady state sim
        final_time (float): final time of the simulation.
        chemical_pot (bool): conservation of chemical potential
        soret (bool): soret effect
        absolute_tolerance (float): the absolute tolerance of the newton
            solver
        relative_tolerance (float): the relative tolerance of the newton
            solver
        maximum_iterations (int): maximum iterations allowed for
            the solver to converge
        traps_element_type (str): Finite element used for traps.
        update_jacobian (bool):
        linear_solver (str): linear solver method for the newton solver
    """

    def __init__(
        self,
        absolute_tolerance,
        relative_tolerance,
        maximum_iterations=30,
        transient=True,
        final_time=None,
        chemical_pot=False,
        soret=False,
        traps_element_type="CG",
        update_jacobian=True,
        linear_solver=None,
    ):

        # TODO maybe transient and final_time are redundant
        self.transient = transient
        self.final_time = final_time
        self.chemical_pot = chemical_pot
        self.soret = soret

        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.maximum_iterations = maximum_iterations

        self.traps_element_type = traps_element_type
        self.update_jacobian = update_jacobian
        self.linear_solver = linear_solver

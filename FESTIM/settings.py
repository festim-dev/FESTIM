
class Settings:
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
        update_jacobian=True
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

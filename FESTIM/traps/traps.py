class Traps:
    def __init__(self, traps=[]) -> None:
        self.traps = traps
        self.F = None
        self.sub_expressions = []

        # add ids if unspecified
        for i, trap in enumerate(self.traps, 1):
            if trap.id is None:
                trap.id = i

    def create_forms(self, mobile, materials, T, dx, dt=None,
                     chemical_pot=False):
        self.F = 0
        for trap in self.traps:
            trap.create_form(mobile, materials, T, dx, dt=dt,
                             chemical_pot=chemical_pot)
            self.F += trap.F
            self.sub_expressions += trap.sub_expressions

    def get_trap(self, id):
        for trap in self.traps:
            if trap.id == id:
                return trap
        raise ValueError("Couldn't find trap {}".format(id))

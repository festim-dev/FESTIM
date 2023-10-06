class Species:
    def __init__(self, name=None) -> None:
        self.name = name

        self.solution = None
        self.prev_solution = None
        self.test_function = None
        self.form = None

    
class Trap(Species):
    def __init__(self, name=None) -> None:
        super().__init__(name)
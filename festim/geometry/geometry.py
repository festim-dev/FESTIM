from dolfinx import fem
import ufl


class Geometry:
    """
    Domain class

    Attributes:

    """

    def __init__(self, mesh=None, subdomains=[]) -> None:
        """
        Args:

        """
        self.mesh = mesh
        self.subdomains = subdomains

    def define_function_space(self):
        elements = ufl.FiniteElement("CG", self.mesh.mesh.ufl_cell(), 1)
        self.function_space = fem.FunctionSpace(self.mesh.mesh, elements)

    def define_subdomains(self):
        self.mesh.define_measures(self.function_space)

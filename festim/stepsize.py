import festim as F


class Stepsize:
    """
    A class for evaluating the stepsize of transient simulations.

    Args:
        initial_value (float, int): initial stepsize.

    Attributes:
        initial_value (float, int): initial stepsize.
    """

    def __init__(
        self,
        initial_value,
    ) -> None:
        self.initial_value = initial_value

    def get_dt(self, mesh):
        """Defines the dt value
        Args:
            mesh (dolfinx.mesh.Mesh): the domain mesh
        Returns:
            fem.Constant: the dt value
        """
        return F.as_fenics_constant(self.initial_value, mesh)

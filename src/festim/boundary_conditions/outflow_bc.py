import ufl
from dolfinx import fem

from festim import subdomain as _subdomain
from festim.advection import VelocityField
from festim.species import Species
from festim.subdomain import VolumeSubdomain


class OutflowBC:
    """
    Dirichlet boundary condition class
    u = value

    Args:
        subdomain: The surface subdomain where the boundary condition is applied
        value: The value of the boundary condition

    Attributes:
        subdomain: The surface subdomain where the boundary condition is applied
        value: The value of the boundary condition
        value_fenics: The value of the boundary condition in fenics format
        bc_expr: The expression of the boundary condition that is used to
            update the `value_fenics`

    """

    subdomain: _subdomain.SurfaceSubdomain

    def __init__(
        self,
        velocity: fem.Function,
        species: Species | list[Species],
        subdomain: _subdomain.SurfaceSubdomain,
    ):
        self.subdomain = subdomain
        self.velocity_field = velocity
        self.species = species

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        err_message = f"velocity must be a fem.Function, or callable not {type(value)}"
        if value is None:
            self._velocity = VelocityField(value)
        elif isinstance(
            value,
            fem.Function,
        ):
            self._velocity = VelocityField(value)
        elif isinstance(value, fem.Constant | fem.Expression | ufl.core.expr.Expr):
            raise TypeError(err_message)
        elif callable(value):
            self._velocity = VelocityField(value)
        else:
            raise TypeError(err_message)

    @property
    def species(self) -> list[Species]:
        return self._species

    @species.setter
    def species(self, value):
        if not isinstance(value, list):
            value = [value]
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not "
                    f"{type(spe)}"
                )
        self._species = value

    @property
    def subdomain(self):
        return self._subdomain

    @subdomain.setter
    def subdomain(self, value):
        if value is None:
            self._subdomain = value
        elif isinstance(value, VolumeSubdomain):
            self._subdomain = value
        else:
            raise TypeError(
                f"Subdomain must be a festim.Subdomain object, not {type(value)}"
            )

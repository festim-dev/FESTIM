import ufl
from dolfinx import fem

from festim.helpers import Value, nmm_interpolate
from festim.species import Species
from festim.subdomain import VolumeSubdomain


class AdvectionTerm:
    """
    Advection term class

    args:
        velocity: the velocity field or function
        subdomain: the volume subdomain where the velocity is to be applied
        species: the species to which the velocity field is acting on

    attributes:
        velocity: the velocity field or function
        subdomain: the volume subdomain where the velocity is to be applied
        species: the species to which the velocity field is acting on

    """

    velocity: fem.Function
    subdomain: VolumeSubdomain
    species: Species

    def __init__(
        self,
        velocity: fem.Function,
        subdomain: VolumeSubdomain,
        species: Species,
    ):
        self.velocity = velocity
        self.subdomain = subdomain
        self.species = species

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if value is None:
            self._velocity = Value(value)
        elif isinstance(
            value,
            fem.Expression | ufl.core.expr.Expr | fem.Function,
        ):
            self._velocity = Value(value)
        elif callable(value):
            self._velocity = Value(value)
        else:
            raise TypeError(
                "velocity must be a fem.Expression, ufl.core.expr.Expr, fem.Function, "
                f"or callable not {value}"
            )

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

    @property
    def species(self) -> list[Species]:
        return self._species

    @species.setter
    def species(self, value):
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not "
                    f"{type(spe)}"
                )
        self._species = value

    def update_velocity_field(self, t: float):
        """Updates the value

        Args:
            t: the time
        """
        nmm_interpolate(self.velocity.fenics_object, self.velocity.input_value(t=t))

from collections.abc import Callable
from typing import Optional

import basix
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


class VelocityField(Value):
    """
    A class to handle input values of velocity fields from users and convert them to a
    relevent fenics object

    Args:
        input_value: The value of the user input

    Attributes:
        input_value : The value of the user input
        fenics_interpolation_expression : The expression of the user input that is used
            to update the `fenics_object`
        fenics_object : The value of the user input in fenics format
        explicit_time_dependent : True if the user input value is explicitly time
            dependent
        temperature_dependent : True if the user input value is temperature dependent
        vector_function_space: the vector function space of the fenics object
    """

    input_value: fem.Function | Callable

    fenics_object: fem.Function
    explicit_time_dependent: bool
    temperature_dependent: bool
    vector_function_space: fem.FunctionSpace

    def convert_input_value(
        self,
        function_space: fem.FunctionSpace,
        t: Optional[fem.Constant] = None,
    ):
        """Converts a user given value to a relevent fenics object

        Args:
            function_space: the function space of the fenics object
            t: the time, optional
        """

        if isinstance(self.input_value, fem.Function):
            vel = self.input_value

        elif callable(self.input_value):
            # if callable function has args other than time, t, raise Typer Error
            args = self.input_value.__code__.co_varnames
            if args != ("t",):
                raise TypeError(
                    "velocity function can only be a function of time arg t"
                )

            vel = self.input_value(t)

            # if function does not return an fem.Fucntion, raise Typer Error
            if not isinstance(vel, fem.Function):
                raise ValueError(
                    "A time dependent advection field should return an fem.Function"
                    f", not a {type(vel)}"
                )

        # create vector function space and function
        v_cg = basix.ufl.element(
            "Lagrange",
            function_space.mesh.topology.cell_name(),
            1,
            shape=(function_space.mesh.topology.dim,),
        )
        self.vector_function_space = fem.functionspace(function_space.mesh, v_cg)
        self.fenics_object = fem.Function(self.vector_function_space)

        # interpolate input value into fenics object function
        nmm_interpolate(self.fenics_object, vel)

    def update(self, t: fem.Constant):
        """Updates the velocity field

        Args:
            t: the time
        """
        nmm_interpolate(self.fenics_object, self.input_value(t=float(t)))

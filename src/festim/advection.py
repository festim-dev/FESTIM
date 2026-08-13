from collections.abc import Callable

import basix
import ufl
from dolfinx import fem

from festim.drift import DriftTermBase
from festim.helpers import Value, nmm_interpolate
from festim.species import Species
from festim.subdomain import VolumeSubdomain


class AdvectionTerm(DriftTermBase):
    """Advection term class.

    Transport by a velocity field given directly, as opposed to one built from a driving
    gradient (see :class:`festim.SoretTerm`, :class:`festim.ElectromigrationTerm`).

    Assembled in divergence form, ``-div(c v)``, like every drift term. This differs
    from the ``v . grad(c)`` FESTIM assembled before: the two agree in the interior
    wherever ``div(v) == 0``, which holds for an incompressible flow, but only the
    divergence form carries species out through a boundary that has no flux condition on
    it -- an outlet, typically. See :func:`festim.drift.drift_form`.

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
        super().__init__(subdomain=subdomain, species=species)
        self.velocity = velocity

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

    def convert_inputs(self, function_space, t=None, temperature=None):
        self.velocity.convert_input_value(function_space=function_space, t=t)

    def time_dependent_inputs(self):
        return [self.velocity]

    def drift_velocity(self, D, temperature):
        return self.velocity.fenics_object


class VelocityField(Value):
    """A class to handle input values of velocity fields from users and convert them to
    a relevent fenics object.

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
        t: fem.Constant | None = None,
    ):
        """Converts a user given value to a relevent fenics object.

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
        # NOTE: the shape is the *geometric* dimension, not the topological one: on a
        # codim-1 subdomain the mesh is a manifold (eg. a line in 2D) whose cells are
        # 1D but whose points, and therefore velocities and gradients, are ambient
        v_cg = basix.ufl.element(
            "Lagrange",
            function_space.mesh.topology.cell_name(),
            1,
            shape=(function_space.mesh.geometry.dim,),
        )
        self.vector_function_space = fem.functionspace(function_space.mesh, v_cg)
        self.fenics_object = fem.Function(self.vector_function_space)

        # interpolate input value into fenics object function
        nmm_interpolate(self.fenics_object, vel)

    def update(self, t: fem.Constant):
        """Updates the velocity field.

        Args:
            t: the time
        """
        nmm_interpolate(self.fenics_object, self.input_value(t=float(t)))

from typing import Union

import ufl
from dolfinx import fem

from festim.helpers import as_fenics_constant
from festim.subdomain.volume_subdomain import (
    VolumeSubdomain as _VolumeSubdomain,
)


class Species:
    """
    Hydrogen species class for H transport simulation.

    Args:
        name: a name given to the species. Defaults to None.
        mobile: whether the species is mobile or not. Defaults to True.
        subdomain: the volume subdomain where the species is. Defaults to None.

    Attributes:
        name: a name given to the species.
        mobile: whether the species is mobile or not.
        solution: the solution for the current timestep
        prev_solution: the solution for the previous timestep
        test_function: the testfunction associated with this species
        sub_function: the sub function of the species in case of multiple species in
            the same function space
        sub_function_space: the subspace of the function space
        collapsed_function_space: the collapsed function space for a species in the
            function space. In case single species case, this is None.
        map_sub_to_main_solution: the mapping from the sub solution dofs to the main
            solution dofs
        post_processing_solution: the solution for post processing
        concentration: the concentration of the species
        subdomains: the volume subdomains where the species is
        subdomain_to_solution: a dictionary mapping subdomains to solutions
        subdomain_to_prev_solution: a dictionary mapping subdomains to previous
            solutions
        subdomain_to_test_function: a dictionary mapping subdomains to test functions
        subdomain_to_post_processing_solution: a dictionary mapping subdomains to post
            processing solutions
        subdomain_to_collapsed_function_space: a dictionary mapping subdomains to
            collapsed function spaces
        subdomain_to_function_space: a dictionary mapping subdomains to function spaces

    Examples:
        :: testsetup:: Species

            from festim import Species

        :: testcode:: Species

            Species(name="H")
            Species(name="Trap", mobile=False)


    """

    name: str | None
    mobile: bool
    solution: fem.Function | None
    prev_solution: fem.Function | None
    test_function: ufl.argument.Argument | None
    sub_function_space: fem.function.FunctionSpace | None
    collapsed_function_space: fem.function.FunctionSpace | None
    map_sub_to_main_solution: list | None
    post_processing_solution: fem.Function | None
    concentration: fem.Function | None

    subdomains: list[_VolumeSubdomain] | _VolumeSubdomain | None
    subdomain_to_solution: dict
    subdomain_to_prev_solution: dict
    subdomain_to_test_function: dict
    subdomain_to_post_processing_solution: dict
    subdomain_to_collapsed_function_space: dict
    subdomain_to_function_space: dict

    def __init__(
        self,
        name: str | None = None,
        mobile: bool = True,
        subdomains: list[_VolumeSubdomain] | _VolumeSubdomain | None = None,
    ) -> None:
        self.name = name
        self.mobile = mobile
        self.solution = None
        self.prev_solution = None
        self.test_function = None
        self.sub_function = None
        self.sub_function_space = None
        self.collapsed_function_space = None
        self.map_sub_to_main_solution = None
        self.post_processing_solution = None

        self.subdomains = subdomains
        self.subdomain_to_solution = {}
        self.subdomain_to_prev_solution = {}
        self.subdomain_to_test_function = {}
        self.subdomain_to_post_processing_solution = {}
        self.subdomain_to_collapsed_function_space = {}
        self.subdomain_to_function_space = {}

    def __repr__(self) -> str:
        return f"Species({self.name})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        return self.solution

    @property
    def legacy(self) -> bool:
        """
        Check if we are using FESTIM 1.0 implementation or FESTIM 2.0
        """
        if not self.subdomain_to_solution:
            return True
        else:
            return False


class ImplicitSpecies:
    """Implicit species class for H transport simulation.
    c = n - others

    Args:
        n: the total concentration of the species
        others: the list of species from which the implicit species concentration is
            computed (c = n - others)
        name: a name given to the species. Defaults to None.

    Attributes:
        n: the total concentration of the species
        others: the list of species from which the implicit species concentration is
            computed (c = n - others)
        name: a name given to the species. Defaults to None.
        concentration: the concentration of the species
        value_fenics: the total concentration as a fenics object
    """

    n: Union[float, callable]
    others: list[Species] | None
    name: str | None
    concentration: ufl.form.Form
    value_fenics: fem.Constant | ufl.core.expr.Expr

    def __init__(
        self,
        n: Union[float, callable],
        others: list[Species] | None = None,
        name: str | None = None,
    ) -> None:
        self.name = name
        self.n = n
        self.others = others

    def __repr__(self) -> str:
        return f"ImplicitSpecies({self.name}, {self.n}, {self.others})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        if len(self.others) > 0:
            for other in self.others:
                if other.solution is None:
                    raise ValueError(
                        f"Cannot compute concentration of {self.name} "
                        + f"because {other.name} has no solution."
                    )
        return self.value_fenics - sum([other.solution for other in self.others])

    def create_value_fenics(self, mesh, t: fem.Constant):
        """Creates the value of the density as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.n, int | float):
            self.value_fenics = as_fenics_constant(mesh=mesh, value=self.n)

        elif isinstance(self.n, fem.Function | ufl.core.expr.Expr):
            self.value_fenics = self.n

        elif callable(self.n):
            arguments = self.n.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.n(t=float(t)), float | int):
                    raise ValueError(
                        "self.value should return a float or an int, not "
                        f"{type(self.n(t=float(t)))}"
                    )
                self.value_fenics = as_fenics_constant(
                    mesh=mesh, value=self.n(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x

                self.value_fenics = self.n(**kwargs)

    def update_density(self, t):
        """Updates the density value (only if the density is a function of time only)

        Args:
            t (float): the time
        """
        if isinstance(self.n, fem.Function | ufl.core.expr.Expr):
            return

        if callable(self.n):
            arguments = self.n.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.n(t=t)


def find_species_from_name(name: str, species: list):
    """Returns the correct species object from a list of species
    based on a string

    Args:
        name (str): the name of the species
        species (list): the list of species

    Returns:
        species (festim.Species): the species object with the correct name

    Raises:
        ValueError: if the species name is not found in the list of species

    """
    for spe in species:
        if spe.name == name:
            return spe
    raise ValueError(f"Species {name} not found in list of species")

import basix.ufl
from dolfinx import fem

from festim.enclosure._utils import check_dolfinx_version_for_enclosures


class GasSpecies:
    """A gas species living in a :py:class:`festim.Enclosure`.

    The partial pressure of the species in the enclosure is an unknown of the problem,
    represented by a real function space (one global degree of freedom).

    Args:
        name: a name given to the species
        initial_pressure: the partial pressure at t=0 (Pa)

    Attributes:
        name: a name given to the species
        initial_pressure: the partial pressure at t=0 (Pa)
        enclosure: the enclosure this species belongs to. Set by
            :py:class:`festim.Enclosure`
        function_space: the real function space holding the pressure
        solution: the pressure at the current timestep
        prev_solution: the pressure at the previous timestep
        test_function: the test function of the real function space
        F: the variational formulation of the pressure balance

    Examples:

        .. testsetup:: GasSpecies

            from festim import Enclosure, GasSpecies

        .. testcode:: GasSpecies

            H2 = GasSpecies(name="H2", initial_pressure=1e5)
            my_enclosure = Enclosure(volume=1e-3, species=[H2], temperature=500)
    """

    def __init__(self, name: str, initial_pressure: float = 0.0):
        self.name = name
        self.initial_pressure = initial_pressure

        self.enclosure = None
        self.function_space = None
        self.solution = None
        self.prev_solution = None
        self.test_function = None
        self.F = None

    def __repr__(self) -> str:
        return f"GasSpecies({self.name})"

    @property
    def pressure(self):
        """The pressure as a fenics object, to be used in boundary conditions."""
        return self.solution

    @property
    def value(self) -> float:
        """The current pressure as a float (Pa), collected across all MPI processes."""
        if self.solution is None:
            raise ValueError(
                f"The pressure of {self.name} is not available before initialise() "
                "is called on the problem."
            )
        self.solution.x.scatter_forward()
        comm = self.function_space.mesh.comm
        # a real function space has a single global dof, owned by one process only
        nb_owned = self.function_space.dofmap.index_map.size_local
        local = self.solution.x.array[:nb_owned]
        candidate = float(local[0]) if len(local) else None
        for value in comm.allgather(candidate):
            if value is not None:
                return value
        raise RuntimeError(f"No process owns the pressure dof of {self.name}")


def create_real_function_space(mesh) -> fem.FunctionSpace:
    """Creates a real function space (one global degree of freedom) on a mesh.

    Args:
        mesh: the dolfinx mesh

    Returns:
        the real function space
    """
    check_dolfinx_version_for_enclosures()
    return fem.functionspace(mesh, basix.ufl.real_element(mesh.basix_cell(), ()))

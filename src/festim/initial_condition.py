from collections.abc import Callable
from typing import Union

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import ufl
from dolfinx import fem

from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class InitialConditionBase:
    """Base initial condition class.

    Args:
        value: the value of the initial condition.
        volume: the volume subdomain where the initial condition is applied

    Attributes:
        value: the value of the initial condition.
        volume: the volume subdomain where the initial condition is applied
    """

    value: (
        float
        | int
        | fem.Constant
        | np.ndarray
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
    )
    volume: VolumeSubdomain

    def __init__(
        self,
        value: (
            float
            | int
            | fem.Constant
            | np.ndarray
            | fem.Expression
            | ufl.core.expr.Expr
            | fem.Function
        ),
        volume: VolumeSubdomain,
    ):
        self.value = value
        self.volume = volume

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # check that volume is festim.VolumeSubdomain
        if not isinstance(value, VolumeSubdomain):
            raise TypeError("volume must be of type festim.VolumeSubdomain")
        self._volume = value


class InitialConcentration(InitialConditionBase):
    """Initial concentration class.

    Args:
        value: the value of the initial concentration of a given species.
        species: the species to which the condition is applied
        volume: the volume subdomain where the initial condition is applied

    Attributes:
        value: the value of the initial concentration of a given species.
        species: the species to which the condition is applied
        volume: the volume subdomain where the initial condition is applied
        expr_fenics: the value of the initial condition in fenics expr format

    Examples:

        .. testsetup:: InitialConcentration

            from festim import InitialConcentration, Species, Material, VolumeSubdomain
            my_species = Species(name='test')
            dummy_mat = Material(D_0=1, E_D=0.1)
            my_vol = VolumeSubdomain(id=1, material=dummy_mat)

        .. testcode:: InitialConcentration

            InitialConcentration(value=1, species=my_species, volume=my_vol)
            InitialConcentration(
                value=lambda x: 1 + x[0],
                species=my_species,
                volume=my_vol
            )
            InitialConcentration(
                value=lambda T: 1 + T,
                species=my_species,
                volume=my_vol
            )
            InitialConcentration(
                value=lambda x, T: 1 + x[0] + T,
                species=my_species,
                volume=my_vol
            )
    """

    expr_fenics: Union[Callable, fem.Expression]
    species: Species

    def __init__(self, value, volume, species: Species):
        super().__init__(value=value, volume=volume)

        self.species = species

        self.expr_fenics = None

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that species is festim.Species or list of festim.Species
        if not isinstance(value, Species):
            raise TypeError("species must be of type festim.Species")

        self._species = value

    def create_expr_fenics(
        self,
        mesh: dolfinx.mesh.Mesh,
        temperature: fem.Function | fem.Constant,
        function_space: fem.functionspace,
    ):
        """Creates the expr_fenics of the initial condition.

        If the value is a float or int, a function is created with an array with the
        shape of the mesh and all set to the value. Otherwise, it is converted to a
        fem.Expression.

        Args:
            mesh: the mesh
            temperature: the temperature
            function_space: the function space of the species
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, int | float):
            self.expr_fenics = lambda x: np.full(x.shape[1], self.value)
        elif isinstance(self.value, fem.Function):
            self.expr_fenics = self.value
        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            kwargs = {}
            if "t" in arguments:
                raise ValueError("Initial condition cannot be a function of time.")
            if "x" in arguments:
                kwargs["x"] = x
            if "T" in arguments:
                kwargs["T"] = temperature

            self.expr_fenics = fem.Expression(
                self.value(**kwargs),
                function_space.element.interpolation_points,
            )


class InitialTemperature(InitialConditionBase):
    """Initial temperature class.

    Args:
        value: the value of the initial temperature
        volume: the volume subdomain where the initial condition is applied

    Attributes:
        value: the value of the initial temperature
        volume: the volume subdomain where the initial condition is applied
        expr_fenics: the value of the initial condition in fenics expr format

    Examples:

        .. testsetup:: InitialTemperature

            from festim import InitialTemperature, Material, VolumeSubdomain
            dummy_mat = Material(D_0=1, E_D=0.1)
            my_vol = VolumeSubdomain(id=1, material=dummy_mat)

        .. testcode:: InitialTemperature

            InitialTemperature(value=1, volume=my_vol)
            InitialTemperature(value=lambda x: 1 + x[0], volume=my_vol)
            InitialTemperature(value=lambda x, t: 1 + x[0] + t, volume=my_vol)
    """

    def __init__(self, value, volume):
        super().__init__(value=value, volume=volume)

        self.expr_fenics = None

    def create_expr_fenics(
        self,
        mesh: dolfinx.mesh.Mesh,
        function_space: fem.functionspace,
    ):
        """Creates the expr_fenics of the initial condition.

        If the value is a float or int, a function is created with an array with the
        shape of the mesh and all set to the value. Otherwise, it is converted to a
        fem.Expression.

        Args:
            mesh: the mesh
            function_space: the function space of the species
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, int | float):
            self.expr_fenics = lambda x: np.full(x.shape[1], self.value)
        elif isinstance(self.value, fem.Function):
            self.expr_fenics = self.value
        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            kwargs = {}
            if "t" in arguments:
                raise ValueError("Initial condition cannot be a function of time.")
            if "x" in arguments:
                kwargs["x"] = x

            self.expr_fenics = fem.Expression(
                self.value(**kwargs),
                function_space.element.interpolation_points,
            )


#: the io4dolfinx backends that can store a checkpoint, and the file each writes
_CHECKPOINT_BACKENDS = {"adios2": ".bp", "h5py": ".h5"}


def read_function_from_file(
    filename: str,
    name: str,
    timestamp: int | float,
    family="P",
    order: int = 1,
    mesh: dolfinx.mesh.Mesh | None = None,
    backend: str = "adios2",
) -> fem.Function:
    """Read a function from a checkpoint file.

    Reads a checkpoint written by an export with ``format="checkpoint"`` (or directly
    by :func:`io4dolfinx.write_function`). The visualisation formats (``"vtx"``,
    ``"vtkhdf"``, ``"xdmf"``) store values interpolated onto the mesh nodes rather than
    the degrees of freedom, and cannot be read back with this function.

    The function space the checkpoint is read into is built from `family` and `order`,
    so these must match the space the function was written from.

    note::
        The function is read from a file using io4dolfinx. For more information
        see the [io4dolfinx documentation](
        scientificcomputing.github.io/io4dolfinx/README.html).

    Args:
        filename: the filename
        name: the name of the function
        timestamp: the timestamp of the function
        family: the family of the function space
        order: the order of the function space
        mesh: Mesh to create input space on.
        backend: the io4dolfinx backend the checkpoint was written with, ``"adios2"``
            (``.bp``) or ``"h5py"`` (``.h5``). Must match the `backend` given to the
            export that wrote the file.

    Returns:
        the function

    Raises:
        ValueError: if `backend` is not a backend that can store a checkpoint
    """
    if backend not in _CHECKPOINT_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}, expected one of "
            f"{sorted(_CHECKPOINT_BACKENDS)}."
        )
    mesh_in = io4dolfinx.read_mesh(
        filename=filename, comm=MPI.COMM_WORLD, backend=backend
    )
    V_in = fem.functionspace(mesh_in, (family, order))
    u_in = fem.Function(V_in)
    io4dolfinx.read_function(
        filename=filename,
        u=u_in,
        name=name,
        time=timestamp,
        backend=backend,
    )
    if mesh is None:
        return u_in
    else:
        V = fem.functionspace(mesh, (family, order))
        u = fem.Function(V)
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        cells = np.arange(num_cells, dtype=np.int32)
        padding = 1e2 * np.finfo(mesh.geometry.x.dtype).eps
        idata = fem.create_interpolation_data(V, V_in, cells=cells, padding=padding)
        u.interpolate_nonmatching(u_in, cells, idata)
        return u

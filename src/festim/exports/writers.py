"""Writer strategies backing the export classes.

Each writer knows how to put a list of :class:`dolfinx.fem.Function` in a file, and
nothing about FESTIM. Adding a new output format means adding a class here and an
entry in :data:`festim.exports.field._FORMAT_TO_WRITER`.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import dolfinx
import io4dolfinx
from dolfinx import fem, io

__all__ = [
    "CheckpointFieldWriter",
    "FieldWriter",
    "VTKHDFFieldWriter",
    "VTXFieldWriter",
    "XDMFFieldWriter",
]


@contextmanager
def _named(functions: list[fem.Function], names: list[str]):
    """Temporarily rename functions.

    Some io4dolfinx writers take the dataset name from ``u.name`` rather than from an
    argument, so we rename around the call and restore afterwards. Mutating ``u.name``
    permanently would change the name of the same function in other exports.
    """
    original = [u.name for u in functions]
    for u, name in zip(functions, names, strict=True):
        u.name = name
    try:
        yield
    finally:
        for u, name in zip(functions, original, strict=True):
            u.name = name


class FieldWriter(ABC):
    """Writes functions to a file at successive times.

    Args:
        filename: the file to write to

    Attributes:
        filename: the file to write to
    """

    filename: Path

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)

    @abstractmethod
    def initialise(
        self,
        functions: list[fem.Function],
        names: list[str],
        mesh: dolfinx.mesh.Mesh,
        block_name: str = "mesh",
        overwrite: bool = True,
    ) -> None:
        """Prepare the file and record what will be written.

        Args:
            functions: the functions to write at every call to :meth:`write`
            names: the name to store each function under, one per function
            mesh: the mesh the functions live on
            block_name: name of the block for formats that hold several meshes in one
                file. Ignored by formats that don't.
            overwrite: if `True`, truncate an existing file. Set to `False` when
                another export has already initialised this file.
        """

    @abstractmethod
    def write(self, t: float) -> None:
        """Write the current values of the functions at time `t`."""

    def close(self) -> None:
        """Release the file. No-op for writers that open the file per call."""


class VTXFieldWriter(FieldWriter):
    """Writes ``.bp`` files with :class:`dolfinx.io.VTXWriter`, readable by ParaView."""

    _writer: io.VTXWriter | None = None

    def initialise(self, functions, names, mesh, block_name="mesh", overwrite=True):
        self._writer = io.VTXWriter(
            comm=functions[0].function_space.mesh.comm,
            filename=self.filename,
            output=functions,
            engine="BP5",
        )

    def write(self, t: float):
        self._writer.write(t)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class XDMFFieldWriter(FieldWriter):
    """Writes ``.xdmf`` (+ ``.h5``) files with :class:`dolfinx.io.XDMFFile`."""

    _writer: io.XDMFFile | None = None

    def initialise(self, functions, names, mesh, block_name="mesh", overwrite=True):
        self._functions = functions
        self._writer = io.XDMFFile(mesh.comm, self.filename, "w")
        self._writer.write_mesh(mesh)

    def write(self, t: float):
        for u in self._functions:
            self._writer.write_function(u, t)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class VTKHDFFieldWriter(FieldWriter):
    """Writes ``.vtkhdf`` files with io4dolfinx, readable by ParaView.

    A single file is a ``MultiBlockDataSet``, so several exports pointing at the same
    filename land as separate blocks: one file for a whole multi-material model instead
    of one per subdomain. Values are interpolated onto the mesh geometry nodes.
    """

    def initialise(self, functions, names, mesh, block_name="mesh", overwrite=True):
        self._check_h5py_mpi(mesh.comm)
        self._functions = functions
        self._names = names
        self._backend_args = {"name": block_name}
        io4dolfinx.write_mesh(
            filename=self.filename,
            mesh=mesh,
            backend="vtkhdf",
            backend_args=self._backend_args,
            mode=io4dolfinx.FileMode.write if overwrite else io4dolfinx.FileMode.append,
        )

    def write(self, t: float):
        # io4dolfinx takes the dataset name from u.name, not from an argument
        with _named(self._functions, self._names):
            for u in self._functions:
                io4dolfinx.write_point_data(
                    filename=self.filename,
                    u=u,
                    time=t,
                    mode=io4dolfinx.FileMode.append,
                    backend_args=self._backend_args,
                    backend="vtkhdf",
                )

    @staticmethod
    def _check_h5py_mpi(comm):
        """Fail early with an actionable message on a serial h5py in parallel.

        The vtkhdf backend writes through h5py; without MPI support io4dolfinx raises a
        bare ValueError from inside a context manager, which is hard to act on.
        """
        if comm.size == 1:
            return
        import h5py

        if not h5py.get_config().mpi:
            raise RuntimeError(
                "Writing .vtkhdf files on more than one process requires h5py built "
                f"with MPI support, but the installed h5py is serial (running on "
                f"{comm.size} processes). Install the MPI build with "
                '`conda install -c conda-forge "h5py=*=mpi_*"`, or export in serial, '
                'or use format="vtx" instead.'
            )


class CheckpointFieldWriter(FieldWriter):
    """Writes io4dolfinx checkpoints, for reloading rather than for visualisation.

    Unlike the visualisation formats, the function is stored in its own function space,
    so it can be read back exactly with
    :func:`festim.initial_condition.read_function_from_file` -- on any number of
    processes, not just the one it was written on. Not readable by ParaView.

    Args:
        filename: the file to write to
        backend: ``"adios2"`` (``.bp``) or ``"h5py"`` (``.h5``)
    """

    def __init__(self, filename: str | Path, backend: str = "adios2"):
        super().__init__(filename)
        self.backend = backend

    def initialise(self, functions, names, mesh, block_name="mesh", overwrite=True):
        self._functions = functions
        self._names = names
        io4dolfinx.write_mesh(
            filename=self.filename,
            mesh=mesh,
            backend=self.backend,
        )

    def write(self, t: float):
        for u, name in zip(self._functions, self._names, strict=True):
            io4dolfinx.write_function(
                filename=self.filename,
                u=u,
                time=t,
                name=name,
                backend=self.backend,
            )

from pathlib import Path

import mpi4py

from dolfinx.io import XDMFFile

from festim.species import Species

from .vtx import ExportBaseClass


class XDMFExport(ExportBaseClass):
    """Export functions to XDMFfile

    Args:
        filename: The name of the output file
        field: The field(s) to export

    Attributes:
        _writer (dolfinx.io.XDMFFile): the XDMF writer
        _field (festim.Species, list of festim.Species): the field index to export
    """

    _mesh_written: bool
    _filename: Path
    _writer: XDMFFile | None

    def __init__(self, filename: str | Path, field: list[Species] | Species) -> None:
        # Initializes the writer
        self._writer = None
        super().__init__(filename, ".xdmf")
        self.field = field
        self._mesh_written = False

    @property
    def field(self) -> list[Species]:
        return self._field

    @field.setter
    def field(self, value: Species | list[Species]):
        # check that field is festim.Species or list of festim.Species
        if isinstance(value, list):
            for element in value:
                if not isinstance(element, Species):
                    raise TypeError(
                        f"Each element in the list must be a species, got {type(element)}."
                    )
            val = value
        elif isinstance(value, Species):
            val = [value]
        else:
            raise TypeError(
                f"field must be of type festim.Species or a list of festim.Species, got "
                f"{type(value)}."
            )
        self._field = val

    def define_writer(self, comm: mpi4py.MPI.Intracomm) -> None:
        """Define the writer

        Args:
            comm (mpi4py.MPI.Intracomm): the MPI communicator
        """
        self._writer = XDMFFile(comm, self.filename, "w")

    def write(self, t: float):
        """Write functions to VTX file

        Args:
            t (float): the time of export
        """
        if not self._mesh_written:
            self._writer.write_mesh(
                self.field[0].post_processing_solution.function_space.mesh
            )
            self._mesh_written = True

        for field in self.field:
            self._writer.write_function(field.post_processing_solution, t)

    def __del__(self):
        if self._writer is not None:
            self._writer.close()

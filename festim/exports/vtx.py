from dolfinx.io import VTXWriter
import mpi4py

import festim as F


class VTXExport:
    """Export functions to VTX file

    Args:
        filename (str): the name of the output file
        field (int): the field index to export

    Attributes:
        filename (str): the name of the output file
        writer (dolfinx.io.VTXWriter): the VTX writer
        field (festim.Species, list of festim.Species): the field index to export

    Usage:
        >>> u = dolfinx.fem.Function(V)
        >>> my_export = festim.VTXExport("my_export.bp")
        >>> my_export.define_writer(mesh.comm, [u])
        >>> for t in range(10):
        ...    u.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
        ...    my_export.write(t)
    """

    def __init__(self, filename: str, field) -> None:
        self.filename = filename
        self.field = field

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise TypeError("filename must be of type str")
        if not value.endswith(".bp"):
            raise ValueError("filename must end with .bp")
        self._filename = value

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        # check that field is festim.Species or list of festim.Species
        if not isinstance(value, F.Species) and not isinstance(value, list):
            raise TypeError(
                "field must be of type festim.Species or list of festim.Species"
            )
        # check that all elements of list are festim.Species
        if isinstance(value, list):
            for element in value:
                if not isinstance(element, F.Species):
                    raise TypeError(
                        "field must be of type festim.Species or list of festim.Species"
                    )
        # if field is festim.Species, convert to list
        if not isinstance(value, list):
            value = [value]

        self._field = value

    def define_writer(self, comm: mpi4py.MPI.Intracomm, functions: list) -> None:
        """Define the writer

        Args:
            comm (mpi4py.MPI.Intracomm): the MPI communicator
            functions (list): the list of functions to export
        """
        self.writer = VTXWriter(comm, self.filename, functions, "BP4")

    def write(self, t: float):
        """Write functions to VTX file

        Args:
            t (float): the time of export
        """
        self.writer.write(t)

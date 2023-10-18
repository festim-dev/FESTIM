import mpi4py
from dolfinx.io import XDMFFile
import festim as F


class XDMFExport:
    """Export functions to XDMFfile

    Args:
        filename (str): the name of the output file
        field (int or festim.Species): the field index to export

    Attributes:
        filename (str): the name of the output file
        writer (dolfinx.io.XDMFFile): the XDMF writer
        field (festim.Species, list of festim.Species): the field index to export
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
        if not value.endswith(".xdmf"):
            raise ValueError("filename must end with .xdmf")
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

    def define_writer(self, comm: mpi4py.MPI.Intracomm) -> None:
        """Define the writer

        Args:
            comm (mpi4py.MPI.Intracomm): the MPI communicator
        """
        self.writer = XDMFFile(comm, self.filename, "w")

    def write(self, t: float):
        """Write functions to VTX file

        Args:
            t (float): the time of export
        """
        for field in self.field:
            self.writer.write_function(field.solution, t)

from dolfinx.io import VTXWriter
import mpi4py


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

    def __init__(self, filename: str, field=0) -> None:
        self.filename = filename
        if not isinstance(field, list):
            field = [field]
        self.field = field

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

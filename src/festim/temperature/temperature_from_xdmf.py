from festim.temperature.temperature import Temperature
from festim.helpers import extract_xdmf_labels
import fenics as f


class TemperatureFromXDMF(Temperature):
    """
    Temperature read from an XDMF file

    Args:
        filename (str): The temperature file. Must end in ".xdmf"
        label (str): How the checkpoints have been labelled

    Attributes:
        filename (str): name of the temperature file
        label (str): How the checkpoints have been labelled
    """

    def __init__(self, filename, label) -> None:
        super().__init__()

        self.filename = filename
        self.label = label

        # check labels match
        if self.label not in extract_xdmf_labels(self.filename):
            raise ValueError(
                "Coudln't find label: {} in {}".format(self.label, self.filename)
            )

    def create_functions(self, mesh):
        """Creates functions self.T, self.T_n
        Args:
            mesh (festim.Mesh): the mesh
        """
        V = f.FunctionSpace(mesh.mesh, "CG", 1)
        self.T = f.Function(V, name="T")

        f.XDMFFile(self.filename).read_checkpoint(self.T, self.label, -1)

        self.T_n = f.Function(V, name="T_n")
        self.T_n.assign(self.T)

    def update(self, t):
        """Allows for the use of this class in transient h transport cases,
        refer to issue #499
        """
        pass

from festim.exports.xdmf_export import XDMFExport
import fenics as f


class TrapDensityXDMF(XDMFExport):
    """
    Args:
        trap (festim.Trap): the trap to export density
        kwargs (): See XDMFExport
    """

    def __init__(self, trap, **kwargs) -> None:

        super().__init__(
            field="1", **kwargs
        )  # field is "1" just to make the code not crash

        self.trap = trap

    def write(self, t):
        """Writes to file

        Args:
            t (float): the time
        """
        functionspace = self.function.function_space().collapse()
        density_as_function = f.project(self.trap.density[0], functionspace)
        self.function = density_as_function
        super().write(t)

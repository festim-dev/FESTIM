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

    def write(self, t, dx):
        """Writes to file

        Args:
            t (float): the time
            dx (fenics.Measure): the measure for dx
        """
        functionspace = self.function.function_space().collapse()
        u = f.Function(functionspace)
        v = f.TestFunction(functionspace)
        F = f.inner(u, v) * dx

        for mat in self.trap.materials:
            F -= f.inner(self.trap.density[0], v) * dx(mat.id)

        f.solve(F == 0, u, bcs=[])
        self.function = u

        super().write(t)

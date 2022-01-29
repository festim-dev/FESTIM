from FESTIM import R, DerivedQuantity
import fenics as f
import numpy as np


class SurfaceFlux(DerivedQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field=field)
        self.surface = surface
        self.title = "Flux surface {}: {}".format(self.surface, self.field)

    def compute(self, soret=False):
        field_to_prop = {
            "0": self.D,
            "solute": self.D,
            0: self.D,
            "T": self.thermal_cond
        }
        self.prop = field_to_prop[self.field]
        flux = f.assemble(self.prop*f.dot(f.grad(self.function), self.n)*self.ds(self.surface))
        if soret and self.field in [0, "0", "solute"]:
            flux += f.assemble(
                        self.prop*self.function*self.H/(R*self.T**2)*f.dot(f.grad(self.T), self.n)*self.ds(self.surface))
        return flux


class HydrogenFlux(SurfaceFlux):
    def __init__(self, surface) -> None:
        super().__init__(field="solute", surface=surface)


class ThermalFlux(SurfaceFlux):
    def __init__(self, surface) -> None:
        super().__init__(field="T", surface=surface)


class AverageVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Average {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function*self.dx(self.volume))/f.assemble(1*self.dx(self.volume))


class MinimumVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Minimum {} volume {}".format(self.field, self.volume)

    def compute(self, volume_markers):
        '''Minimum of f over subdomains cells marked with self.volume'''
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(np.hstack(
            [dm.cell_dofs(c.index())
             for c in f.SubsetIterator(volume_markers, self.volume)]))

        return np.min(self.function.vector().get_local()[subd_dofs])


class MaximumVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Maximum {} volume {}".format(self.field, self.volume)

    def compute(self, volume_markers):
        '''Minimum of f over subdomains cells marked with self.volume'''
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(np.hstack(
            [dm.cell_dofs(c.index())
             for c in f.SubsetIterator(volume_markers, self.volume)]))

        return np.max(self.function.vector().get_local()[subd_dofs])


class TotalVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Total {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function*self.dx(self.volume))


class TotalSurface(DerivedQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field)
        self.surface = surface
        self.title = "Total {} surface {}".format(self.field, self.surface)

    def compute(self):
        return f.assemble(self.function*self.ds(self.surface))

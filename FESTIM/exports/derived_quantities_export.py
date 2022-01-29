from FESTIM import DerivedQuantity
import fenics as f
import numpy as np


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

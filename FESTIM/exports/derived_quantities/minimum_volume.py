from festim import VolumeQuantity
import fenics as f
import numpy as np


class MinimumVolume(VolumeQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field, volume=volume)
        self.title = "Minimum {} volume {}".format(self.field, self.volume)

    def compute(self, volume_markers):
        """Minimum of f over subdomains cells marked with self.volume"""
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(
            np.hstack(
                [
                    dm.cell_dofs(c.index())
                    for c in f.SubsetIterator(volume_markers, self.volume)
                ]
            )
        )

        return np.min(self.function.vector().get_local()[subd_dofs])

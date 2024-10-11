import festim as F
import numpy as np


class MinimumSurface(F.SurfaceQuantity):
    """Computes the minimum value of a field on a given surface

    Args:
        field (festim.Species): species for which the minimum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the minimum surface is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Minimum {self.field.name} surface {self.surface.id}"

    def compute(self):
        """
        Computes the minimum value of the field on the defined surface
        subdomain, and appends it to the data list
        """
        self.value = np.min(self.field.solution.x.array[self.surface.indices])
        self.data.append(self.value)
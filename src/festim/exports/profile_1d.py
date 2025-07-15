import numpy as np


class Profile1DExport:

    def __init__(self, field, subdomain=None):
        self.field = field
        self.data = []
        self.x = []
        self.subdomain = subdomain

        self._dofs = None
        self._sort_coords = None


def compute_profile(u, index):
    V0, dofs = u.function_space.sub(index).collapse()
    coords = V0.tabulate_dof_coordinates()[:, 0]
    sort_coords = np.argsort(coords)
    c = u.x.array[dofs][sort_coords]
    x = coords[sort_coords]

    return x, c

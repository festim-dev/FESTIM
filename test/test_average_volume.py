import numpy as np
import ufl
from dolfinx import fem

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy")


def test_average_volume_compute_1D():
    """Test that the average volume export computes the correct value"""

    # BUILD
    L = 6.0
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=dummy_mat)
    temp, cell_meshtags = my_mesh.define_meshtags(
        surface_subdomains=[], volume_subdomains=[dummy_volume]
    )
    dx = ufl.Measure("dx", domain=my_mesh.mesh, subdomain_data=cell_meshtags)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: x[0] * 0.5 + 2)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.AverageVolume(field=my_species, volume=dummy_volume)

    # RUN
    my_export.compute(my_species.solution, dx)

    # TEST
    expected_value = 3.5
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

import numpy as np
from dolfinx import fem

import festim as F


def test_minimum_volume_compute_1D():
    """Test that the minimum volume export computes the right value"""

    # BUILD
    L = 6
    dummy_material = F.Material(D_0=1.5, E_D=1, name="dummy")
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=dummy_material)

    # create mesh tags
    ft, ct = my_mesh.define_meshtags(
        surface_subdomains=[F.SurfaceSubdomain1D(id=1, x=0)],
        volume_subdomains=[dummy_volume],
        interfaces=None,
    )

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: (x[0] - 2) ** 2 + 0.5)

    my_species = F.Species("H")
    my_species.post_processing_solution = c

    my_export = F.MinimumVolume(field=my_species, volume=dummy_volume)
    my_export.volume_meshtags = ct

    # RUN
    my_export.compute()

    # TEST
    expected_value = 0.5
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

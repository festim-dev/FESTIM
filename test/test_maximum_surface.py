import numpy as np
from dolfinx import fem

import festim as F


def test_maximum_surface_compute_1D():
    """Test that the maximum surface export computes the correct value"""

    # BUILD
    L = 4.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)

    # create mesh tags
    ft, ct = my_mesh.define_meshtags(
        surface_subdomains=[dummy_surface],
        volume_subdomains=[
            F.VolumeSubdomain1D(id=1, material=F.Material(D_0=1, E_D=0), borders=[0, L])
        ],
        interfaces=None,
    )

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: (x[0] - 3) ** 2)

    my_species = F.Species("H")
    my_species.post_processing_solution = c

    my_export = F.MaximumSurface(field=my_species, surface=dummy_surface)
    my_export.D = D
    my_export.facet_meshtags = ft

    # RUN
    my_export.compute()

    # TEST
    expected_value = 1.0
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

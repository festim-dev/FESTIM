import festim as F
import numpy as np
import ufl
from dolfinx.mesh import meshtags
from dolfinx import fem


def test_total_surface_compute():
    """Test that the total surface export computes the correct value"""

    # BUILD
    L = 6
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=6)

    # define mesh ds measure
    facet_indices = np.array(
        dummy_surface.locate_boundary_facet_indices(my_mesh.mesh, 0),
        dtype=np.int32,
    )
    tags_facets = np.array(
        [1],
        dtype=np.int32,
    )
    facet_meshtags = meshtags(my_mesh.mesh, 0, facet_indices, tags_facets)
    ds = ufl.Measure("ds", domain=my_mesh.mesh, subdomain_data=facet_meshtags)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("CG", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: x[0] * 0.5 + 1)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.TotalSurface(field=my_species, surface=dummy_surface)
    my_export.D = D

    # RUN
    my_export.compute(ds)

    # TEST
    expected_value = 15
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

import festim as F
import numpy as np
from dolfinx import fem
import ufl
from dolfinx.mesh import meshtags
import pytest


def test_average_surface_compute_1D():
    """Test that the average surface export computes the correct value"""

    # BUILD
    L = 6.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=6)
    dummy_surface.locate_boundary_facet_indices(mesh=my_mesh.mesh, fdim=my_mesh.fdim)

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
    c.interpolate(lambda x: x[0] * 0.5 + 2)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.AverageSurface(field=my_species, surface=dummy_surface)
    my_export.D = D

    # RUN
    my_export.compute(ds)

    # TEST
    expected_value = 5.0
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

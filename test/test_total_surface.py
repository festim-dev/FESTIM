import festim as F
import numpy as np
from dolfinx import fem

def test_total_surface_compmute():
    """Test that the total surface export computes the correct value"""

    # BUILD 
    L = 6
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0,L,10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=6)

    #give function to species
    V = fem.functionsapce(my_mesh.mesh, ("CG", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: x[0]*0.5 + 1)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.TotalSurface(field=my_species, surface=dummy_surface)
    my_export.D = D

    # RUN 
    my_export.compute()

    # TEST
    expected_value = 15
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)
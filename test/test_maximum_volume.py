import festim as F
import numpy as np
from dolfinx.mesh import meshtags
import ufl
from dolfinx import fem


def test_maximum_volume_compute():
    """Test that the maximum volume export computes the right value"""

    # BUILD
    L = 6
    dummy_material = F.Material(D_0=1.5, E_D=1, name="dummy")
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=dummy_material)

    # define mesh dx measure
    num_cells = my_mesh.mesh.topology.index_map(my_mesh.vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    tags_volumes = np.full(num_cells, 1, dtype=np.int32)
    cell_meshtags = meshtags(
        my_mesh.mesh, my_mesh.vdim, mesh_cell_indices, tags_volumes
    )
    dx = ufl.Measure("dx", domain=my_mesh.mesh, subdomain_data=cell_meshtags)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("CG", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: (x[0] - 1) ** 2 + 2)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.MaximumVolume(field=my_species, volume=dummy_volume)

    # RUN
    my_export.compute()

    # TEST
    expected_value = 27
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)

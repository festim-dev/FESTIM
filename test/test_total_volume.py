import os

import numpy as np
import pytest
import ufl
from dolfinx import fem

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy")


def test_total_volume_export_compute():
    """Test that the total volume export computes the correct value"""

    # BUILD
    L = 4.0
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=dummy_mat)
    temp, cell_meshtags = my_mesh.define_meshtags(
        surface_subdomains=[], volume_subdomains=[dummy_volume]
    )
    dx = ufl.Measure("dx", domain=my_mesh.mesh, subdomain_data=cell_meshtags)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: 2 * x[0] ** 2 + 1)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.TotalVolume(
        field=my_species,
        volume=dummy_volume,
    )

    # RUN
    my_export.compute(my_species.solution, dx=dx)

    # TEST
    # total = int[0,L] c dx = 2/3 * L^3 + L
    expected_value = 2 / 3 * L**3 + L
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)


@pytest.mark.parametrize("value", ["my_export.csv", "my_export.txt"])
def test_title_generation(tmp_path, value):
    """Test that the title is made to be written to the header in a csv or txt file"""
    my_export = F.TotalVolume(
        filename=os.path.join(tmp_path, f"{value}"),
        field=F.Species("TEST"),
        volume=F.VolumeSubdomain1D(id=35, borders=[0, 1], material=dummy_mat),
    )
    my_export.value = 2.0
    my_export.write(0)
    title = np.genfromtxt(my_export.filename, delimiter=",", max_rows=1, dtype=str)

    expected_title = "Total TEST volume 35"

    assert title[1] == expected_title

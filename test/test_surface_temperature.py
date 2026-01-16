import numpy as np
import ufl
from dolfinx import fem
import pytest
import os

import festim as F


@pytest.mark.parametrize(
    "T_function, expected_values",
    [
        (3, 3),
        (lambda t: t, 3.0),
        (lambda x, t: 1.0 + x[0] + t, 10.0),
    ],
)
def test_surface_temperature_compute_1D(T_function, expected_values):
    """Test that the average surface temperature export computes the correct value."""

    # BUILD
    L = 6.0
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=L)
    dummy_volume = F.VolumeSubdomain1D(
        id=1, borders=(0.0, L), material=F.Material(D_0=1, E_D=1, name="dummy")
    )

    facet_meshtags, temp = my_mesh.define_meshtags(
        surface_subdomains=[dummy_surface], volume_subdomains=[dummy_volume]
    )
    ds = ufl.Measure("ds", domain=my_mesh.mesh, subdomain_data=facet_meshtags)

    dt = F.Stepsize(initial_value=1)
    settings = F.Settings(atol=1e05, rtol=1e-10, stepsize=dt, final_time=10)
    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh, temperature=T_function, settings=settings
    )

    my_model.species = [F.Species("H")]
    my_model.subdomains = [
        dummy_surface,
        dummy_volume,
    ]

    my_model.t = fem.Constant(my_mesh.mesh, 0.0)
    dt = fem.Constant(my_mesh.mesh, 1.0)

    my_model.define_temperature()

    my_export = F.AverageSurfaceTemperature(surface=dummy_surface)
    my_export.temperature_field = my_model.temperature_fenics
    my_model.exports = [my_export]

    # RUN
    for _ in range(3):
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

    my_model.initialise()
    my_model.run()

    # TEST
    assert np.isclose(my_export.value, expected_values)


def test_title(tmp_path):
    surf_1 = F.SurfaceSubdomain(id=1)
    results = "test.csv"
    temp = 400
    surface_temp = F.AverageSurfaceTemperature(surface=surf_1, filename=results)

    my_model = F.HydrogenTransportProblem(
        temperature=temp,
    )
    surface_temp.filename = os.path.join(tmp_path, "test.csv")
    surface_temp.value = 1

    assert surface_temp.title == "Temperature surface 1"


@pytest.mark.parametrize("value", ["my_export.csv", "my_export.txt"])
def test_title_generation(tmp_path, value):
    """Test that the title is made to be written to the header in a csv or txt file"""
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 6.0, 10000)), temperature=500
    )
    my_model.define_temperature()

    my_export = F.AverageSurfaceTemperature(
        filename=os.path.join(tmp_path, f"{value}"),
        surface=F.SurfaceSubdomain1D(id=35, x=1),
    )
    my_export.value = 2.0
    my_export.write(0)
    title = np.genfromtxt(my_export.filename, delimiter=",", max_rows=1, dtype=str)

    expected_title = "Temperature surface 35"

    assert title[1] == expected_title


def test_not_implemented_error_raised_with_initialise_exports_multiple_subdomains():
    """Test that NotImplementedError is raised for problems with multiple volume domains."""

    # BUILD
    L = 1.0
    test_mesh = F.Mesh1D(vertices=np.linspace(0, L, num=101))

    my_mat = F.Material(D_0=1, E_D=1)
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=L)
    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=my_mat)
    vol_2 = F.VolumeSubdomain1D(id=1, borders=[0.5, L], material=my_mat)

    H = F.Species("H")
    my_model = F.HydrogenTransportProblemDiscontinuous(
        mesh=test_mesh,
        temperature=10,
        subdomains=[dummy_surface, vol_1, vol_2],
        species=[F.Species("H")],
    )

    my_model.exports = [F.AverageSurfaceTemperature(surface=dummy_surface)]

    # TEST
    with pytest.raises(NotImplementedError):
        my_model.initialise_exports()

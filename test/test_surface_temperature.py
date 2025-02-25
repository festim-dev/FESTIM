import numpy as np
import ufl
from dolfinx import fem
import pytest

import festim as F

@pytest.mark.parametrize(
    "T_function, expected_values",
    [
        (3, 3),
        (lambda t: t, 2.5),
    ],
)

def test_average_surface_temperature_compute_1D(T_function, expected_values):
    """Test that the average surface temperature export computes the correct value"""

    # BUILD
    L = 6.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=6)
    dummy_volume = F.VolumeSubdomain1D(
        id=1, borders=[0, L], material=F.Material(D_0=1, E_D=1, name="dummy")
    )
    facet_meshtags, temp = my_mesh.define_meshtags(
        surface_subdomains=[dummy_surface], volume_subdomains=[dummy_volume]
    )

    ds = ufl.Measure("ds", domain=my_mesh.mesh, subdomain_data=facet_meshtags)

    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh,
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(my_mesh.mesh, 1.0)

    my_model.temperature = T_function

    my_model.define_temperature()

    # RUN
    for i in range(3):
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

    my_export = F.SurfaceTemperature(temperature_field=my_model.temperature, surface=dummy_surface)
    my_export.compute(ds)
    my_export = F.SurfaceTemperature(temperature_field=T_function, surface=dummy_surface)

    # TEST
    for i in range(0,2):
        assert np.isclose(my_export.value, expected_values[i])
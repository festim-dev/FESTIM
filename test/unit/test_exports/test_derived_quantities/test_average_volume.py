from festim import AverageVolume, AverageVolumeCylindrical, AverageVolumeSpherical
import fenics as f
import pytest
import numpy as np
import math


@pytest.mark.parametrize("field,volume", [("solute", 1), ("T", 2)])
def test_title(field, volume):
    """
    A simple test to check that the title is set
    correctly in festim.AverageVolume

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
    """

    my_average = AverageVolume(field, volume)
    assert my_average.title == "Average {} volume {}".format(field, volume)


class TestCompute:
    """Test that the average volume export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    volume_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(volume_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    volume = 1
    my_average = AverageVolume("solute", volume)
    my_average.function = c
    my_average.dx = dx

    def test_h_average(self):
        expected = f.assemble(self.c * self.dx(self.volume)) / f.assemble(
            1 * self.dx(self.volume)
        )
        computed = self.my_average.compute()
        assert computed == expected


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("r0", [0, 2])
@pytest.mark.parametrize("height", [2, 3])
@pytest.mark.parametrize("c_top", [2, 3])
@pytest.mark.parametrize("c_bottom", [2, 3])
def test_compute_cylindrical(r0, radius, height, c_top, c_bottom):
    """
    Test that AverageVolumeCylindrical computes the flux correctly on a hollow cylinder

    Args:
        r0 (float): internal radius
        radius (float): cylinder radius
        height (float): cylinder height
        c_top (float): concentration top
        c_bottom (float): concentration bottom
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0 = 0
    z1 = z0 + height
    mesh_fenics = f.RectangleMesh(f.Point(r0, z0), f.Point(r1, z1), 10, 10)

    # marking physical groups (volumes and surfaces)
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(1)

    dx = f.Measure("dx", domain=mesh_fenics, subdomain_data=volume_markers)

    my_avg_vol = AverageVolumeCylindrical("solute", volume=1)

    V = f.FunctionSpace(mesh_fenics, "P", 1)
    expr = f.Expression(
        f"{c_bottom} + {(c_top - c_bottom)/(z1-z0)} * x[1]",
        degree=1,
    )
    # expr = f.Expression(
    #     "2 * x[0] * x[1] * 2",
    #     degree=1,
    # )
    my_avg_vol.function = f.interpolate(expr, V)
    my_avg_vol.dx = dx

    computed_value = my_avg_vol.compute()
    expected_value = (c_bottom - c_top) / height * (0.5 * r1**2 - 0.5 * r0**2)
    assert np.isclose(computed_value, expected_value)


# @pytest.mark.parametrize("r0", [1, 2])
# @pytest.mark.parametrize("radius", [1, 2])
# @pytest.mark.parametrize("c_left", [3, 4])
# @pytest.mark.parametrize("c_right", [1, 2])
# def test_compute_spherical(r0, radius, c_left, c_right):
#     """
#     Test that SurfaceFluxSpherical computes the flux correctly on a hollow sphere

#     Args:
#         r0 (float): internal radius
#         radius (float): sphere radius
#         c_left (float): concentration left
#         c_right (float): concentration right
#     """
#     # creating a mesh with FEniCS
#     r1 = r0 + radius
#     mesh_fenics = f.IntervalMesh(10, r0, r1)

#     # marking physical groups (volumes and surfaces)
#     volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
#     volume_markers.set_all(1)

#     left_surface = f.CompiledSubDomain(
#         f"on_boundary && near(x[0], {r0}, tol)", tol=1e-14
#     )
#     right_surface = f.CompiledSubDomain(
#         f"on_boundary && near(x[0], {r1}, tol)", tol=1e-14
#     )

#     surface_markers = f.MeshFunction(
#         "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
#     )
#     surface_markers.set_all(0)
#     # Surface ids
#     left_id = 1
#     right_id = 2
#     left_surface.mark(surface_markers, left_id)
#     right_surface.mark(surface_markers, right_id)
#     ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)

#     my_flux = SurfaceFluxSpherical("solute", right_id)
#     my_flux.D = f.Constant(2)
#     V = f.FunctionSpace(mesh_fenics, "P", 1)
#     expr = f.Expression(
#         f"{c_left} + {(c_right - c_left)/(r1-r0)} * x[0]",
#         degree=1,
#     )
#     my_flux.function = f.interpolate(expr, V)

#     my_flux.n = f.FacetNormal(mesh_fenics)
#     my_flux.ds = ds

#     computed_value = my_flux.compute()
#     # expected value is the integral of the flux over the surface
#     flux_value = float(my_flux.D) * (c_left - c_right) / (r1 - r0)
#     expected_value = -4 * math.pi * flux_value * r1**2
#     assert np.isclose(computed_value, expected_value)


# @pytest.mark.parametrize(
#     "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
# )
# def test_azimuthal_range_cylindrical(azimuth_range):
#     """
#     Tests that an error is raised when the azimuthal range is out of bounds
#     """
#     with pytest.raises(ValueError):
#         SurfaceFluxCylindrical("solute", 1, azimuth_range=azimuth_range)


# def test_soret_raises_error():
#     """
#     Tests that an error is raised when the Soret effect is used with SurfaceFluxCylindrical
#     """
#     my_flux = SurfaceFluxCylindrical("T", 1)
#     with pytest.raises(NotImplementedError):
#         my_flux.compute(soret=True)


# @pytest.mark.parametrize(
#     "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
# )
# def test_azimuthal_range_spherical(azimuth_range):
#     """
#     Tests that an error is raised when the azimuthal range is out of bounds
#     """
#     with pytest.raises(ValueError):
#         SurfaceFluxSpherical("solute", 1, azimuth_range=azimuth_range)


# @pytest.mark.parametrize(
#     "polar_range", [(0, 2 * np.pi), (-2 * np.pi, 0), (-2 * np.pi, 3 * np.pi)]
# )
# def test_polar_range_spherical(polar_range):
#     """
#     Tests that an error is raised when the polar range is out of bounds
#     """
#     with pytest.raises(ValueError):
#         SurfaceFluxSpherical("solute", 1, polar_range=polar_range)


# def test_soret_raises_error_spherical():
#     """
#     Tests that an error is raised when the Soret effect is used with SurfaceFluxSpherical
#     """
#     my_flux = SurfaceFluxSpherical("T", 1)
#     with pytest.raises(NotImplementedError):
#         my_flux.compute(soret=True)


# @pytest.mark.parametrize(
#     "function, field, expected_title",
#     [
#         (c_1D, "solute", "solute flux surface 3 (H m-2 s-1)"),
#         (c_1D, "T", "Heat flux surface 3 (W m-2)"),
#         (c_2D, "solute", "solute flux surface 3 (H m-1 s-1)"),
#         (c_2D, "T", "Heat flux surface 3 (W m-1)"),
#         (c_3D, "solute", "solute flux surface 3 (H s-1)"),
#         (c_3D, "T", "Heat flux surface 3 (W)"),
#     ],
# )
# def test_title_with_units(function, field, expected_title):
#     my_flux = SurfaceFlux(field=field, surface=3)
#     my_flux.function = function
#     my_flux.show_units = True

#     assert my_flux.title == expected_title


# def test_cylindrical_flux_title_no_units_solute():
#     """A simple test to check that the title is set correctly in
#     festim.CylindricalSurfaceFlux with a solute field without units"""

#     my_h_flux = SurfaceFluxCylindrical("solute", 2)
#     assert my_h_flux.title == "solute flux surface 2"


# def test_cylindrical_flux_title_no_units_temperature():
#     """A simple test to check that the title is set correctly in
#     festim.CylindricalSurfaceFlux with a T field without units"""

#     my_heat_flux = SurfaceFluxCylindrical("T", 4)
#     assert my_heat_flux.title == "Heat flux surface 4"


# def test_spherical_flux_title_no_units_solute():
#     """A simple test to check that the title is set correctly in
#     festim.SphericalSurfaceFlux with a solute field without units"""

#     my_h_flux = SurfaceFluxSpherical("solute", 3)
#     assert my_h_flux.title == "solute flux surface 3"


# def test_spherical_flux_title_no_units_temperature():
#     """A simple test to check that the title is set correctly in
#     festim.CSphericalSurfaceFlux with a T field without units"""

#     my_heat_flux = SurfaceFluxSpherical("T", 5)
#     assert my_heat_flux.title == "Heat flux surface 5"

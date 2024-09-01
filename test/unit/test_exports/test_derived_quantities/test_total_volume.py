from festim import TotalVolume, TotalVolumeCylindrical
import fenics as f
import pytest
from .tools import c_1D, c_2D, c_3D
import pytest
import numpy as np


@pytest.mark.parametrize("field,volume", [("solute", 1), ("T", 2)])
def test_title(field, volume):
    """
    A simple test to check that the title is set
    correctly in festim.TotalVolume

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_total = TotalVolume(field, volume)
    assert my_total.title == "Total {} volume {}".format(field, volume)


class TestCompute:
    """Test that the total volume export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    volume_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(volume_markers, 2)

    dx = f.Measure("dx", domain=mesh, subdomain_data=volume_markers)

    volume = 1
    my_total = TotalVolume("solute", volume)
    my_total.function = c
    my_total.dx = dx

    def test_minimum(self):
        expected = f.assemble(self.c * self.dx(self.volume))

        produced = self.my_total.compute()
        assert produced == expected


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("r0", [0, 2])
@pytest.mark.parametrize("height", [2, 3])
@pytest.mark.parametrize("azimuth_range", [(0, np.pi), (0, np.pi/2)])
def test_compute_cylindrical(r0, radius, height, azimuth_range):
    """
    Test that TotalVolumeCylindrical computes the volume correctly on a cylinder

    Args:
        r0 (float): internal radius
        radius (float): cylinder radius
        height (float): cylinder height
        azimuth_range (tuple): range of the azimuthal angle
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0 = 0
    z1 = z0 + height
    mesh_fenics = f.RectangleMesh(f.Point(r0, z0), f.Point(r1, z1), 10, 10)

    # marking physical groups (volumes and surfaces)
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(1)

    volume = 1
    my_total = TotalVolumeCylindrical("solute", volume, azimuth_range)

    dx = f.Measure("dx", domain=mesh_fenics, subdomain_data=volume_markers)

    V = f.FunctionSpace(mesh_fenics, "P", 1)
    r = f.interpolate(f.Expression("x[0]", degree=1), V)
    c = 2*r

    my_total.dx = dx
    my_total.function = c
    my_total.r = f.Expression("x[0]", degree=1)

    az0, az1 = azimuth_range

    expected_value = f.assemble((az1 - az0) * c * r * dx(volume))
    computed_value = my_total.compute()

    assert np.isclose(expected_value, computed_value)


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_cylindrical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        TotalVolumeCylindrical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute volume 2 (H m-2)"),
        (c_1D, "T", "Total T volume 2 (K m)"),
        (c_2D, "solute", "Total solute volume 2 (H m-1)"),
        (c_2D, "T", "Total T volume 2 (K m2)"),
        (c_3D, "solute", "Total solute volume 2 (H)"),
        (c_3D, "T", "Total T volume 2 (K m3)"),
    ],
)
def test_title_with_units(function, field, expected_title):
    my_export = TotalVolume(volume=2, field=field)
    my_export.function = function
    my_export.show_units = True

    assert my_export.title == expected_title

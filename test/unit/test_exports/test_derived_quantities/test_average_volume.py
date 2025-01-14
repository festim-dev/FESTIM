from festim import AverageVolume, AverageVolumeCylindrical, AverageVolumeSpherical, x
import fenics as f
import pytest
import numpy as np
from sympy.printing import ccode


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
    assert (
        my_average.title
        == f"Average {field} volume {volume} ({my_average.export_unit})"
    )


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


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("r0", [3, 5])
@pytest.mark.parametrize("height", [2, 7])
def test_compute_cylindrical(r0, radius, height):
    """
    Test that AverageVolumeCylindrical computes the value correctly on a hollow cylinder

    Args:
        r0 (float): internal radius
        radius (float): cylinder radius
        height (float): cylinder height
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0, z1 = 0, height

    mesh_fenics = f.RectangleMesh(f.Point(r0, z0), f.Point(r1, z1), 10, 10)

    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(1)
    dx = f.Measure("dx", domain=mesh_fenics, subdomain_data=volume_markers)

    my_export = AverageVolumeCylindrical("solute", 1)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: 3 * r
    expr = f.Expression(ccode(c_fun(x)), degree=1)
    my_export.function = f.interpolate(expr, V)
    my_export.dx = dx

    expected_value = 2 * (r1**3 - r0**3) / (r1**2 - r0**2)

    computed_value = my_export.compute()

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("r0", [3, 5])
def test_compute_spherical(r0, radius):
    """
    Test that AverageVolumeSpherical computes the average value correctly
    on a hollow sphere

    Args:
        r0 (float): internal radius
        radius (float): sphere  radius
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    mesh_fenics = f.IntervalMesh(10, r0, r1)

    # marking physical groups (volumes and surfaces)
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(1)
    dx = f.Measure("dx", domain=mesh_fenics, subdomain_data=volume_markers)

    my_export = AverageVolumeSpherical("solute", 1)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: 4 * r
    expr = f.Expression(
        ccode(c_fun(x)),
        degree=1,
    )
    my_export.function = f.interpolate(expr, V)

    my_export.dx = dx

    expected_value = 3 * (r1**4 - r0**4) / (r1**3 - r0**3)

    computed_value = my_export.compute()

    assert np.isclose(computed_value, expected_value)


def test_average_volume_cylindrical_title_no_units_solute():
    """A simple test to check that the title is set correctly in
    festim.AverageVolumeCylindrical with a solute field without units"""

    my_export = AverageVolumeCylindrical("solute", 4)
    assert my_export.title == "Average solute volume 4 (H m-3)"


def test_average_volume_cylindrical_title_no_units_temperature():
    """A simple test to check that the title is set correctly in
    festim.AverageVolumeCylindrical with a T field without units"""

    my_export = AverageVolumeCylindrical("T", 5)
    assert my_export.title == "Average T volume 5 (K)"


def test_average_volume_spherical_title_no_units_solute():
    """A simple test to check that the title is set correctly in
    festim.AverageVolumeSpherical with a solute field without units"""

    my_export = AverageVolumeSpherical("solute", 6)
    assert my_export.title == "Average solute volume 6 (H m-3)"


def test_average_volume_spherical_title_no_units_temperature():
    """A simple test to check that the title is set correctly in
    festim.AverageVolumeSpherical with a T field without units"""

    my_export = AverageVolumeSpherical("T", 9)
    assert my_export.title == "Average T volume 9 (K)"


def test_avg_vol_cylindrical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using AverageVolumeCylindrical"""

    my_export = AverageVolumeCylindrical("solute", 3)

    assert my_export.allowed_meshes == ["cylindrical"]


def test_avg_vol_spherical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using AverageVolumeSpherical"""

    my_export = AverageVolumeSpherical("solute", 5)

    assert my_export.allowed_meshes == ["spherical"]

from festim import x, y, TotalVolume, TotalVolumeCylindrical, TotalVolumeSpherical
import fenics as f
import pytest
from .tools import c_1D, c_2D, c_3D, mesh_1D, mesh_2D
from sympy.printing import ccode
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
    my_total.function = c_2D
    assert my_total.title == f"Total {field} volume {volume} ({my_total.export_unit})"


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


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("r0", [0, 1.5])
@pytest.mark.parametrize("height", [2, 3])
def test_compute_cylindrical(r0, radius, height):
    """
    Test that TotalVolumeCylindrical computes the total value of a function
    correctly on a hollow cylinder

    Args:
        r0 (float): internal radius
        radius (float): cylinder radius
        height (float): cylinder height
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0, z1 = 0, height

    mesh_fenics = f.RectangleMesh(f.Point(r0, z0), f.Point(r1, z1), 10, 10)

    volume_id = 3
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(volume_id)
    dx = f.Measure("dx", domain=mesh_fenics, subdomain_data=volume_markers)

    my_exp = TotalVolumeCylindrical("solute", volume_id)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r, z: r + z
    expr = f.Expression(
        ccode(c_fun(x, y)),
        degree=1,
    )
    my_exp.function = f.interpolate(expr, V)
    my_exp.dx = dx

    expected_value = ((np.pi * z1) / 6) * (
        (-4 * r0**3) - (3 * r0**2 * z1) + (r1**2 * (4 * r1 + 3 * z1))
    )

    computed_value = my_exp.compute()

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("radius", [1.5, 2.5])
@pytest.mark.parametrize("r0", [0, 1])
def test_compute_spherical(r0, radius):
    """
    Test that TotalVolumeSpherical computes the total value of a function
    correctly on a hollow sphere

    Args:
        r0 (float): internal radius
        radius (float): sphere radius
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius

    mesh_fenics = f.IntervalMesh(100, r0, r1)

    volume_id = 2
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(volume_id)
    dx = f.Measure("dx", domain=mesh_fenics, subdomain_data=volume_markers)

    my_exp = TotalVolumeSpherical("solute", volume_id)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: r**2
    expr = f.Expression(
        ccode(c_fun(x)),
        degree=1,
    )
    my_exp.function = f.interpolate(expr, V)
    my_exp.dx = dx

    expected_value = (4 * np.pi / 5) * (r1**5 - r0**5)

    computed_value = my_exp.compute()

    assert np.isclose(computed_value, expected_value, rtol=1e-04)


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
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_spherical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        TotalVolumeSpherical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "polar_range", [(0, 2 * np.pi), (-np.pi, 0), (-2 * np.pi, 3 * np.pi)]
)
def test_polar_range_spherical(polar_range):
    """
    Tests that an error is raised when the polar range is out of bounds
    """
    with pytest.raises(ValueError):
        TotalVolumeSpherical("solute", 1, polar_range=polar_range)


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute volume 3 (H m-1)"),
        (c_1D, "T", "Total T volume 3 (K m2)"),
        (c_2D, "solute", "Total solute volume 3 (H)"),
        (c_2D, "T", "Total T volume 3 (K m3)"),
    ],
)
def test_TotalVolumeCylindrical_title_with_units(function, field, expected_title):
    my_exp = TotalVolumeCylindrical(field=field, volume=3)
    my_exp.function = function
    my_exp.show_units = True

    assert my_exp.title == expected_title


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute volume 4 (H)"),
        (c_1D, "T", "Total T volume 4 (K m3)"),
    ],
)
def test_TotalVolumeSpherical_title_with_units(function, field, expected_title):
    my_exp = TotalVolumeSpherical(field=field, volume=4)
    my_exp.function = function
    my_exp.show_units = True

    assert my_exp.title == expected_title


def test_tot_vol_cylindrical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using TotalVolumeCylindrical"""

    my_export = TotalVolumeCylindrical("solute", 2)

    assert my_export.allowed_meshes == ["cylindrical"]


def test_tot_vol_spherical_allow_meshes():
    """A simple test to check spherical meshes are the only
    meshes allowed when using TotalVolumeSpherical"""

    my_export = TotalVolumeSpherical("solute", 1)

    assert my_export.allowed_meshes == ["spherical"]


@pytest.mark.parametrize(
    "mesh, field, expected_title",
    [
        (mesh_1D, "solute", "Total solute volume 1 (H m-1)"),
        (mesh_1D, "T", "Total T volume 1 (K m2)"),
        (mesh_2D, "solute", "Total solute volume 1 (H)"),
        (mesh_2D, "T", "Total T volume 1 (K m3)"),
    ],
)
def test_tot_vol_cyl_get_dimension_from_mesh(mesh, field, expected_title):
    """A test to ensure the dimension required for the units can be taken
    from a mesh and produces the expected title"""

    my_export = TotalVolumeCylindrical(field, 1)

    vm = f.MeshFunction("size_t", mesh, mesh.topology().dim())
    dx = f.Measure("dx", domain=mesh, subdomain_data=vm)

    my_export.dx = dx

    assert my_export.title == expected_title

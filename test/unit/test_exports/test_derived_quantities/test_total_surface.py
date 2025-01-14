from festim import x, y, TotalSurface, TotalSurfaceCylindrical, TotalSurfaceSpherical
import fenics as f
import pytest
from .tools import c_1D, c_2D, c_3D, mesh_1D, mesh_2D
import pytest
from sympy.printing import ccode
import numpy as np


@pytest.mark.parametrize("field,surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.TotalSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_total = TotalSurface(field, surface)
    my_total.function = c_2D
    assert my_total.title == f"Total {field} surface {surface} ({my_total.export_unit})"


class TestCompute:
    """Test that the total surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    surface_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(surface_markers, 2)

    ds = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_total = TotalSurface("solute", surface)
    my_total.function = c
    my_total.ds = ds

    def test_minimum(self):
        expected = f.assemble(self.c * self.ds(self.surface))

        produced = self.my_total.compute()
        assert produced == expected


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute surface 8 (H m-2)"),
        (c_1D, "T", "Total T surface 8 (K)"),
        (c_2D, "solute", "Total solute surface 8 (H m-1)"),
        (c_2D, "T", "Total T surface 8 (K m)"),
        (c_3D, "solute", "Total solute surface 8 (H)"),
        (c_3D, "T", "Total T surface 8 (K m2)"),
    ],
)
def test_title_with_units(function, field, expected_title):
    my_export = TotalSurface(surface=8, field=field)
    my_export.function = function
    my_export.show_units = True

    assert my_export.title == expected_title


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("r0", [0, 2])
@pytest.mark.parametrize("height", [2, 3])
def test_compute_cylindrical(r0, radius, height):
    """
    Test that TotalSurfaceCylindrical computes the total value of a function
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

    outer_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[0], {r1}, tol)", tol=1e-14
    )

    surface_markers = f.MeshFunction(
        "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
    )
    surface_markers.set_all(0)
    ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)
    outer_id = 1
    outer_surface.mark(surface_markers, outer_id)

    my_exp = TotalSurfaceCylindrical("solute", outer_id)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r, z: r**2 + z
    expr = f.Expression(
        ccode(c_fun(x, y)),
        degree=1,
    )
    my_exp.function = f.interpolate(expr, V)
    my_exp.ds = ds

    expected_value = 2 * np.pi * r1**3 * z1 + np.pi * r1 * z1**2

    computed_value = my_exp.compute()

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("r0", [0, 1.5])
@pytest.mark.parametrize("radius", [3, 4])
def test_compute_spherical(r0, radius):
    """
    Test that TotalSurfaceSpherical computes the total value of a function
    correctly on a hollow sphere

    Args:
        r0 (float): internal radius
        radius (float): sphere radius
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    mesh_fenics = f.IntervalMesh(10, r0, r1)

    # marking physical groups (volumes and surfaces)
    outer_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[0], {r1}, tol)", tol=1e-14
    )
    surface_markers = f.MeshFunction(
        "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
    )
    surface_markers.set_all(0)
    outer_id = 1
    outer_surface.mark(surface_markers, outer_id)
    ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)

    my_tot = TotalSurfaceSpherical("solute", outer_id)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: r**2 + r
    expr = f.Expression(
        ccode(c_fun(x)),
        degree=1,
    )
    my_tot.function = f.interpolate(expr, V)
    my_tot.ds = ds

    expected_value = 4 * np.pi * r1**3 * (1 + r1)

    computed_value = my_tot.compute()

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_cylindrical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        TotalSurfaceCylindrical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_spherical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        TotalSurfaceSpherical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "polar_range", [(0, 2 * np.pi), (-np.pi, 0), (-2 * np.pi, 3 * np.pi)]
)
def test_polar_range_spherical(polar_range):
    """
    Tests that an error is raised when the polar range is out of bounds
    """
    with pytest.raises(ValueError):
        TotalSurfaceSpherical("solute", 1, polar_range=polar_range)


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute surface 3 (H m-1)"),
        (c_1D, "T", "Total T surface 3 (K m)"),
        (c_2D, "solute", "Total solute surface 3 (H)"),
        (c_2D, "T", "Total T surface 3 (K m2)"),
    ],
)
def test_TotalSurfaceCylindrical_title_with_units(function, field, expected_title):
    my_exp = TotalSurfaceCylindrical(field=field, surface=3)
    my_exp.function = function
    my_exp.show_units = True

    assert my_exp.title == expected_title


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "Total solute surface 4 (H)"),
        (c_1D, "T", "Total T surface 4 (K m2)"),
    ],
)
def test_TotalSurfaceSpherical_title_with_units(function, field, expected_title):
    my_exp = TotalSurfaceSpherical(field=field, surface=4)
    my_exp.function = function
    my_exp.show_units = True

    assert my_exp.title == expected_title


def test_tot_surf_cylindrical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using TotalSurfaceCylindrical"""

    my_export = TotalSurfaceCylindrical("solute", 2)

    assert my_export.allowed_meshes == ["cylindrical"]


def test_tot_surf_spherical_allow_meshes():
    """A simple test to check spherical meshes are the only
    meshes allowed when using TotalSurfaceSpherical"""

    my_export = TotalSurfaceSpherical("solute", 1)

    assert my_export.allowed_meshes == ["spherical"]


@pytest.mark.parametrize(
    "mesh, field, expected_title",
    [
        (mesh_1D, "solute", "Total solute surface 1 (H m-1)"),
        (mesh_1D, "T", "Total T surface 1 (K m)"),
        (mesh_2D, "solute", "Total solute surface 1 (H)"),
        (mesh_2D, "T", "Total T surface 1 (K m2)"),
    ],
)
def test_tot_surf_cyl_get_dimension_from_mesh(mesh, field, expected_title):
    """A test to ensure the dimension required for the units can be taken
    from a mesh and produces the expected title"""

    my_export = TotalSurfaceCylindrical(field, 1)

    vm = f.MeshFunction("size_t", mesh, mesh.topology().dim())
    dx = f.Measure("dx", domain=mesh, subdomain_data=vm)

    my_export.dx = dx

    assert my_export.title == expected_title

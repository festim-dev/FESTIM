from festim import (
    AverageSurface,
    AverageSurfaceCylindrical,
    AverageSurfaceSpherical,
    x,
    y,
)
import fenics as f
import pytest
import numpy as np
from sympy.printing import ccode


@pytest.mark.parametrize("field, surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.AverageSurface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_average = AverageSurface(field, surface)
    assert (
        my_average.title
        == f"Average {field} surface {surface} ({my_average.export_unit})"
    )


class TestCompute:
    """Test that the average surface export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    left = f.CompiledSubDomain("x[0] < 0.5")
    surface_markers = f.MeshFunction("size_t", mesh, 1, 1)
    left.mark(surface_markers, 2)

    ds = f.Measure("dx", domain=mesh, subdomain_data=surface_markers)

    surface = 1
    my_average = AverageSurface("solute", surface)
    my_average.function = c
    my_average.ds = ds

    def test_h_average(self):
        expected = f.assemble(self.c * self.ds(self.surface)) / f.assemble(
            1 * self.ds(self.surface)
        )
        computed = self.my_average.compute()
        assert computed == expected


@pytest.mark.parametrize("radius", [1, 4])
@pytest.mark.parametrize("r0", [3, 5])
@pytest.mark.parametrize("height", [2, 7])
def test_compute_cylindrical(r0, radius, height):
    """
    Test that AverageSurfaceCylindrical computes the value correctly on a hollow cylinder

    Args:
        r0 (float): internal radius
        radius (float): cylinder radius
        height (float): cylinder height
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0, z1 = 0, height

    mesh_fenics = f.RectangleMesh(f.Point(r0, z0), f.Point(r1, z1), 10, 10)

    top_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[1], {z1}, tol)", tol=1e-14
    )
    surface_markers = f.MeshFunction(
        "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
    )
    surface_markers.set_all(0)
    ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)
    # Surface ids
    top_id = 2
    top_surface.mark(surface_markers, top_id)

    my_export = AverageSurfaceCylindrical("solute", top_id)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: 2 * r

    expr = f.Expression(
        ccode(c_fun(x)),
        degree=1,
    )
    my_export.function = f.interpolate(expr, V)
    my_export.ds = ds

    expected_value = 4 / 3 * (r1**3 - r0**3) / (r1**2 - r0**2)

    computed_value = my_export.compute()

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("r0", [3, 5])
def test_compute_spherical(r0, radius):
    """
    Test that AverageSurfaceSpherical computes the average value correctly
    on a hollow sphere

    Args:
        r0 (float): internal radius
        radius (float): sphere  radius
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    mesh_fenics = f.IntervalMesh(10, r0, r1)

    # marking physical groups (volumes and surfaces)
    right_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[0], {r1}, tol)", tol=1e-14
    )
    surface_markers = f.MeshFunction(
        "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
    )
    surface_markers.set_all(0)
    # Surface ids
    right_id = 2
    right_surface.mark(surface_markers, right_id)
    ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)

    my_export = AverageSurfaceSpherical("solute", right_id)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: 4 * r
    expr = f.Expression(
        ccode(c_fun(x)),
        degree=1,
    )
    my_export.function = f.interpolate(expr, V)

    my_export.ds = ds

    expected_value = 4 * r1

    computed_value = my_export.compute()

    assert np.isclose(computed_value, expected_value)


def test_average_surface_cylindrical_title_no_units_solute():
    """A simple test to check that the title is set correctly in
    festim.AverageSurfaceCylindrical with a solute field without units"""

    my_export = AverageSurfaceCylindrical("solute", 4)
    assert my_export.title == "Average solute surface 4 (H m-3)"


def test_average_surface_cylindrical_title_no_units_temperature():
    """A simple test to check that the title is set correctly in
    festim.AverageSurfaceCylindrical with a T field without units"""

    my_export = AverageSurfaceCylindrical("T", 5)
    assert my_export.title == "Average T surface 5 (K)"


def test_average_surface_spherical_title_no_units_solute():
    """A simple test to check that the title is set correctly in
    festim.AverageSurfaceSpherical with a solute field without units"""

    my_export = AverageSurfaceSpherical("solute", 6)
    assert my_export.title == "Average solute surface 6 (H m-3)"


def test_average_surface_spherical_title_no_units_temperature():
    """A simple test to check that the title is set correctly in
    festim.AverageSurfaceSpherical with a T field without units"""

    my_export = AverageSurfaceSpherical("T", 9)
    assert my_export.title == "Average T surface 9 (K)"


def test_avg_surf_cylindrical_allow_meshes():
    """A simple test to check cylindrical meshes are the only
    meshes allowed when using AverageSurfaceCylindrical"""

    my_export = AverageSurfaceCylindrical("solute", 2)

    assert my_export.allowed_meshes == ["cylindrical"]


def test_avg_surf_spherical_allow_meshes():
    """A simple test to check spherical meshes are one of the
    meshes allowed when using AverageSurfaceSpherical"""

    my_export = AverageSurfaceSpherical("T", 6)

    assert "spherical" in my_export.allowed_meshes

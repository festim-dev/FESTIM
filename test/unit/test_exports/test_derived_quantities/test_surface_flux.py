from festim import SurfaceFlux, k_B, SurfaceFluxCylindrical, SurfaceFluxSpherical
import fenics as f
import math
import numpy as np
import pytest
from festim import SurfaceFlux, k_B, x, y
from .tools import c_1D, c_2D, c_3D
from sympy.printing import ccode


@pytest.mark.parametrize("field,surface", [("solute", 1), ("T", 2)])
def test_title(field, surface):
    """
    A simple test to check that the title is set
    correctly in festim.SurfaceFlux

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
    """

    my_h_flux = SurfaceFlux(field, surface)
    assert my_h_flux.title == "Flux surface {}: {}".format(surface, field)


class TestCompute:
    """Test that the surface flux export computes the correct value"""

    mesh = f.UnitIntervalMesh(10)
    V = f.FunctionSpace(mesh, "P", 1)

    c = f.interpolate(f.Expression("x[0]", degree=1), V)
    T = f.interpolate(f.Expression("2*x[0]", degree=1), V)

    left = f.CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = f.CompiledSubDomain("near(x[0], 0) && on_boundary")
    surface_markers = f.MeshFunction("size_t", mesh, 0)
    left.mark(surface_markers, 1)
    right.mark(surface_markers, 2)

    ds = f.Measure("ds", domain=mesh, subdomain_data=surface_markers)
    D = f.Constant(2)
    thermal_cond = f.Constant(3)
    Q = f.Constant(4)

    surface = 1
    n = f.FacetNormal(mesh)
    my_h_flux = SurfaceFlux("solute", surface)
    my_h_flux.D = D
    my_h_flux.thermal_cond = thermal_cond
    my_h_flux.function = c
    my_h_flux.n = n
    my_h_flux.ds = ds
    my_h_flux.T = T
    my_h_flux.Q = Q

    my_heat_flux = SurfaceFlux("T", surface)
    my_heat_flux.D = D
    my_heat_flux.thermal_cond = thermal_cond
    my_heat_flux.function = T
    my_heat_flux.n = n
    my_heat_flux.ds = ds

    def test_h_flux_no_soret(self):
        expected_flux = f.assemble(
            self.D * f.dot(f.grad(self.c), self.n) * self.ds(self.surface)
        )
        flux = self.my_h_flux.compute()
        assert flux == expected_flux

    def test_heat_flux(self):
        expected_flux = f.assemble(
            self.thermal_cond * f.dot(f.grad(self.c), self.n) * self.ds(self.surface)
        )
        flux = self.my_heat_flux.compute()
        assert flux == expected_flux

    def test_h_flux_with_soret(self):
        expected_flux = f.assemble(
            self.D * f.dot(f.grad(self.c), self.n) * self.ds(self.surface)
        )
        expected_flux += f.assemble(
            self.D
            * self.c
            * self.Q
            / (k_B * self.T**2)
            * f.dot(f.grad(self.T), self.n)
            * self.ds(self.surface)
        )
        flux = self.my_h_flux.compute(soret=True)
        assert flux == expected_flux


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("r0", [0, 2])
@pytest.mark.parametrize("height", [2, 3])
@pytest.mark.parametrize("c_top", [2, 3])
@pytest.mark.parametrize("c_bottom", [2, 3])
@pytest.mark.parametrize("soret", [False, True])
def test_compute_cylindrical(r0, radius, height, c_top, c_bottom, soret):
    """
    Test that SurfaceFluxCylindrical computes the flux correctly on a hollow cylinder

    Args:
        r0 (float): internal radius
        radius (float): cylinder radius
        height (float): cylinder height
        c_top (float): concentration top
        c_bottom (float): concentration bottom
        soret (bool): if True, the Soret effect will be set
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    z0 = 0
    z1 = z0 + height
    mesh_fenics = f.RectangleMesh(f.Point(r0, z0), f.Point(r1, z1), 10, 10)

    # marking physical groups (volumes and surfaces)
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(1)

    bot_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[1], {z0}, tol)", tol=1e-14
    )
    top_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[1], {z1}, tol)", tol=1e-14
    )

    surface_markers = f.MeshFunction(
        "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
    )
    surface_markers.set_all(0)
    ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)
    # Surface ids
    bottom_id = 1
    top_id = 2
    bot_surface.mark(surface_markers, bottom_id)
    top_surface.mark(surface_markers, top_id)

    my_flux = SurfaceFluxCylindrical("solute", top_id)
    my_flux.D = f.Constant(2)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda z: c_bottom + (c_top - c_bottom) / (z1 - z0) * z
    expr = f.Expression(
        ccode(c_fun(y)),
        degree=1,
    )
    my_flux.function = f.interpolate(expr, V)

    my_flux.n = f.FacetNormal(mesh_fenics)
    my_flux.ds = ds

    expected_value = (
        -2
        * math.pi
        * float(my_flux.D)
        * (c_bottom - c_top)
        / height
        * (0.5 * r1**2 - 0.5 * r0**2)
    )

    if soret:
        my_flux.Q = f.Constant(2 * k_B)
        growth_rate = 7
        T = lambda z: 3 + growth_rate * z
        T_expr = f.Expression(ccode(T(y)), degree=1)
        my_flux.T = f.interpolate(T_expr, V)
        expected_value += (
            2
            * math.pi
            * float(my_flux.D)
            * float(my_flux.Q)
            * c_fun(z1)
            / k_B
            / T(z1) ** 2
            * growth_rate
            * (0.5 * r1**2 - 0.5 * r0**2)
        )

    computed_value = my_flux.compute(soret=soret)

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("r0", [1, 2])
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("c_left", [3, 4])
@pytest.mark.parametrize("c_right", [1, 2])
@pytest.mark.parametrize("soret", [False, True])
def test_compute_spherical(r0, radius, c_left, c_right, soret):
    """
    Test that SurfaceFluxSpherical computes the flux correctly on a hollow sphere

    Args:
        r0 (float): internal radius
        radius (float): sphere radius
        c_left (float): concentration left
        c_right (float): concentration right
        soret (bool): if True, the Soret effect will be set
    """
    # creating a mesh with FEniCS
    r1 = r0 + radius
    mesh_fenics = f.IntervalMesh(10, r0, r1)

    # marking physical groups (volumes and surfaces)
    volume_markers = f.MeshFunction("size_t", mesh_fenics, mesh_fenics.topology().dim())
    volume_markers.set_all(1)

    left_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[0], {r0}, tol)", tol=1e-14
    )
    right_surface = f.CompiledSubDomain(
        f"on_boundary && near(x[0], {r1}, tol)", tol=1e-14
    )

    surface_markers = f.MeshFunction(
        "size_t", mesh_fenics, mesh_fenics.topology().dim() - 1
    )
    surface_markers.set_all(0)
    # Surface ids
    left_id = 1
    right_id = 2
    left_surface.mark(surface_markers, left_id)
    right_surface.mark(surface_markers, right_id)
    ds = f.Measure("ds", domain=mesh_fenics, subdomain_data=surface_markers)

    my_flux = SurfaceFluxSpherical("solute", right_id)
    my_flux.D = f.Constant(2)
    V = f.FunctionSpace(mesh_fenics, "P", 1)
    c_fun = lambda r: c_left + (c_right - c_left) / (r1 - r0) * r
    expr = f.Expression(
        ccode(c_fun(x)),
        degree=1,
    )
    my_flux.function = f.interpolate(expr, V)

    my_flux.n = f.FacetNormal(mesh_fenics)
    my_flux.ds = ds

    # expected value is the integral of the flux over the surface
    flux_value = float(my_flux.D) * (c_left - c_right) / (r1 - r0)
    expected_value = -4 * math.pi * flux_value * r1**2

    if soret:
        growth_rate = 3
        T = lambda r: 10 + growth_rate * r
        T_expr = f.Expression(ccode(T(x)), degree=1)
        my_flux.Q = f.Constant(5 * k_B)
        my_flux.T = f.interpolate(T_expr, V)
        expected_value += (
            4
            * math.pi
            * float(my_flux.D)
            * float(my_flux.Q)
            * c_fun(r1)
            / k_B
            / T(r1) ** 2
            * growth_rate
            * r1**2
        )

    computed_value = my_flux.compute(soret=soret)

    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_cylindrical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        SurfaceFluxCylindrical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "azimuth_range", [(-1, np.pi), (0, 3 * np.pi), (-1, 3 * np.pi)]
)
def test_azimuthal_range_spherical(azimuth_range):
    """
    Tests that an error is raised when the azimuthal range is out of bounds
    """
    with pytest.raises(ValueError):
        SurfaceFluxSpherical("solute", 1, azimuth_range=azimuth_range)


@pytest.mark.parametrize(
    "polar_range", [(0, 2 * np.pi), (-2 * np.pi, 0), (-2 * np.pi, 3 * np.pi)]
)
def test_polar_range_spherical(polar_range):
    """
    Tests that an error is raised when the polar range is out of bounds
    """
    with pytest.raises(ValueError):
        SurfaceFluxSpherical("solute", 1, polar_range=polar_range)


@pytest.mark.parametrize(
    "function, field, expected_title",
    [
        (c_1D, "solute", "solute flux surface 3 (H m-2 s-1)"),
        (c_1D, "T", "Heat flux surface 3 (W m-2)"),
        (c_2D, "solute", "solute flux surface 3 (H m-1 s-1)"),
        (c_2D, "T", "Heat flux surface 3 (W m-1)"),
        (c_3D, "solute", "solute flux surface 3 (H s-1)"),
        (c_3D, "T", "Heat flux surface 3 (W)"),
    ],
)
def test_title_with_units(function, field, expected_title):
    my_flux = SurfaceFlux(field=field, surface=3)
    my_flux.function = function
    my_flux.show_units = True

    assert my_flux.title == expected_title


def test_cylindrical_flux_title_no_units_solute():
    """A simple test to check that the title is set correctly in
    festim.CylindricalSurfaceFlux with a solute field without units"""

    my_h_flux = SurfaceFluxCylindrical("solute", 2)
    assert my_h_flux.title == "solute flux surface 2"


def test_cylindrical_flux_title_no_units_temperature():
    """A simple test to check that the title is set correctly in
    festim.CylindricalSurfaceFlux with a T field without units"""

    my_heat_flux = SurfaceFluxCylindrical("T", 4)
    assert my_heat_flux.title == "Heat flux surface 4"


def test_spherical_flux_title_no_units_solute():
    """A simple test to check that the title is set correctly in
    festim.SphericalSurfaceFlux with a solute field without units"""

    my_h_flux = SurfaceFluxSpherical("solute", 3)
    assert my_h_flux.title == "solute flux surface 3"


def test_spherical_flux_title_no_units_temperature():
    """A simple test to check that the title is set correctly in
    festim.CSphericalSurfaceFlux with a T field without units"""

    my_heat_flux = SurfaceFluxSpherical("T", 5)
    assert my_heat_flux.title == "Heat flux surface 5"

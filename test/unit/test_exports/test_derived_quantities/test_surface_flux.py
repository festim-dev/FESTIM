from festim import SurfaceFlux, k_B, SurfaceFluxCylindrical, SurfaceFluxSpherical
import fenics as f
import math
import numpy as np
import pytest


def test_title_H():
    surface = 1
    field = "solute"
    my_h_flux = SurfaceFlux(field, surface)
    assert my_h_flux.title == "Flux surface {}: {}".format(surface, field)


def test_title_heat():
    surface = 2
    field = "T"
    my_h_flux = SurfaceFlux(field, surface)
    assert my_h_flux.title == "Flux surface {}: {}".format(surface, field)


class TestCompute:
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
def test_compute_cylindrical(r0, radius, height, c_top, c_bottom):
    """
    Test that SurfaceFluxCylindrical computes the flux correctly on a hollow cylinder

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
    expr = f.Expression(
        f"{c_bottom} + {(c_top - c_bottom)/(z1-z0)} * x[1]",
        degree=1,
    )
    my_flux.function = f.interpolate(expr, V)

    my_flux.n = f.FacetNormal(mesh_fenics)
    my_flux.ds = ds

    computed_value = my_flux.compute()
    expected_value = (
        -2
        * math.pi
        * float(my_flux.D)
        * (c_bottom - c_top)
        / height
        * (0.5 * r1**2 - 0.5 * r0**2)
    )
    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("r0", [1, 2])
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("c_left", [3, 4])
@pytest.mark.parametrize("c_right", [1, 2])
def test_compute_spherical(r0, radius, c_left, c_right):
    """
    Test that SurfaceFluxSpherical computes the flux correctly on a hollow sphere

    Args:
        r0 (float): internal radius
        radius (float): sphere radius
        c_left (float): concentration left
        c_right (float): concentration right
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
    expr = f.Expression(
        f"{c_left} + {(c_right - c_left)/(r1-r0)} * x[0]",
        degree=1,
    )
    my_flux.function = f.interpolate(expr, V)

    my_flux.n = f.FacetNormal(mesh_fenics)
    my_flux.ds = ds

    computed_value = my_flux.compute()
    # expected value is the integral of the flux over the surface
    flux_value = float(my_flux.D) * (c_left - c_right) / (r1 - r0)
    expected_value = -4 * math.pi * flux_value * r1**2
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


def test_soret_raises_error():
    """
    Tests that an error is raised when the Soret effect is used with SurfaceFluxCylindrical
    """
    my_flux = SurfaceFluxCylindrical("T", 1)
    with pytest.raises(NotImplementedError):
        my_flux.compute(soret=True)


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


def test_soret_raises_error_spherical():
    """
    Tests that an error is raised when the Soret effect is used with SurfaceFluxSpherical
    """
    my_flux = SurfaceFluxSpherical("T", 1)
    with pytest.raises(NotImplementedError):
        my_flux.compute(soret=True)

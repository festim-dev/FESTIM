from festim import SurfaceFlux, R
import fenics as f


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
    H = f.Constant(4)

    surface = 1
    n = f.FacetNormal(mesh)
    my_h_flux = SurfaceFlux("solute", surface)
    my_h_flux.D = D
    my_h_flux.thermal_cond = thermal_cond
    my_h_flux.function = c
    my_h_flux.n = n
    my_h_flux.ds = ds
    my_h_flux.T = T
    my_h_flux.H = H

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
            * self.H
            / (R * self.T**2)
            * f.dot(f.grad(self.T), self.n)
            * self.ds(self.surface)
        )
        flux = self.my_h_flux.compute(soret=True)
        assert flux == expected_flux

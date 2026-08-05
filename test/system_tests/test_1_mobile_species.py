from mpi4py import MPI

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import create_unit_cube, create_unit_square, locate_entities

import festim as F

from .tools import error_L2

test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
test_mesh_2d = create_unit_square(MPI.COMM_WORLD, 50, 50)
test_mesh_3d = create_unit_cube(MPI.COMM_WORLD, 20, 20, 20)
x_1d = ufl.SpatialCoordinate(test_mesh_1d.mesh)
x_2d = ufl.SpatialCoordinate(test_mesh_2d)
x_3d = ufl.SpatialCoordinate(test_mesh_3d)


def test_1_mobile_MMS_steady_state():
    """MMS test with one mobile species at steady state."""

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1

    def T_expr(x):
        return 500 + 100 * x[0]

    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_1d)))
    my_model.sources = [F.ParticleSource(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 1e-7


def test_1_mobile_MMS_transient():
    """MMS test with 1 mobile species in 0.1s transient, the value at the last time step
    is compared to an analytical solution."""

    final_time = 0.1

    def u_exact(mod):
        return lambda x, t: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * t**2

    def u_exact_alt(mod):
        return lambda x: u_exact(mod)(x, final_time)

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact_alt(np)

    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1

    def T_expr(x):
        return 600 + 50 * x[0]

    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    # FESTIM model

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    def init_value(x):
        return 1 + ufl.sin(2 * ufl.pi * x[0])

    my_model.initial_conditions = [
        F.InitialConcentration(value=init_value, species=H, volume=vol)
    ]

    def f(x, t):
        return 4 * t - ufl.div(D * ufl.grad(H_analytical_ufl(x, t)))

    my_model.sources = [F.ParticleSource(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=final_time)
    my_model.settings.stepsize = final_time / 50

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 5e-4


def test_1_mobile_MMS_2D():
    """Tests that a steady simulation can be run in a 2D domain with 1 mobile
    species."""

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + mod.cos(2 * mod.pi * x[1])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    V = fem.functionspace(test_mesh_2d, ("Lagrange", 1))
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1

    def T_expr(x):
        return 500 + 100 * x[0]

    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_2d)

    class LefSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0))
            return indices

    class RightSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1))
            return indices

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    left = LefSurface(id=1)
    right = RightSurface(id=2)
    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_2d)))
    my_model.sources = [F.ParticleSource(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 2e-3


def test_1_mobile_MMS_3D():
    """Tests that a steady simulation can be run in a 3D domain with 1 mobile
    species."""

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + mod.cos(2 * mod.pi * x[1])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    V = fem.functionspace(test_mesh_3d, ("Lagrange", 1))
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1

    def T_expr(x):
        return 500 + 100 * x[0]

    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_3d)

    class LefSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0))
            return indices

    class RightSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1))
            return indices

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    left = LefSurface(id=1)
    right = RightSurface(id=2)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_3d)))
    my_model.sources = [F.ParticleSource(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 1e-2


def test_species_dependent_source_MMS_steady_state():
    """MMS test that a species-dependent source works: species A has a source that
    depends on the concentration of species B (value=lambda B: B**2). B has a linear
    manufactured solution, which P1 elements represent exactly, so the species-dependent
    source is exact and A must converge to its own manufactured solution."""

    def A_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0])

    # linear -> represented exactly by P1, so the B**2 source is evaluated exactly
    def B_exact(x):
        return 1 + x[0]

    A_analytical_ufl = A_exact(ufl)
    A_analytical_np = A_exact(np)

    D_0 = 1
    E_D = 0  # -> constant diffusivity D = D_0

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_model.subdomains = [vol, left, right]

    A = F.Species("A")
    B = F.Species("B")
    my_model.species = [A, B]

    my_model.temperature = 500

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=A_analytical_ufl, species=A),
        F.DirichletBC(subdomain=right, value=A_analytical_ufl, species=A),
        F.DirichletBC(subdomain=left, value=B_exact, species=B),
        F.DirichletBC(subdomain=right, value=B_exact, species=B),
    ]

    # source on A that depends on the concentration of B
    species_dependent_source = F.ParticleSource(
        value=lambda B: B**2,
        volume=vol,
        species=A,
        species_dependent_value={"B": B},
    )
    # manufactured source so that, together with the B**2 source, A satisfies A_exact
    f = -ufl.div(D_0 * ufl.grad(A_analytical_ufl(x_1d))) - B_exact(x_1d) ** 2
    manufactured_source = F.ParticleSource(value=f, volume=vol, species=A)

    my_model.sources = [species_dependent_source, manufactured_source]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    A_computed = A.post_processing_solution

    L2_error = error_L2(A_computed, A_analytical_np)

    assert L2_error < 1e-3

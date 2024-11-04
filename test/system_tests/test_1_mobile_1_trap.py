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


def test_1_mobile_1_trap_MMS_steady_state():
    """
    MMS test with 1 mobile species and 1 trap at steady state
    """

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0])

    def v_exact(mod):
        return lambda x: mod.sin(3 * mod.pi * x[0])

    mobile_analytical_ufl = u_exact(ufl)
    mobile_analytical_np = u_exact(np)
    trapped_analytical_ufl = v_exact(ufl)
    trapped_analytical_np = v_exact(np)

    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Function(V)
    f = fem.Function(V)
    g = fem.Function(V)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(mobile_analytical_ufl(x_1d)))
        + k * mobile_analytical_ufl(x_1d) * (n_trap - trapped_analytical_ufl(x_1d))
        - p * trapped_analytical_ufl(x_1d)
    )

    g = p * trapped_analytical_ufl(x_1d) - k * mobile_analytical_ufl(x_1d) * (
        n_trap - trapped_analytical_ufl(x_1d)
    )

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=1)
    my_model.subdomains = [vol, left, right]

    mobile = F.Species("mobile")
    trapped = F.Species("trapped", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[trapped])
    my_model.species = [mobile, trapped]

    my_model.reactions = [
        F.Reaction(
            reactant=[mobile, traps],
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_analytical_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_analytical_ufl, species=trapped),
    ]

    my_model.sources = [
        F.ParticleSource(value=f, volume=vol, species=mobile),
        F.ParticleSource(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_analytical_np)
    L2_error_trapped = error_L2(trapped_computed, trapped_analytical_np)

    assert L2_error_mobile < 2e-07
    assert L2_error_trapped < 1e-07


def test_1_mobile_1_trap_MMS_transient():
    """
    MMS test with 1 mobile species and 1 trap in 0.1s transient, the value at the last time step is
    compared to an analytical solution
    """

    final_time = 0.1

    def u_exact(mod):
        return lambda x, t: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * t**2

    def u_exact_alt(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * final_time**2

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact_alt(np)

    V = fem.functionspace(test_mesh_1d.mesh, ("Lagrange", 1))
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 600 + 50 * x[0]
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

    init_value = lambda x: 1 + ufl.sin(2 * ufl.pi * x[0])
    my_model.initial_conditions = [F.InitialCondition(value=init_value, species=H)]

    f = lambda x, t: 4 * t - ufl.div(D * ufl.grad(H_analytical_ufl(x_1d, t)))
    my_model.sources = [F.ParticleSource(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=final_time)
    my_model.settings.stepsize = final_time / 50

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 5e-4


def test_1_mobile_1_trap_MMS_3D():
    """Tests that a steady simulation can be run in a 3D domain with
    1 mobile and 1 trapped species"""

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])

    def v_exact(mod):
        return lambda x: mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])

    mobile_analytical_ufl = u_exact(ufl)
    mobile_analytical_np = u_exact(np)
    trapped_analytical_ufl = v_exact(ufl)
    trapped_analytical_np = v_exact(np)

    V = fem.functionspace(test_mesh_3d, ("Lagrange", 1))
    T = fem.Function(V)
    f = fem.Function(V)
    g = fem.Function(V)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(mobile_analytical_ufl(x_3d)))
        + k * mobile_analytical_ufl(x_3d) * (n_trap - trapped_analytical_ufl(x_3d))
        - p * trapped_analytical_ufl(x_3d)
    )

    g = p * trapped_analytical_ufl(x_3d) - k * mobile_analytical_ufl(x_3d) * (
        n_trap - trapped_analytical_ufl(x_3d)
    )

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_3d)

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)

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

    left = LefSurface(id=1)
    right = RightSurface(id=2)

    my_model.subdomains = [vol, left, right]

    mobile = F.Species("mobile")
    trapped = F.Species("trapped", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[trapped])
    my_model.species = [mobile, trapped]

    my_model.reactions = [
        F.Reaction(
            reactant=[mobile, traps],
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_analytical_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_analytical_ufl, species=trapped),
    ]

    my_model.sources = [
        F.ParticleSource(value=f, volume=vol, species=mobile),
        F.ParticleSource(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_analytical_np)
    L2_error_trapped = error_L2(trapped_computed, trapped_analytical_np)

    assert L2_error_mobile < 3e-02
    assert L2_error_trapped < 9e-03


def test_1_mobile_1_trap_MMS_2D():
    """Tests that a steady simulation can be run in a 2D domain with
    1 mobile and 1 trapped species"""

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])

    def v_exact(mod):
        return lambda x: mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])

    mobile_analytical_ufl = u_exact(ufl)
    mobile_analytical_np = u_exact(np)
    trapped_analytical_ufl = v_exact(ufl)
    trapped_analytical_np = v_exact(np)

    V = fem.functionspace(test_mesh_2d, ("Lagrange", 1))
    T = fem.Function(V)
    f = fem.Function(V)
    g = fem.Function(V)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(mobile_analytical_ufl(x_2d)))
        + k * mobile_analytical_ufl(x_2d) * (n_trap - trapped_analytical_ufl(x_2d))
        - p * trapped_analytical_ufl(x_2d)
    )

    g = p * trapped_analytical_ufl(x_2d) - k * mobile_analytical_ufl(x_2d) * (
        n_trap - trapped_analytical_ufl(x_2d)
    )

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

    mobile = F.Species("mobile")
    trapped = F.Species("trapped", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[trapped])
    my_model.species = [mobile, trapped]

    my_model.reactions = [
        F.Reaction(
            reactant=[mobile, traps],
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_analytical_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_analytical_ufl, species=trapped),
    ]

    my_model.sources = [
        F.ParticleSource(value=f, volume=vol, species=mobile),
        F.ParticleSource(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_analytical_np)
    L2_error_trapped = error_L2(trapped_computed, trapped_analytical_np)

    assert L2_error_mobile < 4e-03
    assert L2_error_trapped < 2e-03

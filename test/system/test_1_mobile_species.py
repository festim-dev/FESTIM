import festim as F
import numpy as np
from dolfinx import fem
import ufl
from tools import error_L2
from fenicsx import mesh
from mpi4py import MPI


test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
test_mesh_2d = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
test_mesh_3d = mesh.create_unit_cube(MPI.COMM_WORLD, 20, 20, 20)
x_1d = ufl.SpatialCoordinate(test_mesh_1d.mesh)
x_2d = ufl.SpatialCoordinate(test_mesh_2d.mesh)
x_3d = ufl.SpatialCoordinate(test_mesh_3d.mesh)

def test_1_mobile_MMS_steady_state():
    """
    MMS test with one mobile species at steady state
    """

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
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
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 1e-7


def test_MMS_transient():
    """
    MMS test with one mobile species in 0.1s transient. Analytical solution
    """

    final_time = 0.1

    def u_exact(mod):
        return lambda x, t: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * t**2

    def u_exact_alt(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * final_time**2

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact_alt(np)

    elements = ufl.FiniteElement("P", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
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

    f = lambda x, t: 4 * t - ufl.div(D * ufl.grad(H_analytical_ufl(x, t)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=final_time)
    my_model.settings.stepsize = final_time / 50

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 5e-4


def test_MMS_2D():

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + + mod.cos(2 * mod.pi * x[1])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_2d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_2d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_2d

    


    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, material=my_mat)
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

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 1e-7


def test_MMS_multivolume():
    """Tests that a steady simulation can be run if a reaction is not defined
    in every volume"""

    def u_exact(mod):
        return lambda x: 1 + mod.cos(2 * mod.pi * x[0])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=my_mat)
    vol_2 = F.VolumeSubdomain1D(id=4, borders=[0.5, 1], material=my_mat)

    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol_1, vol_2, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x)))
    my_model.sources = [
        F.Source(value=f, volume=vol_1, species=H),
        F.Source(value=f, volume=vol_2, species=H),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [F.XDMFExport(filename="test_output.xdmf", field=H)]

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 6e-4



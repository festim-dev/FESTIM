import festim as F
import numpy as np
from dolfinx import fem
import ufl
from dolfinx.io import XDMFFile
import mpi4py.MPI as MPI


test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))


def error_L2(u_computed, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_computed.function_space.ufl_element().degree()
    family = u_computed.function_space.ufl_element().family()
    mesh = u_computed.function_space.mesh
    W = fem.FunctionSpace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(u_computed)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = fem.Function(W)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_exact, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_exact)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def test_MMS_steady_state():
    """
    MMS test with one mobile species at steady state
    """

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0])

    u_ufl = u_exact(ufl)
    u_numpy = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    exact = fem.Function(V)
    exact.interpolate(u_numpy)
    exact_xdmf = XDMFFile(MPI.COMM_WORLD, "results/mms/exact_solution.xdmf", "w")
    exact_xdmf.write_mesh(test_mesh_1d.mesh)
    exact_xdmf.write_function(exact)

    D_0 = 1
    E_D = 0
    D = D_0 * ufl.exp(-E_D / (F.k_B * 1))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_model.subdomains = [vol, left, right]

    A = F.Species("A")
    my_model.species = [A]

    my_model.temperature = 1

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=u_ufl, species=A),
        F.DirichletBC(subdomain=right, value=u_ufl, species=A),
    ]

    f_expr = lambda x: D * 4 * np.pi**2 * np.sin(2 * np.pi * x[0])
    f = fem.Function(V)
    f.interpolate(f_expr)

    my_model.sources = [F.Source(value=f, volume=vol, species=A)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [F.XDMFExport("results/mms/computed_solution.xdmf", field=A)]

    my_model.initialise()

    my_model.run()

    u_computed = my_model.species[0].post_processing_solution

    L2_error = error_L2(u_computed, u_numpy)

    assert L2_error < 1e-7


def test_MMS_steady_state_1_trap():
    """
    MMS test with one mobile species and one trap at steady state
    """

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0])

    def v_exact(mod):
        return lambda x: mod.sin(mod.pi * x[0])

    u_ufl = u_exact(ufl)
    u_numpy = u_exact(np)
    v_ufl = v_exact(ufl)
    v_numpy = v_exact(np)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T = 500
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    V = fem.FunctionSpace(test_mesh_1d.mesh, ("CG", 1))
    f = fem.Function(V)
    f_expr = lambda x: D * 9 * ufl.pi**2 * np.sin(3 * ufl.pi * x[0])
    f.interpolate(f_expr)

    g = fem.Function(V)
    g_expr = lambda x: p * v_numpy(x) - k * u_numpy(x) * (n_trap - v_numpy(x))
    g.interpolate(g_expr)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=1)
    my_model.subdomains = [vol, left, right]

    A = F.Species("A")
    B = F.Species("B", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[B])
    my_model.species = [A, B]

    my_model.reactions = [
        F.Reaction(
            reactant1=A,
            reactant2=traps,
            product=B,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=u_ufl, species=A),
        F.DirichletBC(subdomain=right, value=u_ufl, species=A),
        F.DirichletBC(subdomain=left, value=v_ufl, species=B),
        F.DirichletBC(subdomain=right, value=v_ufl, species=B),
    ]

    my_model.sources = [
        F.Source(value=f, volume=vol, species=A),
        F.Source(value=g, volume=vol, species=B),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [
        F.XDMFExport("results/mms/computed_solution_A.xdmf", field=A),
        F.XDMFExport("results/mms/computed_solution_B.xdmf", field=B),
    ]

    my_model.initialise()

    my_model.run()

    u_computed = my_model.species[0].post_processing_solution
    v_computed = my_model.species[1].post_processing_solution

    L2_error_A = error_L2(u_computed, u_numpy)
    L2_error_B = error_L2(v_computed, v_numpy)

    assert L2_error_A < 1e-02
    assert L2_error_B < 1e-07

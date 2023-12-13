import festim as F
import numpy as np
from dolfinx import fem
import ufl
import mpi4py.MPI as MPI


test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
x = ufl.SpatialCoordinate(test_mesh_1d.mesh)


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
        F.DirichletBC(subdomain=left, value=u_ufl, species=H),
        F.DirichletBC(subdomain=right, value=u_ufl, species=H),
    ]

    f_value = -ufl.div(D * ufl.grad(u_ufl(x)))
    f_expr = fem.Expression(f_value, V.element.interpolation_points())
    f = fem.Function(V)
    f.interpolate(f_expr)

    my_model.sources = [F.Source(value=f_value, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

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
        return lambda x: mod.sin(3 * mod.pi * x[0])

    mobile_ufl = u_exact(ufl)
    mobile_numpy = u_exact(np)
    trapped_ufl = v_exact(ufl)
    trapped_numpy = v_exact(np)

    elements = ufl.FiniteElement("P", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
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

    f_value = (
        -ufl.div(D * ufl.grad(mobile_ufl(x)))
        + k * mobile_ufl(x) * (n_trap - trapped_ufl(x))
        - p * trapped_ufl(x)
    )
    f_expr = fem.Expression(f_value, V.element.interpolation_points())
    f.interpolate(f_expr)

    g_value = p * trapped_ufl(x) - k * mobile_ufl(x) * (n_trap - trapped_ufl(x))
    g_expr = fem.Expression(g_value, V.element.interpolation_points())
    g.interpolate(g_expr)

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
            reactant1=mobile,
            reactant2=traps,
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
        F.DirichletBC(subdomain=left, value=mobile_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_ufl, species=trapped),
    ]

    my_model.sources = [
        F.Source(value=f, volume=vol, species=mobile),
        F.Source(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_numpy)
    L2_error_trapped = error_L2(trapped_computed, trapped_numpy)

    assert L2_error_mobile < 2e-07
    assert L2_error_trapped < 1e-07

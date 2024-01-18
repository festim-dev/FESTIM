import festim as F
import numpy as np
from dolfinx import fem
import ufl
import mpi4py.MPI as MPI


def source_from_exact_solution(
    exact_solution, thermal_conductivity, density, heat_capacity
):
    import sympy as sp
    from sympy.vector import CoordSys3D, divergence, gradient

    R = CoordSys3D("R")
    x = [R.x, R.y, R.z]
    t = sp.symbols("t")

    u = exact_solution(x, t)
    density = density(x, t)
    heat_capacity = heat_capacity(x, t)
    thermal_cond = thermal_conductivity(x, t)
    source = density * heat_capacity * sp.diff(u, t) - divergence(
        thermal_cond * gradient(u, x), x
    )

    return sp.lambdify([x, t], source)


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


def test_MMS_1():
    thermal_conductivity = 4.0
    exact_solution = lambda x: 2 * x[0] ** 2
    mms_source = -4 * thermal_conductivity

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_problem.surface_subdomains = [left, right]
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity

    my_problem.volume_subdomains = [
        F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    ]

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    L2_error = error_L2(computed_solution, exact_solution)
    assert L2_error < 1e-7


def test_MMS_T_dependent_thermal_cond():
    """MMS test with space T dependent thermal cond"""
    thermal_conductivity = lambda T: 3 * T + 2
    exact_solution = lambda x: 2 * x[0] ** 2 + 1
    mms_source = lambda x: -(72 * x[0] ** 2 + 20)  # TODO would be nice to automate

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_problem.surface_subdomains = [left, right]
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity

    my_problem.volume_subdomains = [
        F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    ]

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    L2_error = error_L2(computed_solution, exact_solution)
    assert L2_error < 1e-7


def test_heat_transfer_transient():
    """
    MMS test for transient heat transfer
    constant thermal conductivity density and heat capacity
    """
    density = 2
    heat_capacity = 3
    thermal_conductivity = 4
    exact_solution = lambda x, t: 2 * x[0] ** 2 + 20 * t
    dTdt = 20
    mms_source = density * heat_capacity * dTdt - thermal_conductivity * 4

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(2, 3, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=2)
    right = F.SurfaceSubdomain1D(id=2, x=3)
    my_problem.surface_subdomains = [left, right]
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity
    mat.density = density
    mat.heat_capacity = heat_capacity

    my_problem.volume_subdomains = [
        F.VolumeSubdomain1D(id=1, borders=[2, 3], material=mat)
    ]

    # NOTE: it's good to check that without the IC the solution is not the exact one
    my_problem.initial_condition = F.InitialTemperature(lambda x: exact_solution(x, 0))

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-8,
        rtol=1e-10,
        final_time=1,  # final time shouldn't be too long so that a potential error at the initial timestep is not negligible
    )

    # Forward euler isn't great so dt should be small
    # although it's ok here since the time derivative is constant
    my_problem.settings.stepsize = F.Stepsize(0.1)

    my_problem.exports = [
        F.VTXExportForTemperature(filename="test_transient_heat_transfer.bp")
    ]

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    final_time_sim = (
        my_problem.t.value
    )  # we use the exact final time of the simulation which may differ from the one specified in the settings
    exact_solution_end = lambda x: exact_solution(x, final_time_sim)
    L2_error = error_L2(computed_solution, exact_solution_end)
    assert L2_error < 1e-7


def test_sympify():
    exact_solution = lambda x, t: 2 * x[0] ** 2 + 20 * t

    density = lambda T: 0.2 * T + 2
    heat_capacity = lambda T: 0.2 * T + 3
    thermal_conductivity = lambda T: 0.1 * T + 4

    mms_source_from_sp = source_from_exact_solution(
        exact_solution,
        density=lambda x, t: density(exact_solution(x, t)),
        heat_capacity=lambda x, t: heat_capacity(exact_solution(x, t)),
        thermal_conductivity=lambda x, t: thermal_conductivity(exact_solution(x, t)),
    )
    mms_source = lambda x, t: mms_source_from_sp((x[0], None, None), t)

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(2, 3, 2100))
    left = F.SurfaceSubdomain1D(id=1, x=2)
    right = F.SurfaceSubdomain1D(id=2, x=3)
    my_problem.surface_subdomains = [left, right]
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity
    mat.density = density
    mat.heat_capacity = heat_capacity

    my_problem.volume_subdomains = [
        F.VolumeSubdomain1D(id=1, borders=[2, 3], material=mat)
    ]

    # NOTE: it's good to check that without the IC the solution is not the exact one
    my_problem.initial_condition = F.InitialTemperature(lambda x: exact_solution(x, 0))

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-8,
        rtol=1e-10,
        final_time=1,  # final time shouldn't be too long so that a potential error at the initial timestep is not negligible
    )

    # Forward euler isn't great so dt should be small
    # although it's ok here since the time derivative is constant
    my_problem.settings.stepsize = F.Stepsize(0.05)

    my_problem.exports = [
        F.VTXExportForTemperature(filename="test_transient_heat_transfer.bp")
    ]

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    final_time_sim = (
        my_problem.t.value
    )  # we use the exact final time of the simulation which may differ from the one specified in the settings

    exact_solution_end = lambda x: exact_solution(x, final_time_sim)
    L2_error = error_L2(computed_solution, exact_solution_end)
    assert L2_error < 1e-7

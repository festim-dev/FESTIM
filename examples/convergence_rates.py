import mpi4py.MPI as MPI

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem

import festim as F


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


def run(N):
    def exact_solution(x, t):
        return 2 * x[0] ** 2 + 20 * t

    def density(T):
        return 0.2 * T + 2

    def heat_capacity(T):
        return 0.2 * T + 3

    def thermal_conductivity(T):
        return 0.1 * T + 4

    mms_source_from_sp = source_from_exact_solution(
        exact_solution,
        density=lambda x, t: density(exact_solution(x, t)),
        heat_capacity=lambda x, t: heat_capacity(exact_solution(x, t)),
        thermal_conductivity=lambda x, t: thermal_conductivity(exact_solution(x, t)),
    )

    def mms_source(x, t):
        return mms_source_from_sp((x[0], None, None), t)

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(2, 3, N))
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
        F.VTXSpeciesExport(filename="test_transient_heat_transfer.bp")
    ]

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    # we use the exact final time of the simulation which may differ from the one specified in the settings
    final_time_sim = my_problem.t.value

    def exact_solution_end(x):
        return exact_solution(x, final_time_sim)

    L2_error = error_L2(computed_solution, exact_solution_end)
    return L2_error


if __name__ == "__main__":
    Ns = np.geomspace(10, 3300, 10)
    errors = []
    for N in Ns:
        L2_error = run(int(N))
        print(f"With {int(N)} cells, L2_error = {L2_error}")
        errors.append(L2_error)

    plt.loglog(Ns, errors, "-o", label="L2")
    # plot a slope of 2
    plt.loglog(Ns, 1 / Ns**2, "--", color="black", label="order 2")
    plt.xlabel("Number of cells")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

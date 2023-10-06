from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
    dirichletbc,
    Function,
    locate_dofs_topological,
    form,
    assemble_scalar,
)
from dolfinx.fem.petsc import (
    NonlinearProblem,
)
from dolfinx.nls.petsc import NewtonSolver
from ufl import (
    dot,
    grad,
    TestFunction,
    exp,
    FacetNormal,
    dx,
    ds,
)
from dolfinx import log
import numpy as np
import tqdm.autonotebook


import festim as F

def test_permeation_problem():
    # mesh nodes
    vertices = np.linspace(0, 3e-4, num=1001)

    my_mesh = F.Mesh1D(vertices)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = my_mesh

    my_model.initialise()

    V = my_model.function_space
    u = Function(V)
    u_n = Function(V)
    v = TestFunction(V)


    temperature = 500
    k_B = F.k_B

    # TODO this should be a property of Mesh
    n = FacetNormal(my_mesh.mesh)


    def siverts_law(T, S_0, E_S, pressure):
        S = S_0 * exp(-E_S / k_B / T)
        return S * pressure**0.5

    fdim = my_mesh.mesh.topology.dim - 1
    left_facets = my_model.facet_tags.find(1)
    left_dofs = locate_dofs_topological(V, fdim, left_facets)
    right_facets = my_model.facet_tags.find(2)
    right_dofs = locate_dofs_topological(V, fdim, right_facets)

    surface_conc = siverts_law(T=temperature, S_0=4.02e21, E_S=1.04, pressure=100)
    bc_sieverts = dirichletbc(
        Constant(my_mesh.mesh, PETSc.ScalarType(surface_conc)), left_dofs, V
    )
    bc_outgas = dirichletbc(Constant(my_mesh.mesh, PETSc.ScalarType(0)), right_dofs, V)
    bcs = [bc_sieverts, bc_outgas]

    D_0 = 1.9e-7
    E_D = 0.2

    D = D_0 * exp(-E_D / k_B / temperature)

    dt = 1 / 20
    final_time = 50

    # f = Constant(my_mesh.mesh, (PETSc.ScalarType(0)))
    variational_form = dot(D * grad(u), grad(v)) * dx
    variational_form += ((u - u_n) / dt) * v * dx

    problem = NonlinearProblem(variational_form, u, bcs=bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    my_model.settings.solver.convergence_criterion = "incremental"
    # solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e10


    solver.report = True
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()
    # log.set_log_level(log.LogLevel.INFO)

    mobile_xdmf = XDMFFile(MPI.COMM_WORLD, "mobile_concentration.xdmf", "w")
    mobile_xdmf.write_mesh(my_mesh.mesh)

    flux_values = []
    times = []
    t = 0
    progress = tqdm.autonotebook.tqdm(desc="Solving H transport problem", total=final_time)
    while t < final_time:
        progress.update(dt)
        t += dt

        solver.solve(u)

        # post process
        surface_flux = form(D * dot(grad(u), n) * ds(2))
        flux = assemble_scalar(surface_flux)
        flux_values.append(flux)
        times.append(t)

        # export
        np.savetxt("outgassing_flux.txt", np.array(flux_values))
        np.savetxt("times.txt", np.array(times))

        mobile_xdmf.write_function(u, t)

        # update previous solution
        u_n.x.array[:] = u.x.array[:]

    mobile_xdmf.close()

if __name__=="__main__":
    test_permeation_problem()
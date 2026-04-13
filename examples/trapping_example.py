from petsc4py import PETSc

import numpy as np

import festim as F

L = 3e-04
vertices = np.linspace(0, L, num=1000)

my_mesh = F.Mesh1D(vertices)

my_model = F.HydrogenTransportProblem()
my_model.mesh = my_mesh

my_mat = F.Material(D_0=1.9e-7, E_D=0.2, name="my_mat")
my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=my_mat)
left_surface = F.SurfaceSubdomain1D(id=1, x=0)
right_surface = F.SurfaceSubdomain1D(id=2, x=L)
my_model.subdomains = [
    my_subdomain,
    left_surface,
    right_surface,
]

mobile_H = F.Species("H")
trapped_H = F.Species("trapped_H", mobile=False)
empty_trap = F.ImplicitSpecies(n=1e19, others=[trapped_H], name="empty_trap")
my_model.species = [mobile_H, trapped_H]

my_model.reactions = [
    F.Reaction(
        p_0=1e13,
        E_p=1.2,
        k_0=3.8e-17,
        E_k=0.2,
        reactant=[mobile_H, empty_trap],
        product=trapped_H,
    )
]

temperature = 500.0
my_model.temperature = temperature

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=right_surface, value=0, species="H"),
    F.DirichletBC(subdomain=left_surface, value=1e12, species=mobile_H),
]
my_model.exports = [
    F.VTXSpeciesExport(
        "mobile_concentration_h.bp", field=mobile_H
    ),  # produces 0 in file
    F.VTXSpeciesExport(
        "trapped_concentration_h.bp", field=trapped_H
    ),  # produces 0 in file
    F.XDMFExport("mobile_concentration_h.xdmf", field=mobile_H),
    F.XDMFExport("trapped_concentration_h.xdmf", field=trapped_H),
]

my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    max_iterations=30,
    final_time=50,
)

my_model.settings.stepsize = F.Stepsize(initial_value=1 / 20)

my_model.initialise()


my_model.solver.convergence_criterion = "incremental"
ksp = my_model.solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

times, flux_values = my_model.run()

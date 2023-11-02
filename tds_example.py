import festim as F
import numpy as np

L = 20e-6
vertices = np.concatenate(
    [
        np.linspace(0, 30e-9, num=200, endpoint=False),
        np.linspace(30e-9, 3e-6, num=300, endpoint=False),
        np.linspace(3e-6, 20e-6, num=200),
    ]
)
my_mesh = F.Mesh1D(vertices)

my_model = F.HydrogenTransportProblem()
my_model.mesh = my_mesh

w_atom_density = 6.3e28  # atom/m3

tungsten = F.Material(D_0=4.1e-7, E_D=0.39, name="my_mat")
my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
left_surface = F.SurfaceSubdomain1D(id=1, x=0)
right_surface = F.SurfaceSubdomain1D(id=2, x=L)
my_model.subdomains = [
    my_subdomain,
    left_surface,
    right_surface,
]

mobile_H = F.Species("H")
trapped_H1 = F.Species("trapped_H1", mobile=False)
empty_trap1 = F.ImplicitSpecies(
    n=1.3e-3 * w_atom_density, others=[trapped_H1], name="empty_trap1"
)
trapped_H2 = F.Species("trapped_H2", mobile=False)
empty_trap2 = F.ImplicitSpecies(
    n=4e-3 * w_atom_density, others=[trapped_H2], name="empty_trap2"
)
my_model.species = [mobile_H, trapped_H1, trapped_H2]

my_model.reactions = [
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=0.87,
        reactant1=mobile_H,
        reactant2=empty_trap1,
        product=trapped_H1,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        reactant1=mobile_H,
        reactant2=empty_trap2,
        product=trapped_H2,
    ),
]

implantation_time = 400
start_tds = implantation_time + 50
implantation_temp = 300
temperature_ramp = 8  # K/s


def temp_function(t):
    if t < start_tds:
        return implantation_temp
    else:
        return implantation_temp + temperature_ramp * (t - start_tds)


my_model.temperature = temp_function


def left_conc_value(T):
    D = tungsten.D_0 * np.exp(-tungsten.E_D / F.k_B / T)
    return 2.5e19 * 4.5e-9 / D


left_concentration = F.DirichletBC(
    subdomain=left_surface,
    value=lambda t: left_conc_value(temp_function(t)) if t < implantation_time else 0.0,
    species=mobile_H,
)

my_model.boundary_conditions = [
    left_concentration,
    F.DirichletBC(subdomain=right_surface, value=0, species=mobile_H),
]
my_model.exports = [
    # F.VTXExport("mobile_concentration_h.bp", field=mobile_H),
    # F.VTXExport("trapped_concentration_h.bp", field=trapped_H1),
    F.XDMFExport("mobile_concentration_h.xdmf", field=mobile_H),
    F.XDMFExport("trapped_concentration_h1.xdmf", field=trapped_H1),
    F.XDMFExport("trapped_concentration_h2.xdmf", field=trapped_H2),
]

my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    max_iterations=30,
    final_time=500,
)

my_model.settings.stepsize = F.Stepsize(initial_value=0.5)

my_model.initialise()

# from petsc4py import PETSc

# my_model.solver.convergence_criterion = "incremental"
# ksp = my_model.solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()

from dolfinx.fem import form, assemble_scalar
from ufl import dot, grad


# needs monkey patching of iterate method


def new_iterate(cls, skip_postprocessing=False):
    cls.progress.update(cls.dt.value)
    cls.t.value += cls.dt.value

    cls.update_time_dependent_values()

    cls.solver.solve(cls.u)

    for idx, spe in enumerate(cls.species):
        spe.post_processing_solution = cls.u.sub(idx)

    cm, *trapped_cs = cls.u.split()

    D = cls.subdomains[0].material.get_diffusion_coefficient(
        cls.mesh.mesh, cls.temperature_fenics, mobile_H
    )
    surface_flux_left = assemble_scalar(form(D * dot(grad(cm), cls.mesh.n) * cls.ds(1)))
    surface_flux_right = assemble_scalar(
        form(D * dot(grad(cm), cls.mesh.n) * cls.ds(2))
    )

    cls.times.append(float(cls.t))
    cls.flux_values_1.append(surface_flux_left)
    cls.flux_values_2.append(surface_flux_right)

    for export in cls.exports:
        if isinstance(export, (F.VTXExport, F.XDMFExport)):
            export.write(float(cls.t))

    # update previous solution
    cls.u_n.x.array[:] = cls.u.x.array[:]


F.HydrogenTransportProblem.iterate = new_iterate


times, flux_values = my_model.run()
np.savetxt(
    "outgassing_flux_tds.txt",
    np.array(my_model.flux_values_1) + np.array(my_model.flux_values_2),
)
np.savetxt("times_tds.txt", np.array(times))

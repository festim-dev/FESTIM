import numpy as np

import festim as F

my_model = F.HydrogenTransportProblem()

# -------- Mesh --------- #

L = 5e-6
vertices = np.linspace(0, L, num=2000)
my_model.mesh = F.Mesh1D(vertices)


# -------- Materials and subdomains --------- #

w_atom_density = 6.3e28  # atom/m3

tungsten = F.Material(D_0=4.1e-7, E_D=0.39, name="tungsten")

my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
left_surface = F.SurfaceSubdomain1D(id=1, x=0)
right_surface = F.SurfaceSubdomain1D(id=2, x=L)

my_model.subdomains = [
    my_subdomain,
    left_surface,
    right_surface,
]

# -------- Hydrogen species and reactions --------- #

mobile_H = F.Species("H")
mobile_D = F.Species("D")


trapped_H1 = F.Species("trapped_H1", mobile=False)
trapped_D1 = F.Species("trapped_D1", mobile=False)
trapped_H2 = F.Species("trapped_H2", mobile=False)
trapped_D2 = F.Species("trapped_D2", mobile=False)
trapped_HD = F.Species("trapped_HD", mobile=False)

empty_trap = F.ImplicitSpecies(
    n=1e21,
    others=[trapped_H1, trapped_D1, trapped_H2, trapped_HD, trapped_D2],
    name="empty_trap",
)


my_model.species = [
    mobile_H,
    mobile_D,
    trapped_H1,
    trapped_D1,
    trapped_H2,
    trapped_HD,
    trapped_D2,
]

my_model.reactions = [
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.2,
        reactant=[mobile_H, empty_trap],
        product=trapped_H1,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        reactant=[mobile_H, trapped_H1],
        product=trapped_H2,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.2,
        reactant=[mobile_D, empty_trap],
        product=trapped_D1,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.2,
        reactant=[mobile_D, trapped_D1],
        product=trapped_D2,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        reactant=[mobile_H, trapped_D1],
        product=trapped_HD,
    ),
    F.Reaction(
        k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
        E_k=0.39,
        p_0=1e13,
        E_p=1.0,
        reactant=[mobile_D, trapped_H1],
        product=trapped_HD,
    ),
]

# -------- Temperature --------- #

my_model.temperature = 300

# -------- Boundary conditions --------- #


my_model.boundary_conditions = [
    F.DirichletBC(subdomain=left_surface, value=1e20, species=mobile_H),
    F.DirichletBC(subdomain=right_surface, value=1e19, species=mobile_D),
    F.DirichletBC(subdomain=right_surface, value=0, species=mobile_H),
    F.DirichletBC(subdomain=left_surface, value=0, species=mobile_D),
]

# -------- Exports --------- #

left_flux = F.SurfaceFlux(field=mobile_H, surface=left_surface)
right_flux = F.SurfaceFlux(field=mobile_H, surface=right_surface)

folder = "multi_isotope_trapping_example"

my_model.exports = [
    F.XDMFExport(f"{folder}/mobile_concentration_h.xdmf", field=mobile_H),
    F.XDMFExport(f"{folder}/mobile_concentration_d.xdmf", field=mobile_D),
    F.XDMFExport(f"{folder}/trapped_concentration_h1.xdmf", field=trapped_H1),
    F.XDMFExport(f"{folder}/trapped_concentration_h2.xdmf", field=trapped_H2),
    F.XDMFExport(f"{folder}/trapped_concentration_d1.xdmf", field=trapped_D1),
    F.XDMFExport(f"{folder}/trapped_concentration_d2.xdmf", field=trapped_D2),
    F.XDMFExport(f"{folder}/trapped_concentration_hd.xdmf", field=trapped_HD),
]

# -------- Settings --------- #

my_model.settings = F.Settings(
    atol=1e-10, rtol=1e-10, max_iterations=30, final_time=3000
)

my_model.settings.stepsize = F.Stepsize(initial_value=20)

# -------- Run --------- #

my_model.initialise()

print(my_model.formulation)
# exit()
my_model.run()

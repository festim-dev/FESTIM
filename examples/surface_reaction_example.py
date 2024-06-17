import festim as F
import numpy as np

import dolfinx.fem as fem
import ufl


class FluxFromSurfaceReaction(F.SurfaceFlux):
    def __init__(self, reaction: F.SurfaceReactionBC):
        super().__init__(
            F.Species(),  # just a dummy species here
            reaction.subdomain,
        )
        self.reaction = reaction.flux_bcs[0]

    def compute(self, ds):
        self.value = fem.assemble_scalar(
            fem.form(self.reaction.value_fenics * ds(self.surface.id))
        )
        self.data.append(self.value)


my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 1000))
my_mat = F.Material(name="mat", D_0=1, E_D=0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=1)

my_model.subdomains = [vol, left, right]

H = F.Species("H")
D = F.Species("D")
my_model.species = [H, D]

my_model.temperature = 500

surface_reaction_hd = F.SurfaceReactionBC(
    reactant=[H, D],
    gas_pressure=0,
    k_r0=0.01,
    E_kr=0,
    k_d0=0,
    E_kd=0,
    subdomain=right,
)

surface_reaction_hh = F.SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=lambda t: ufl.conditional(ufl.gt(t, 1), 2, 0),
    k_r0=0.02,
    E_kr=0,
    k_d0=0.03,
    E_kd=0,
    subdomain=right,
)

surface_reaction_dd = F.SurfaceReactionBC(
    reactant=[D, D],
    gas_pressure=0,
    k_r0=0.01,
    E_kr=0,
    k_d0=0,
    E_kd=0,
    subdomain=right,
)

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=left, value=2, species=H),
    F.DirichletBC(subdomain=left, value=2, species=D),
    surface_reaction_hd,
    surface_reaction_hh,
    surface_reaction_dd,
]

H_flux_right = F.SurfaceFlux(H, right)
H_flux_left = F.SurfaceFlux(H, left)
D_flux_right = F.SurfaceFlux(D, right)
D_flux_left = F.SurfaceFlux(D, left)
HD_flux = FluxFromSurfaceReaction(surface_reaction_hd)
HH_flux = FluxFromSurfaceReaction(surface_reaction_hh)
DD_flux = FluxFromSurfaceReaction(surface_reaction_dd)
my_model.exports = [
    F.XDMFExport("test.xdmf", H),
    H_flux_left,
    H_flux_right,
    D_flux_left,
    D_flux_right,
    HD_flux,
    HH_flux,
    DD_flux,
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5, transient=True)

my_model.settings.stepsize = 0.1

my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt

plt.stackplot(
    H_flux_left.t,
    np.abs(H_flux_left.data),
    np.abs(D_flux_left.data),
    labels=["H_in", "D_in"],
)
plt.stackplot(
    H_flux_right.t,
    -np.abs(H_flux_right.data),
    -np.abs(D_flux_right.data),
    labels=["H_out", "D_out"],
)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Flux (atom/m^2/s)")
plt.figure()
plt.stackplot(
    HD_flux.t,
    np.abs(HH_flux.data),
    np.abs(HD_flux.data),
    np.abs(DD_flux.data),
    labels=["HH", "HD", "DD"],
)
plt.legend(reverse=True)
plt.xlabel("Time (s)")
plt.ylabel("Flux (molecule/m^2/s)")


plt.figure()
plt.plot(H_flux_right.t, -np.array(H_flux_right.data), label="from gradient (H)")
plt.plot(
    H_flux_right.t,
    2 * np.array(HH_flux.data) + np.array(HD_flux.data),
    linestyle="--",
    label="from reaction rates (H)",
)

plt.plot(D_flux_right.t, -np.array(D_flux_right.data), label="from gradient (D)")
plt.plot(
    D_flux_right.t,
    2 * np.array(DD_flux.data) + np.array(HD_flux.data),
    linestyle="--",
    label="from reaction rates (D)",
)
plt.xlabel("Time (s)")
plt.ylabel("Flux (atom/m^2/s)")
plt.legend()
plt.show()

# check that H_flux_right == 2*HH_flux + HD_flux
H_flux_from_gradient = -np.array(H_flux_right.data)
H_flux_from_reac = 2 * np.array(HH_flux.data) + np.array(HD_flux.data)
assert np.allclose(
    H_flux_from_gradient,
    H_flux_from_reac,
    rtol=0.5e-2,
    atol=0.005,
)
# check that D_flux_right == 2*DD_flux + HD_flux
D_flux_from_gradient = -np.array(D_flux_right.data)
D_flux_from_reac = 2 * np.array(DD_flux.data) + np.array(HD_flux.data)
assert np.allclose(
    D_flux_from_gradient,
    D_flux_from_reac,
    rtol=0.5e-2,
    atol=0.005,
)

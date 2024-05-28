import festim as F
import numpy as np

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
    gas_pressure=0,
    k_r0=0.01,
    E_kr=0,
    k_d0=0,
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
    F.DirichletBC(subdomain=left, value=5, species=H),
    F.DirichletBC(subdomain=left, value=5, species=D),
    surface_reaction_hd,
    surface_reaction_hh,
    surface_reaction_dd,
]

my_model.exports = [F.XDMFExport("test.xdmf", H)]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=10, transient=True)

my_model.settings.stepsize = 0.1

my_model.initialise()
my_model.run()

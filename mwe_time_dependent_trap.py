import festim as F
import ufl
import numpy as np

my_model = F.HydrogenTransportProblem()

L = 1
my_model.mesh = F.Mesh1D(np.linspace(0, L, num=500))

mat = F.Material(D_0=1, E_D=0)

left_subdomain = F.SurfaceSubdomain1D(id=1, x=0)
right_subdomain = F.SurfaceSubdomain1D(id=2, x=L)
volume = F.VolumeSubdomain1D(id=1, borders=[0, L], material=mat)

H = F.Species(name="H")
trapped_H = F.Species(name="H_t", mobile=False)

time_dependent_density = lambda x: ufl.conditional(ufl.lt(x[0], 0.5), 10, 0)
empty_traps = F.ImplicitSpecies(n=time_dependent_density, others=[trapped_H])

my_model.species = [H, trapped_H]

my_model.reactions = [
    F.Reaction(
        reactant=[H, empty_traps],
        product=[trapped_H],
        k_0=0.1,
        E_k=0,
        p_0=0,
        E_p=0,
        volume=volume,
    ),
]


my_model.subdomains = [left_subdomain, right_subdomain, volume]

my_model.temperature = 500

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_subdomain, value=10, species=H),
    F.FixedConcentrationBC(subdomain=right_subdomain, value=0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=100)
my_model.settings.stepsize = F.Stepsize(0.2)

my_model.exports = [F.VTXExport("results.bp", field=empty_traps)]
my_model.initialise()
my_model.run()

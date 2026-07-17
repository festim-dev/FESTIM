import numpy as np

import festim as F

# MWE for a source term that depends on BOTH time and the concentration of another
# species. This currently crashes in Value.convert_input_value: the "t-only" fast path
# is taken (because "t" is an argument and "x"/"T" are not), which calls the lambda with
# only t and misses the species argument B.

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

mat = F.Material(D_0=1.0, E_D=0.0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
left = F.SurfaceSubdomain1D(id=2, x=0)
right = F.SurfaceSubdomain1D(id=3, x=1)
my_model.subdomains = [vol, left, right]

A = F.Species("A")
B = F.Species("B")
my_model.species = [A, B]

my_model.temperature = 500.0

# B is driven by Dirichlet BCs so it develops a non-trivial profile the source can see
my_model.boundary_conditions = [
    F.FixedConcentrationBC(left, value=1.0, species=B),
    F.FixedConcentrationBC(right, value=0.0, species=B),
]

# source on A that depends on both the time and the concentration of B
my_model.sources = [
    F.ParticleSource(
        volume=vol,
        species=A,
        value=lambda t, B: t * B,
        species_dependent_value={"B": B},
    ),
]

my_model.settings = F.Settings(
    atol=1e-10, rtol=1e-10, final_time=10, stepsize=F.Stepsize(0.5)
)

my_model.exports = [
    F.VTXSpeciesExport("A.bp", field=A),
    F.VTXSpeciesExport("B.bp", field=B),
]

my_model.initialise()
my_model.run()

import festim as F
import numpy as np
import ufl

# simple 1D diffusion with changing profile
L = 100
vertices = np.linspace(0, L, 1000)

model = F.HydrogenTransportProblem()

H = F.Species("H")
model.species = [H]

model.mesh = F.Mesh1D(vertices)

material = F.Material(D_0=1.0, E_D=0.0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
left = F.SurfaceSubdomain1D(id=1, x=0)

model.subdomains = [vol, left]

model.temperature = 500

model.boundary_conditions = [F.FixedConcentrationBC(species=H, subdomain=left, value=0)]

# initial condition: step function
initial_concentration = lambda x: ufl.conditional(x[0] < 10, 1.0, 0.0)
model.initial_conditions = [
    F.InitialConcentration(value=initial_concentration, species=H, volume=vol)
]

profile_times = [1.0, 10.0]

profile = F.Profile1DExport(field=H, subdomain=vol, times=profile_times)
model.exports = [profile]

model.settings = F.Settings(atol=1e-5, rtol=1e-5, final_time=10.0)
model.settings.stepsize = F.Stepsize(
    initial_value=0.01,
    growth_factor=1.1,
    cutback_factor=0.5,
    target_nb_iterations=10,
    milestones=profile_times,
)


model.initialise()
model.run()

print("export times:", profile.t)
print("max absolute diff:", np.max(np.abs(profile.data[1] - profile.data[0])))

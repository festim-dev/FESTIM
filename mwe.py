from mpi4py import MPI

import numpy as np
from dolfinx.mesh import create_unit_square

import festim as F

my_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# Create materials for the two subdomains
mat_omega = F.Material(D_0=1e-5, E_D=0.0)
mat_gamma = F.Material(D_0=1e-6, E_D=0.0)


# Subdomains
omega = F.VolumeSubdomain(
    id=1, material=mat_omega, locator=lambda x: np.full_like(x[0], True), name="omega"
)
gamma = F.VolumeSubdomain(
    id=2,
    material=mat_gamma,
    dim=1,
    locator=lambda x: np.isclose(x[0], 0.0),
    name="gamma",
)
gamma_as_boundary = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 0.0))
# note: gamma is defined as both a volume subdomain and a surface subdomain

H_om = F.Species("H_om", subdomains=[omega])
H_gam = F.Species("H_gam", subdomains=[gamma])
species = [H_om, H_gam]


# Volumetric source in the gamma subdomain
source_in_gamma = F.ParticleSource(
    value=lambda x: x[1] * 10, species=H_gam, volume=gamma
)


# Coupling between the two subdomains
# here we need to provide the same expression k(c-c) as a ParticleSource and as a boundary condition
k = 0.1

# NOTE: this is tricky as it needs H_om but H_om doesn't "exist" on gamma. Later when calling
# convert_source_input_values_to_fenics_objects() it will look for H_gam.subdomain_to_solution[gamma] which doesn't exist.
# we need to find a way to tell it "use the solution of H_om on omega instead".
coupling_source_gamma = F.ParticleSource(
    value=lambda c_g, c_o: k * (c_o - c_g),
    species=H_gam,
    volume=gamma,
    species_dependent_value={"c_o": H_om, "c_g": H_gam},
)

# NOTE: same issue as above, but now we need H_gam on omega.
# We need to find a way to tell it "use the solution of H_gam on gamma instead".
coupling_source_omega = F.ParticleFluxBC(
    subdomain=gamma_as_boundary,
    value=lambda c_g, c_o: k * (c_g - c_o),
    species=H_om,
    species_dependent_value={"c_o": H_om, "c_g": H_gam},
)

my_model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(my_mesh),
    species=species,
    subdomains=[omega, gamma, gamma_as_boundary],
    sources=[],
    boundary_conditions=[],
    temperature=500,
)
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
my_model.initialise()

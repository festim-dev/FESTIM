from mpi4py import MPI

import numpy as np
import ufl
from dolfinx.mesh import create_unit_square

import festim as F

my_mesh = create_unit_square(MPI.COMM_WORLD, 20, 20)

# Create materials for the two subdomains
mat_omega = F.Material(D_0=1.5, E_D=0.0)
mat_gamma = F.Material(D_0=0.7, E_D=0.0)


# Subdomains. gamma has dim=1: it is a manifold embedded in the 2D mesh, carrying its
# own transport equation. It is tagged in the facet meshtags and can be used directly
# wherever a surface is expected.
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
right = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 1.0))

H_om = F.Species("H_om", subdomains=[omega])
H_gam = F.Species("H_gam", subdomains=[gamma])
species = [H_om, H_gam]

k = 2.0
beta = 1.0 - 1.5 / k

# Volumetric source in the gamma subdomain
source_in_gamma = F.ParticleSource(
    value=lambda x: (0.7 * np.pi**2 * beta - 1.5) * ufl.cos(np.pi * x[1]),
    species=H_gam,
    volume=gamma,
)

# Coupling between the two subdomains: the same flux J = k (c_omega - c_gamma) enters
# the gamma equation as a source and leaves omega as a flux boundary condition.
coupling_source_gamma = F.ParticleSource(
    value=lambda c_g, c_o: k * (c_o - c_g),
    species=H_gam,
    volume=gamma,
    species_dependent_value={"c_o": H_om, "c_g": H_gam},
)
coupling_flux_omega = F.ParticleFluxBC(
    subdomain=gamma,
    value=lambda c_g, c_o: k * (c_g - c_o),
    species=H_om,
    species_dependent_value={"c_o": H_om, "c_g": H_gam},
)

exact_omega = F.FixedConcentrationBC(
    subdomain=right,
    value=lambda x: 1 + x[0] ** 2 + (1 + x[0]) * ufl.cos(np.pi * x[1]),
    species=H_om,
)

source_omega = F.ParticleSource(
    value=lambda x: -1.5 * (2 - np.pi**2 * (1 + x[0]) * ufl.cos(np.pi * x[1])),
    species=H_om,
    volume=omega,
)

my_model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(my_mesh),
    species=species,
    subdomains=[omega, gamma, right],
    sources=[source_in_gamma, coupling_source_gamma, source_omega],
    boundary_conditions=[coupling_flux_omega, exact_omega],
    temperature=500,
    exports=[
        F.VTXSpeciesExport(filename="H_om.bp", field=H_om, subdomain=omega),
        F.VTXSpeciesExport(filename="H_gam.bp", field=H_gam, subdomain=gamma),
    ],
)
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
my_model.initialise()
my_model.run()

c_om = H_om.subdomain_to_post_processing_solution[omega]
c_gam = H_gam.subdomain_to_post_processing_solution[gamma]
print("c_omega range", c_om.x.array.min(), c_om.x.array.max())
print("c_gamma range", c_gam.x.array.min(), c_gam.x.array.max())
print("expected c_gamma range", 1 - beta, 1 + beta)

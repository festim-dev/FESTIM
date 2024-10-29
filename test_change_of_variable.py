import numpy as np
import festim as F

my_model = F.HydrogenTransportProblemDiscontinuousChangeVar()

interface_1 = 0.2
interface_2 = 0.8

vertices = np.concatenate(
    [
        np.linspace(0, interface_1, num=100),
        np.linspace(interface_1, interface_2, num=100),
        np.linspace(interface_2, 1, num=100),
    ]
)

my_model.mesh = F.Mesh1D(vertices)

material_left = F.Material(D_0=2.0, E_D=0.1, K_S_0=2.0, E_K_S=0)
material_mid = F.Material(D_0=2.0, E_D=0.1, K_S_0=4.0, E_K_S=0)
material_right = F.Material(D_0=2.0, E_D=0.1, K_S_0=6.0, E_K_S=0)

left_domain = F.VolumeSubdomain1D(3, borders=[0, interface_1], material=material_left)
middle_domain = F.VolumeSubdomain1D(
    4, borders=[interface_1, interface_2], material=material_mid
)
right_domain = F.VolumeSubdomain1D(5, borders=[interface_2, 1], material=material_right)

left_surface = F.SurfaceSubdomain1D(id=1, x=vertices[0])
right_surface = F.SurfaceSubdomain1D(id=2, x=vertices[-1])

my_model.subdomains = [
    left_domain,
    middle_domain,
    right_domain,
    left_surface,
    right_surface,
]

H = F.SpeciesChangeVar("H", mobile=True)
trapped_H = F.SpeciesChangeVar("H_trapped", mobile=False)
empty_trap = F.ImplicitSpecies(n=0.5, others=[trapped_H])

my_model.species = [H, trapped_H]

for species in [H, trapped_H]:
    species.subdomains = [left_domain, middle_domain, right_domain]

my_model.surface_to_volume = {right_surface: right_domain, left_surface: left_domain}


my_model.reactions = [
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=2,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=domain,
    )
    for domain in [left_domain, middle_domain, right_domain]
]


K_left = material_left.K_S_0
K_right = material_right.K_S_0

my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surface, value=0.05 / K_left, species=H),
    F.FixedConcentrationBC(right_surface, value=0.2 / K_right, species=H),
]


my_model.temperature = lambda x: 300 + 100 * x[0]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.exports = [
    F.VTXSpeciesExport(f"u_{field}.bp", field=field) for field in [H, trapped_H]
]
my_model.initialise()
my_model.run()


# print(my_model.u.x.array[:])
# print(H.post_processing_solution.x.array[:])
# print(trapped_H.post_processing_solution.x.array[:])

import numpy as np

import festim as F


def test_several_reactions_on_same_interface():
    my_model = F.HydrogenTransportProblemDiscontinuous()

    mat = F.Material(D_0=1, E_D=0)
    vol1 = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, 0.5])
    vol2 = F.VolumeSubdomain1D(id=2, material=mat, borders=[0.5, 1])
    left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))

    my_model.subdomains = [vol1, vol2, left]
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 21))

    A = F.Species("A", subdomains=[vol1])
    B = F.Species("B", subdomains=[vol2])
    C = F.Species("C", subdomains=[vol2])
    my_model.species = [A, B, C]

    my_model.interfaces = [
        F.InterfaceReaction(
            id=1,
            subdomains=[vol1, vol2],
            k_plus=1,
            k_minus=1,
            reactants=[A, A],
            products=[B],
        ),
        F.InterfaceReaction(
            id=1,
            subdomains=[vol1, vol2],
            k_plus=1,
            k_minus=1,
            reactants=[A],
            products=[C],
        ),
    ]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(species=A, subdomain=left, value=1),
    ]
    my_model.temperature = 300
    my_model.settings = F.Settings(final_time=1, atol=1e-10, rtol=1e-10, stepsize=0.1)

    my_model.initialise()
    my_model.run()

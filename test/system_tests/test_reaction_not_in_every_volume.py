import numpy as np

import festim as F


def test_sim_reaction_not_in_every_volume():
    """Tests that a steady simulation can be run if a reaction is not defined
    in every volume"""
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(np.linspace(0, 2e-04, num=1001))

    my_mat = F.Material(D_0=4.1e-07, E_D=0.37, name="my_mat")
    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 1e-04], material=my_mat)
    vol_2 = F.VolumeSubdomain1D(id=2, borders=[1e-04, 2e-04], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    my_model.subdomains = [vol_1, vol_2, left]

    cm, ct = F.Species("cm"), F.Species("ct", mobile=False)
    my_model.species = [cm, ct]
    trap = F.ImplicitSpecies(n=8.19e25, others=[ct])
    my_model.reactions = [
        F.Reaction(
            reactant=[cm, trap],
            product=ct,
            k_0=8.9e-17,
            E_k=0.39,
            p_0=1e13,
            E_p=0.87,
            volume=vol_1,
        ),
    ]

    my_model.temperature = 600

    my_model.settings = F.Settings(atol=1e10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

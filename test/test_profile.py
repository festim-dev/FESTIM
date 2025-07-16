import festim as F
import numpy as np


def test_profile():
    my_model = F.HydrogenTransportProblem()

    protium = F.Species("H")
    deuterium = F.Species("D")
    tritium = F.Species("T")
    my_model.species = [protium, deuterium, tritium]

    my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material = F.Material(D_0=1, E_D=0)

    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

    my_model.subdomains = [vol, left_surf, right_surf]

    my_model.boundary_conditions = [
        # Protium BCs
        F.FixedConcentrationBC(left_surf, value=10, species=protium),
        F.FixedConcentrationBC(right_surf, value=0, species=protium),
        # Deuterium BCs
        F.FixedConcentrationBC(left_surf, value=5, species=deuterium),
        F.FixedConcentrationBC(right_surf, value=0, species=deuterium),
        # Tritium BCs
        F.FixedConcentrationBC(left_surf, value=0, species=tritium),
        F.FixedConcentrationBC(right_surf, value=2, species=tritium),
    ]

    my_model.temperature = 300

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5)

    my_model.settings.stepsize = F.Stepsize(1)

    my_model.exports = [
        F.Profile1DExport(protium),
        F.Profile1DExport(deuterium),
    ]

    my_model.initialise()
    my_model.run()

    assert my_model.exports[0].x is not None
    assert my_model.exports[1].x is not None
    assert len(my_model.exports[0].data) > 0
    assert len(my_model.exports[1].data) > 0

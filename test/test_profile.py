import numpy as np
import pytest

import festim as F


@pytest.mark.parametrize("times", [None, [2, 5], [5]])
def test_profile(times):
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
        F.Profile1DExport(protium, times=times),
        F.Profile1DExport(deuterium, times=times),
    ]

    my_model.initialise()
    my_model.run()

    assert my_model.exports[0].x is not None
    assert my_model.exports[1].x is not None
    if times is None:
        assert len(my_model.exports[0].data) > 0
        assert len(my_model.exports[1].data) > 0
    else:
        assert len(my_model.exports[0].data) == len(times)
        assert len(my_model.exports[1].data) == len(times)


def test_profile_single_species():
    my_model = F.HydrogenTransportProblem()

    protium = F.Species("H")
    my_model.species = [protium]

    my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material = F.Material(D_0=1, E_D=0)

    my_model.species = [protium]

    my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material = F.Material(D_0=1, E_D=0)

    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

    my_model.subdomains = [vol, left_surf, right_surf]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(left_surf, value=10, species=protium),
        F.FixedConcentrationBC(right_surf, value=0, species=protium),
    ]

    my_model.temperature = 300

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5)

    my_model.settings.stepsize = F.Stepsize(1)

    my_model.exports = [
        F.Profile1DExport(protium),
    ]

    my_model.initialise()
    my_model.run()

    assert my_model.exports[0].x is not None
    assert len(my_model.exports[0].data) > 0


def test_profile_discontinuous():
    my_model = F.HydrogenTransportProblemDiscontinuous()

    protium = F.Species("H")
    deuterium = F.Species("D")
    tritium = F.Species("T")
    my_model.species = [protium, deuterium, tritium]

    vertices_left = np.linspace(0, 0.5, 50)
    vertices_right = np.linspace(0.5, 1, 50)
    vertices = np.concatenate((vertices_left, vertices_right))

    my_model.mesh = F.Mesh1D(vertices)

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material_left = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    material_right = F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0)

    vol1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=material_left)
    vol2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=material_right)

    my_model.interfaces = [
        F.Interface(id=3, subdomains=[vol1, vol2], penalty_term=1000)
    ]

    my_model.subdomains = [vol1, vol2, left_surf, right_surf]

    for spe in my_model.species:
        spe.subdomains = [vol1, vol2]

    my_model.surface_to_volume = {
        left_surf: vol1,
        right_surf: vol2,
    }

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

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=0.5)

    my_model.settings.stepsize = F.Stepsize(0.1)

    my_model.exports = [
        F.Profile1DExport(field=protium, subdomain=vol1),
        F.Profile1DExport(field=deuterium, subdomain=vol1),
        F.Profile1DExport(field=tritium, subdomain=vol1),
        F.Profile1DExport(field=protium, subdomain=vol2),
        F.Profile1DExport(field=deuterium, subdomain=vol2),
        F.Profile1DExport(field=tritium, subdomain=vol2),
    ]

    my_model.initialise()
    my_model.run()

    for export in my_model.exports:
        assert export.x is not None
        assert len(export.data) > 0


def test_profile_discontinuous_single_species():
    my_model = F.HydrogenTransportProblemDiscontinuous()

    protium = F.Species("H")
    my_model.species = [protium]

    vertices_left = np.linspace(0, 0.5, 50)
    vertices_right = np.linspace(0.5, 1, 50)
    vertices = np.concatenate((vertices_left, vertices_right))

    my_model.mesh = F.Mesh1D(vertices)

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material_left = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    material_right = F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0)

    vol1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=material_left)
    vol2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=material_right)

    my_model.interfaces = [
        F.Interface(id=3, subdomains=[vol1, vol2], penalty_term=1000)
    ]

    my_model.subdomains = [vol1, vol2, left_surf, right_surf]

    for spe in my_model.species:
        spe.subdomains = [vol1, vol2]

    my_model.surface_to_volume = {
        left_surf: vol1,
        right_surf: vol2,
    }

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(left_surf, value=10, species=protium),
        F.FixedConcentrationBC(right_surf, value=0, species=protium),
    ]

    my_model.temperature = 300

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=0.5)

    my_model.settings.stepsize = F.Stepsize(0.1)

    my_model.exports = [
        F.Profile1DExport(field=protium, subdomain=vol1),
        F.Profile1DExport(field=protium, subdomain=vol2),
    ]

    my_model.initialise()
    my_model.run()

    for export in my_model.exports:
        assert export.x is not None
        assert len(export.data) > 0

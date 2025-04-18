import dolfinx
import festim as F
import numpy as np
from .test_multi_mat_penalty import generate_mesh


def test_petsc_options():
    my_model = F.HydrogenTransportProblem(
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "coucoucou": 3,
        }
    )

    tungsten = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, num=100))
    vol1 = F.VolumeSubdomain1D(id=1, material=tungsten, borders=[0, 1])
    surface1 = F.SurfaceSubdomain1D(id=4, x=0)
    surface2 = F.SurfaceSubdomain1D(id=5, x=1)

    my_model.subdomains = [vol1, surface1, surface2]

    mobile = F.Species(name="H", mobile=True)

    my_model.species = [mobile]

    my_model.temperature = 600

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=surface1, species=mobile, value=1),
        F.FixedConcentrationBC(subdomain=surface2, species=mobile, value=0),
    ]

    my_model.settings = F.Settings(atol=1e-6, rtol=1e-6, transient=False)

    my_model.initialise()
    my_model.run()


def test_D_global_on_2d_mesh():
    """
    Test that the D_global is defined correctly on a 2D mesh with two different
    materials. The D_global should be defined as a piecewise constant function
    with two different values, one for each material.
    """
    mesh, mt, ct = generate_mesh(20)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_top = F.Material(D_0=5.0, E_D=0)
    material_bottom = F.Material(D_0=1.0, E_D=0)

    top_domain = F.VolumeSubdomain(4, material=material_top)
    bottom_domain = F.VolumeSubdomain(3, material=material_bottom)

    top_surface = F.SurfaceSubdomain(id=1)
    bottom_surface = F.SurfaceSubdomain(id=2)
    my_model.subdomains = [
        bottom_domain,
        top_domain,
        top_surface,
        bottom_surface,
    ]

    H = F.Species("H", mobile=True)
    my_model.species = [H]

    my_model.boundary_conditions = []

    my_model.temperature = 500.0

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()

    # run

    D, _ = my_model.define_D_global(species=H)

    # test that there are two values in D
    assert len(np.unique(D.x.array[:])) == 2


def test_min_max_vol_on_2d_mesh():
    """Added test that catches bug #908"""
    mesh, mt, ct = generate_mesh(10)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_bottom = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    material_top = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

    top_domain = F.VolumeSubdomain(4, material=material_top)
    bottom_domain = F.VolumeSubdomain(3, material=material_bottom)

    top_surface = F.SurfaceSubdomain(id=1)
    bottom_surface = F.SurfaceSubdomain(id=2)
    my_model.subdomains = [
        bottom_domain,
        top_domain,
        top_surface,
        bottom_surface,
    ]

    H = F.Species("H", mobile=True)

    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(top_surface, value=1, species=H),
        F.FixedConcentrationBC(bottom_surface, value=0, species=H),
    ]

    my_model.temperature = 500.0

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
    max_top = F.MaximumVolume(field=H, volume=top_domain)
    max_bottom = F.MaximumVolume(field=H, volume=bottom_domain)
    my_model.exports = [max_top, max_bottom]

    my_model.initialise()
    my_model.run()

    assert max_top.value != max_bottom.value

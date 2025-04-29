import dolfinx
import festim as F
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "species",
    [
        [F.Species("H", mobile=True)],
        [F.Species("H", mobile=True), F.Species("D", mobile=True)],
    ],
)
def test_min_max_vol_on_2d_mesh(species):
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

    H = species[0]

    my_model.species = species

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(top_surface, value=1, species=H),
        F.FixedConcentrationBC(bottom_surface, value=0, species=H),
    ]

    my_model.temperature = 500.0

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
    max_top = F.MaximumVolume(field=H, volume=top_domain)
    max_bottom = F.MaximumVolume(field=H, volume=bottom_domain)
    max_top_boundary = F.MaximumSurface(field=H, surface=top_surface)
    max_bottom_boundary = F.MaximumSurface(field=H, surface=bottom_surface)
    min_top = F.MinimumVolume(field=H, volume=top_domain)
    min_bottom = F.MinimumVolume(field=H, volume=bottom_domain)
    min_top_boundary = F.MinimumSurface(field=H, surface=top_surface)
    min_bottom_boundary = F.MinimumSurface(field=H, surface=bottom_surface)
    my_model.exports = [
        max_top,
        max_bottom,
        max_top_boundary,
        max_bottom_boundary,
        min_top,
        min_bottom,
        min_top_boundary,
        min_bottom_boundary,
    ]
    if len(species) == 2:
        my_model.boundary_conditions.append(
            F.FixedConcentrationBC(top_surface, value=0, species=species[1])
        )
        my_model.boundary_conditions.append(
            F.FixedConcentrationBC(bottom_surface, value=1, species=species[1])
        )
        my_model.exports += [
            F.MaximumVolume(field=species[1], volume=top_domain),
            F.MaximumVolume(field=species[1], volume=bottom_domain),
            F.MaximumSurface(field=species[1], surface=top_surface),
            F.MaximumSurface(field=species[1], surface=bottom_surface),
            F.MinimumVolume(field=species[1], volume=top_domain),
            F.MinimumVolume(field=species[1], volume=bottom_domain),
            F.MinimumSurface(field=species[1], surface=top_surface),
            F.MinimumSurface(field=species[1], surface=bottom_surface),
        ]

    my_model.initialise()
    my_model.run()

    assert max_top.value != max_bottom.value
    assert max_top_boundary.value != max_bottom_boundary.value
    assert min_top.value != min_bottom.value
    assert min_top_boundary.value != min_bottom_boundary.value

    if len(species) == 2:
        assert my_model.exports[-1].value != my_model.exports[-2].value
        assert my_model.exports[-1].value != min_bottom_boundary.value


def test_temp_dependent_bc_mixed_domain_temperature_as_function():
    """Test to catch bug 986"""
    mesh, mt, ct = generate_mesh(8)

    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    T = dolfinx.fem.Function(V)
    T.interpolate(lambda x: 800.0 + 100 * x[0] + x[1])

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_top = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    material_bottom = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

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

    my_model.surface_to_volume = {
        top_surface: top_domain,
        bottom_surface: bottom_domain,
    }

    H = F.Species("H", mobile=True, subdomains=[bottom_domain, top_domain])

    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(top_surface, value=lambda T: 2 * T, species=H),
        F.FixedConcentrationBC(bottom_surface, value=0, species=H),
    ]

    my_model.temperature = T

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [
        F.VTXSpeciesExport(field=H, filename="species.bp", subdomain=top_domain),
    ]

    my_model.initialise()
    my_model.run()

    expected_value = 2 * T.x.array[:].max()
    computed_value = (
        H.subdomain_to_post_processing_solution[top_domain].x.array[:].max()
    )
    assert np.isclose(computed_value, expected_value)

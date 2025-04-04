import festim as F
import ufl
import numpy as np
import pytest
import dolfinx.fem as fem
import basix

step_function_space = lambda x: ufl.conditional(ufl.gt(x[0], 0.5), 10, 0.0)
step_function_time_non_homogeneous = lambda x, t: ufl.conditional(
    ufl.gt(t, 50), 10 - 10 * x[0], 0.0
)
step_function_time_homogeneous = lambda t: 10.0 if t > 50 else 2.0
simple_space = lambda x: 10 - 10 * x[0]


@pytest.mark.parametrize(
    "density_func, expected_value",
    [
        (step_function_space, 5.0),
        (simple_space, 5.0),
        (step_function_time_non_homogeneous, 5.0),
        (step_function_time_homogeneous, 10.0),
    ],
)
def test_non_homogeneous_density(density_func, expected_value, tmpdir):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))
    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    trapped_H = F.Species("trapped_H", mobile=False)

    empty = F.ImplicitSpecies(n=density_func, name="empty", others=[trapped_H])
    my_model.species = [H, trapped_H]

    my_model.reactions = [
        F.Reaction(
            reactant=[H, empty],
            product=trapped_H,
            k_0=1,
            E_k=0,
            p_0=0,
            E_p=0,
            volume=vol,
        )
    ]

    my_model.temperature = 600

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=1, species=H),
        F.DirichletBC(subdomain=right, value=1, species=H),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=100)
    my_model.settings.stepsize = 1

    total_trapped = F.TotalVolume(field=trapped_H, volume=vol)
    my_model.exports = [
        F.VTXSpeciesExport(filename=tmpdir + "/trapped_c.bp", field=trapped_H),
        F.VTXSpeciesExport(filename=tmpdir + "/c.bp", field=H),
        total_trapped,
    ]

    my_model.initialise()
    my_model.run()

    print(total_trapped.value)
    print(expected_value)
    assert np.isclose(total_trapped.value, expected_value)


def test_density_as_function(tmpdir):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))
    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    trapped_H = F.Species("trapped_H", mobile=False)

    degree = 1
    element_CG = basix.ufl.element(
        basix.ElementFamily.P,
        my_model.mesh.mesh.basix_cell(),
        degree,
        basix.LagrangeVariant.equispaced,
    )

    density = fem.Function(fem.functionspace(my_model.mesh.mesh, element_CG))
    density.interpolate(lambda x: 10 - 10 * x[0])

    empty = F.ImplicitSpecies(n=density, name="empty", others=[trapped_H])
    my_model.species = [H, trapped_H]

    my_model.reactions = [
        F.Reaction(
            reactant=[H, empty],
            product=trapped_H,
            k_0=1,
            E_k=0,
            p_0=0,
            E_p=0,
            volume=vol,
        )
    ]

    my_model.temperature = 600

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=1, species=H),
        F.DirichletBC(subdomain=right, value=1, species=H),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=100)
    my_model.settings.stepsize = 1

    total_trapped = F.TotalVolume(field=trapped_H, volume=vol)
    my_model.exports = [
        F.VTXSpeciesExport(filename=tmpdir + "/trapped_c.bp", field=trapped_H),
        F.VTXSpeciesExport(filename=tmpdir + "/c.bp", field=H),
        total_trapped,
    ]

    my_model.initialise()
    my_model.run()

    expected_value = 5.0

    assert np.isclose(total_trapped.value, expected_value)

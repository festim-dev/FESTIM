import numpy as np

from dolfinx import fem

import festim as F

from .tools import error_L2


def test_run_MMS_cylindrical():
    """
    Tests that festim produces the correct concentration field in cylindrical
    coordinates
    """

    my_mesh = F.Mesh1D(vertices=np.linspace(1, 2, 500), coordinate_system="cylindrical")

    u_exact = lambda x: 1 + x[0] ** 2

    f = -4

    my_mat = F.Material(D_0=1.0, E_D=0)

    left = F.SurfaceSubdomain1D(id=1, x=1)
    right = F.SurfaceSubdomain1D(id=2, x=2)
    my_vol = F.VolumeSubdomain1D(id=3, borders=[1, 2], material=my_mat)

    my_subdomains = [my_vol, left, right]

    H = F.Species("H")
    D = F.Species("D")

    my_bcs = [
        F.FixedConcentrationBC(subdomain=left, value=u_exact, species=H),
        F.FixedConcentrationBC(subdomain=right, value=u_exact, species=H),
    ]

    my_temp = 500

    my_sources = [
        F.ParticleSource(value=f, volume=my_vol, species=H),
    ]

    my_settings = F.Settings(
        atol=1e-10,
        rtol=1e-9,
        max_iterations=50,
        transient=False,
    )

    my_exports = [
        F.VTXSpeciesExport("results/single_domain_u.bp", field=H, subdomain=my_vol),
    ]

    my_sim = F.HydrogenTransportProblem(
        mesh=my_mesh,
        species=[H, D],
        subdomains=my_subdomains,
        boundary_conditions=my_bcs,
        temperature=my_temp,
        sources=my_sources,
        settings=my_settings,
        exports=my_exports,
    )

    my_sim.initialise()
    my_sim.run()

    computed_solution = H.post_processing_solution

    L2_error = error_L2(computed_solution, u_exact)

    assert L2_error < 1e-6


def test_run_MMS_cylindrical_mixed_domain():
    """
    Tests that festim produces the correct concentration field in cylindrical
    coordinates in a discontinuous domain with two materials
    """

    my_model = F.HydrogenTransportProblemDiscontinuous()

    r_interface = 2
    left_domain = np.linspace(1, r_interface, num=1000)
    right_domain = np.linspace(r_interface, r_interface + 1, num=1000)

    vertices = np.concatenate(
        [
            left_domain,
            right_domain,
        ]
    )
    my_mesh = F.Mesh1D(vertices=vertices, coordinate_system="cylindrical")

    my_model.mesh = my_mesh

    K_S_left = 3.0
    K_S_right = 2.0

    def c_exact_left(x):
        return (r_interface - x[0]) ** 2 + 2

    def c_exact_right(x):
        return K_S_right / K_S_left * c_exact_left(x)

    lap_c = lambda r: 4 - 2 * r_interface / r
    D = 2.0

    mat_1 = F.Material(D_0=D, E_D=0, K_S_0=K_S_left, E_K_S=0, solubility_law="sievert")
    mat_2 = F.Material(D_0=D, E_D=0, K_S_0=K_S_right, E_K_S=0, solubility_law="sievert")

    left = F.SurfaceSubdomain1D(id=1, x=left_domain[0])
    right = F.SurfaceSubdomain1D(id=2, x=right_domain[-1])
    vol_1 = F.VolumeSubdomain1D(
        id=3, borders=[left_domain[0], left_domain[-1]], material=mat_1
    )
    vol_2 = F.VolumeSubdomain1D(
        id=4, borders=[right_domain[0], right_domain[-1]], material=mat_2
    )

    my_model.subdomains = [vol_1, vol_2, left, right]

    my_model.interfaces = [F.Interface(5, (vol_1, vol_2), penalty_term=100)]
    my_model.surface_to_volume = {
        left: vol_1,
        right: vol_2,
    }

    H = F.Species("H", mobile=True, subdomains=[vol_1, vol_2])
    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=c_exact_left, species=H),
        F.FixedConcentrationBC(subdomain=right, value=c_exact_right, species=H),
    ]

    my_model.temperature = 500

    f_left = lambda x: -D * lap_c(x[0])
    f_right = lambda x: -D * K_S_right / K_S_left * lap_c(x[0])

    my_model.sources = [
        F.ParticleSource(value=f_left, volume=vol_1, species=H),
        F.ParticleSource(value=f_right, volume=vol_2, species=H),
    ]

    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, max_iterations=10, transient=False, element_degree=1
    )

    my_model.exports = [
        F.VTXSpeciesExport("results/u_left.bp", field=H, subdomain=vol_1),
        F.VTXSpeciesExport("results/u_right.bp", field=H, subdomain=vol_2),
    ]

    my_model.initialise()
    my_model.run()

    c_l_computed = H.subdomain_to_post_processing_solution[vol_1]
    c_r_computed = H.subdomain_to_post_processing_solution[vol_2]

    L2_error_l = error_L2(c_l_computed, c_exact_left)
    L2_error_r = error_L2(c_r_computed, c_exact_right)

    V_l = fem.functionspace(vol_1.submesh, ("Lagrange", 2))
    V_r = fem.functionspace(vol_2.submesh, ("Lagrange", 2))
    c_l_exact = fem.Function(V_l)
    c_l_exact.interpolate(c_exact_left)
    c_r_exact = fem.Function(V_r)
    c_r_exact.interpolate(c_exact_right)

    from dolfinx.io import VTXWriter
    from mpi4py import MPI

    writer_l = VTXWriter(MPI.COMM_WORLD, "results/u_left_exact.bp", c_l_exact, "BP5")
    writer_l.write(t=0)

    writer_r = VTXWriter(MPI.COMM_WORLD, "results/u_right_exact.bp", c_r_exact, "BP5")
    writer_r.write(t=0)

    assert L2_error_l < 1e-2
    assert L2_error_r < 3e-2


def test_run_MMS_cylindrical_mixed_domain_one_subdomain():
    """
    Tests that festim produces the correct concentration field in cylindrical
    coordinates in a discontinuous domain with two materials
    """

    my_model = F.HydrogenTransportProblemDiscontinuous()

    my_mesh = F.Mesh1D(vertices=np.linspace(1, 2, 500), coordinate_system="cylindrical")

    my_model.mesh = my_mesh

    u_exact = lambda x: 1 + x[0] ** 2

    f = -4

    mat_1 = F.Material(D_0=1.0, E_D=0, K_S_0=1, E_K_S=0, solubility_law="sievert")

    left = F.SurfaceSubdomain1D(id=1, x=1)
    right = F.SurfaceSubdomain1D(id=2, x=2)
    vol_1 = F.VolumeSubdomain1D(id=3, borders=[1, 2], material=mat_1)

    my_model.subdomains = [vol_1, left, right]

    my_model.interfaces = []
    my_model.surface_to_volume = {
        left: vol_1,
        right: vol_1,
    }

    H = F.Species("H", mobile=True, subdomains=[vol_1])
    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=u_exact, species=H),
        F.FixedConcentrationBC(subdomain=right, value=u_exact, species=H),
    ]

    my_model.temperature = 500

    my_model.sources = [
        F.ParticleSource(value=f, volume=vol_1, species=H),
    ]

    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, max_iterations=10, transient=False, element_degree=1
    )

    my_model.exports = [
        F.VTXSpeciesExport("results/u.bp", field=H, subdomain=vol_1),
    ]

    my_model.initialise()

    print(vol_1.F)

    my_model.run()

    c_l_computed = H.subdomain_to_post_processing_solution[vol_1]

    L2_error = error_L2(c_l_computed, u_exact)

    V_l = fem.functionspace(my_model.mesh.mesh, ("Lagrange", 2))
    c_l_exact = fem.Function(V_l)
    c_l_exact.interpolate(u_exact)

    from dolfinx.io import VTXWriter
    from mpi4py import MPI

    writer_l = VTXWriter(MPI.COMM_WORLD, "results/u_exact.bp", c_l_exact, "BP5")
    writer_l.write(t=0)

    assert L2_error < 1e-2

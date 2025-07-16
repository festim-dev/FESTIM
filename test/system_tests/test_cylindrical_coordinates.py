import numpy as np
import ufl
from dolfinx import fem

import festim as F

from .test_multi_material import generate_mesh
from .tools import error_L2


def test_run_MMS_cylindrical():
    """
    Tests that festim produces the correct concentration field in cylindrical
    coordinates
    """

    my_mesh = F.Mesh1D(vertices=np.linspace(1, 2, 500), coordinate_system="cylindrical")
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))

    u_exact = lambda x: 1 + x[0] ** 2

    f = -4

    my_mat = F.Material(D_0=1.0, E_D=0)

    left = F.SurfaceSubdomain1D(id=1, x=1)
    right = F.SurfaceSubdomain1D(id=2, x=2)
    my_vol = F.VolumeSubdomain1D(id=3, borders=[1, 2], material=my_mat)

    my_subdomains = [my_vol, left, right]

    H = F.Species("H")

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

    my_sim = F.HydrogenTransportProblem(
        mesh=my_mesh,
        species=[H],
        subdomains=my_subdomains,
        boundary_conditions=my_bcs,
        temperature=my_temp,
        sources=my_sources,
        settings=my_settings,
    )

    my_sim.initialise()
    my_sim.run()

    computed_solution = H.post_processing_solution

    L2_error = error_L2(computed_solution, u_exact)

    assert L2_error < 1e-6


# def test_run_MMS_cylindrical_mixed_domain():
#     """
#     Tests that festim produces the correct concentration field in cylindrical
#     coordinates
#     """

#     my_model = F.HydrogenTransportProblemDiscontinuous()

#     vertices = np.concatenate(
#         [
#             np.linspace(0, 0.5, num=1000),
#             np.linspace(0.5, 1, num=1000),
#         ]
#     )
#     my_mesh = F.Mesh1D(vertices=vertices, coordinate_system="cylindrical")

#     my_model.mesh = my_mesh

#     K_S_left = 3.0
#     K_S_right = 6.0

#     def c_exact_left_ufl(x):
#         return 1 + ufl.sin(ufl.pi * (2 * x[0] + 0.5))

#     def c_exact_right_ufl(x):
#         return K_S_left / K_S_right * c_exact_left_ufl(x)

#     def c_exact_left_np(x):
#         return 1 + np.sin(np.pi * (2 * x[0] + 0.5))

#     def c_exact_right_np(x):
#         return K_S_left / K_S_right * c_exact_left_np(x)

#     mat_1 = F.Material(D_0=2.0, E_D=0, K_S_0=K_S_left, E_K_S=0)
#     mat_2 = F.Material(D_0=4.0, E_D=0, K_S_0=K_S_right, E_K_S=0)

#     left = F.SurfaceSubdomain1D(id=1, x=0)
#     right = F.SurfaceSubdomain1D(id=2, x=1)
#     vol_1 = F.VolumeSubdomain1D(id=3, borders=[0, 0.5], material=mat_1)
#     vol_2 = F.VolumeSubdomain1D(id=4, borders=[0.5, 1], material=mat_2)

#     my_model.subdomains = [vol_1, vol_2, left, right]

#     my_model.interfaces = [F.Interface(5, (vol_1, vol_2))]
#     my_model.surface_to_volume = {
#         left: vol_1,
#         right: vol_2,
#     }

#     H = F.Species("H", mobile=True, subdomains=[vol_1, vol_2])
#     my_model.species = [H]

#     my_model.boundary_conditions = [
#         F.FixedConcentrationBC(subdomain=left, value=c_exact_left_ufl, species=H),
#         F.FixedConcentrationBC(subdomain=right, value=c_exact_right_ufl, species=H),
#     ]

#     my_model.temperature = 500

#     def source_left_val(x):
#         return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0])

#     def source_right_val(x):
#         return 40 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0])

#     my_model.sources = [
#         F.ParticleSource(value=source_left_val, volume=vol_1, species=H),
#         F.ParticleSource(value=source_right_val, volume=vol_2, species=H),
#     ]

#     my_model.settings = F.Settings(
#         atol=1e-10, rtol=1e-10, max_iterations=10, transient=False
#     )

#     my_model.exports = [
#         F.VTXSpeciesExport("results/u_left.bp", field=H, subdomain=vol_1),
#         F.VTXSpeciesExport("results/u_right.bp", field=H, subdomain=vol_2),
#     ]

#     my_model.initialise()
#     my_model.run()

#     c_l_computed = H.subdomain_to_post_processing_solution[vol_1]
#     c_r_computed = H.subdomain_to_post_processing_solution[vol_2]

#     L2_error_l = error_L2(c_l_computed, c_exact_left_np)
#     L2_error_r = error_L2(c_r_computed, c_exact_right_np)

#     left_mesh = F.Mesh1D(vertices=np.linspace(0, 0.5, num=1000))
#     V_l = fem.functionspace(left_mesh.mesh, ("Lagrange", 1))
#     right_mesh = F.Mesh1D(vertices=np.linspace(0.5, 1, num=1000))
#     V_r = fem.functionspace(right_mesh.mesh, ("Lagrange", 1))
#     c_l_exact = fem.Function(V_l)
#     c_l_exact.interpolate(c_exact_left_np)
#     c_r_exact = fem.Function(V_r)
#     c_r_exact.interpolate(c_exact_right_np)

#     from dolfinx.io import VTXWriter
#     from mpi4py import MPI

#     writer_l = VTXWriter(MPI.COMM_WORLD, "results/u_left_exact.bp", c_l_exact, "BP5")
#     writer_l.write(t=0)

#     writer_r = VTXWriter(MPI.COMM_WORLD, "results/u_right_exact.bp", c_r_exact, "BP5")
#     writer_r.write(t=0)

#     assert L2_error_l < 1e-3
#     assert L2_error_r < 1e-3

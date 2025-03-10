import numpy as np
import ufl
from dolfinx import fem
import festim as F
import basix
from dolfinx.mesh import locate_entities
from festim.helpers import nmm_interpolate

from .tools import error_L2, test_mesh_2d, x_2d


def test_MMS_coupled_problem():
    """MMS coupled heat and hydrogen test with 1 mobile species and 1 trap in a 1s
    transient, the values of the temperature, mobile and trapped solutions at the last
    time step is compared to an analytical solution"""

    class LefSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0))
            return indices

    class RightSurface(F.SurfaceSubdomain):
        def locate_boundary_facet_indices(self, mesh):
            fdim = mesh.topology.dim - 1
            indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1))
            return indices

    # coupled simulation properties
    D_0, E_D = 1.2, 0.1
    k_0, E_k = 2.2, 0.2
    p_0, E_p = 0.5, 0.1
    n_trap = 5
    k_B = F.k_B

    # common festim objects
    test_mat = F.Material(D_0=D_0, E_D=E_D)
    test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)
    left_sub = LefSurface(id=1)
    right_sub = RightSurface(id=2)
    test_mobile = F.Species("mobile", mobile=True)
    test_trapped = F.Species(name="trapped", mobile=False, subdomains=[test_vol_sub])
    test_traps = F.ImplicitSpecies(n=n_trap, others=[test_mobile, test_trapped])

    V = fem.functionspace(test_mesh_2d, ("Lagrange", 1))
    T = fem.Function(V)
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)

    # create velocity field
    v_cg = basix.ufl.element(
        "Lagrange",
        test_mesh_2d.topology.cell_name(),
        2,
        shape=(test_mesh_2d.geometry.dim,),
    )
    V_velocity = fem.functionspace(test_mesh_2d, v_cg)
    u = fem.Function(V_velocity)

    def velocity_func(x):
        values = np.zeros((2, x.shape[1]))  # Initialize with zeros

        scalar_value = 5**2 * (-10 * x[1] * (x[1] - 1))  # Compute the scalar function
        values[0] = scalar_value  # Assign to first component
        values[1] = 0  # Second component remains zero

        return values

    u.interpolate(velocity_func)

    # define hydrogen problem
    exact_mobile_solution = lambda x: 2 * x[0] ** 2
    exact_trapped_solution = lambda x: 4 * x[0] ** 2

    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = -ufl.div(D * ufl.grad(exact_mobile_solution(x_2d))) - ufl.inner(
        u, ufl.grad(exact_mobile_solution(x_2d))
    )

    g = k * exact_mobile_solution(x_2d) * (
        n_trap - exact_trapped_solution(x_2d)
    ) - p * exact_trapped_solution(x_2d)

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=F.Mesh(test_mesh_2d),
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(
                subdomain=left_sub,
                value=exact_mobile_solution,
                species=test_mobile,
            ),
            F.FixedConcentrationBC(
                subdomain=right_sub,
                value=exact_mobile_solution,
                species=test_mobile,
            ),
            # F.FixedConcentrationBC(
            #     subdomain=left_sub,
            #     value=exact_trapped_solution,
            #     species=test_trapped,
            # ),
            # F.FixedConcentrationBC(
            #     subdomain=right_sub,
            #     value=exact_trapped_solution,
            #     species=test_trapped,
            # ),
        ],
        # species=[test_mobile, test_trapped],
        species=[test_mobile],
        temperature=T,
        # reactions=[
        #     F.Reaction(
        #         reactant=[test_mobile, test_traps],
        #         product=test_trapped,
        #         k_0=k_0,
        #         E_k=E_k,
        #         p_0=p_0,
        #         E_p=E_p,
        #         volume=test_vol_sub,
        #     )
        # ],
        sources=[
            F.ParticleSource(value=f, volume=test_vol_sub, species=test_mobile),
            # F.ParticleSource(value=g, volume=test_vol_sub, species=test_trapped),
        ],
        advection_terms=[
            F.AdvectionTerm(velocity=u, subdomain=test_vol_sub, species=test_mobile)
        ],
        settings=F.Settings(
            atol=1e-10,
            rtol=1e-10,
            transient=False,
        ),
    )

    test_hydrogen_problem.initialise()
    test_hydrogen_problem.run()

    # compare computed values with exact solutions
    mobile_computed = test_mobile.post_processing_solution
    trapped_computed = test_trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, exact_mobile_solution)
    L2_error_trapped = error_L2(trapped_computed, exact_trapped_solution)

    print(L2_error_mobile)
    print(L2_error_trapped)
    quit()

    # # TEST ensure L2 error below 2e-7
    # assert L2_error_mobile < 2e-07
    # assert L2_error_trapped < 2e-07


# def test_coupled_problem_non_matching_mesh():
#     """MMS coupled heat and hydrogen test with 1 mobile species in a 1s transient with
#     mismatched meshes, the value of the mobile solution at the last time step is
#     compared to an analytical solution"""

#     # coupled simulation properties
#     density, heat_capacity = 1.3, 2.3
#     thermal_conductivity = 4.1
#     D_0, E_D = 1.3, 0.2
#     k_B = F.k_B
#     final_time = 5

#     # common festim objects
#     test_mat = F.Material(
#         D_0=D_0,
#         E_D=E_D,
#         thermal_conductivity=thermal_conductivity,
#         density=density,
#         heat_capacity=heat_capacity,
#     )
#     test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)
#     left_sub = F.SurfaceSubdomain1D(id=2, x=0)
#     right_sub = F.SurfaceSubdomain1D(id=3, x=1)
#     test_mobile = F.Species("mobile", mobile=True)

#     # define temperature sim
#     exact_T_solution = lambda x, t: 2 * x[0] ** 2 + 5 * t

#     dTdt = 5

#     mms_T_source = (
#         test_mat.density * test_mat.heat_capacity * dTdt
#         - test_mat.thermal_conductivity * 4
#     )

#     # Corse mesh for heat trasnfer problem
#     test_mesh = F.Mesh1D(vertices=np.linspace(0, 1, 500))

#     test_heat_problem = F.HeatTransferProblem(
#         mesh=test_mesh,
#         subdomains=[test_vol_sub, left_sub, right_sub],
#         boundary_conditions=[
#             F.FixedTemperatureBC(subdomain=left_sub, value=exact_T_solution),
#             F.FixedTemperatureBC(subdomain=right_sub, value=exact_T_solution),
#         ],
#         sources=[F.HeatSource(value=mms_T_source, volume=test_vol_sub)],
#         initial_condition=F.InitialTemperature(lambda x: exact_T_solution(x, 0)),
#         settings=F.Settings(
#             atol=1e-10,
#             rtol=1e-10,
#             transient=True,
#             final_time=final_time,
#             stepsize=final_time / 20,
#         ),
#     )

#     # define hydrogen problem
#     exact_mobile_solution = lambda x, t: 4 * x[0] ** 2 + 10 * t

#     dmobiledt = 10

#     D = lambda x, t: D_0 * ufl.exp(-E_D / (k_B * exact_T_solution(x, t)))

#     def mms_mobile_source(x, t):
#         return dmobiledt - ufl.div(D(x, t) * ufl.grad(exact_mobile_solution(x, t)))

#     # Fine mesh for hydrogen transport problem
#     test_mesh_2 = F.Mesh1D(vertices=np.linspace(0, 1, 2000))

#     test_hydrogen_problem = F.HydrogenTransportProblem(
#         mesh=test_mesh_2,
#         subdomains=[test_vol_sub, left_sub, right_sub],
#         boundary_conditions=[
#             F.FixedConcentrationBC(
#                 subdomain=left_sub, value=exact_mobile_solution, species=test_mobile
#             ),
#             F.FixedConcentrationBC(
#                 subdomain=right_sub, value=exact_mobile_solution, species=test_mobile
#             ),
#         ],
#         species=[test_mobile],
#         initial_conditions=[
#             F.InitialCondition(
#                 value=lambda x: exact_mobile_solution(x, t=0), species=test_mobile
#             ),
#         ],
#         sources=[
#             F.ParticleSource(
#                 value=mms_mobile_source, volume=test_vol_sub, species=test_mobile
#             ),
#         ],
#         settings=F.Settings(
#             atol=1e-10,
#             rtol=1e-10,
#             transient=True,
#             final_time=final_time,
#             stepsize=final_time / 20,
#         ),
#     )

#     test_coupled_problem = F.CoupledTransientHeatTransferHydrogenTransport(
#         heat_problem=test_heat_problem,
#         hydrogen_problem=test_hydrogen_problem,
#     )

#     test_coupled_problem.initialise()
#     test_coupled_problem.run()

#     exact_solution = lambda x: exact_mobile_solution(x, t=final_time)

#     computed_solution = test_mobile.post_processing_solution

#     L2_error = error_L2(computed_solution, exact_solution)

#     assert L2_error < 2e-07

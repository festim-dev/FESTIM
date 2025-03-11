import numpy as np
import ufl
from dolfinx import fem
import festim as F
import basix
from .tools import error_L2

from dolfinx.mesh import create_unit_square
from mpi4py import MPI


def test_MMS_coupled_problem():
    """MMS coupled heat and hydrogen test with 1 mobile species and 1 trap in a 1s
    transient, the values of the temperature, mobile and trapped solutions at the last
    time step is compared to an analytical solution"""

    test_mesh_2d = create_unit_square(MPI.COMM_WORLD, 200, 200)
    x_2d = ufl.SpatialCoordinate(test_mesh_2d)

    # coupled simulation properties
    D_0, E_D = 1.2, 0.1
    k_0, E_k = 2.2, 0.6
    p_0, E_p = 0.5, 0.1
    n_trap = 100
    k_B = F.k_B

    # common festim objects
    test_mat = F.Material(D_0=D_0, E_D=E_D)
    test_vol_sub = F.VolumeSubdomain(id=1, material=test_mat)

    boundary = F.SurfaceSubdomain(id=1)
    test_mobile = F.Species("mobile", mobile=True)
    test_trapped = F.Species(name="trapped", mobile=True)
    empty_trap = F.ImplicitSpecies(n=n_trap, others=[test_trapped])

    V = fem.functionspace(test_mesh_2d, ("Lagrange", 1))
    T = fem.Function(V)
    T_expr = lambda x: 100 + 200 * x[0] + 100 * x[1]
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

        scalar_value = x[1] * (x[1] - 1)  # Compute the scalar function
        values[0] = scalar_value  # Assign to first component
        values[1] = 0  # Second component remains zero

        return values

    u.interpolate(velocity_func)

    # define hydrogen problem
    exact_mobile_solution = lambda x: 200 * x[0] ** 2 + 300 * x[1] ** 2
    exact_trapped_solution = lambda x: 10 * x[0] ** 2 + 10 * x[1] ** 2

    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(exact_mobile_solution(x_2d)))
        + ufl.inner(u, ufl.grad(exact_mobile_solution(x_2d)))
        + k * exact_mobile_solution(x_2d) * (n_trap - exact_trapped_solution(x_2d))
        - p * exact_trapped_solution(x_2d)
    )

    g = (
        -ufl.div(D * ufl.grad(exact_trapped_solution(x_2d)))
        - k * exact_mobile_solution(x_2d) * (n_trap - exact_trapped_solution(x_2d))
        + p * exact_trapped_solution(x_2d)
    )

    my_bcs = []
    for species, value in zip(
        [test_mobile, test_trapped], [exact_mobile_solution, exact_trapped_solution]
    ):
        my_bcs.append(
            F.FixedConcentrationBC(subdomain=boundary, value=value, species=species)
        )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=F.Mesh(test_mesh_2d),
        subdomains=[test_vol_sub, boundary],
        boundary_conditions=my_bcs,
        species=[test_mobile, test_trapped],
        temperature=T,
        reactions=[
            F.Reaction(
                reactant=[test_mobile, empty_trap],
                product=test_trapped,
                k_0=k_0,
                E_k=E_k,
                p_0=p_0,
                E_p=E_p,
                volume=test_vol_sub,
            )
        ],
        sources=[
            F.ParticleSource(value=f, volume=test_vol_sub, species=test_mobile),
            F.ParticleSource(value=g, volume=test_vol_sub, species=test_trapped),
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

    assert L2_error_mobile < 2e-03
    assert L2_error_trapped < 7e-05

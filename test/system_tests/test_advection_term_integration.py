from mpi4py import MPI

import basix
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
import dolfinx

import festim as F

from .tools import error_L2


def test_MMS_1_species_1_trap_with_advection():
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


def test_multi_material_with_advection():
    def generate_mesh(n=20):
        def bottom_boundary(x):
            return np.isclose(x[1], 0.0)

        def top_boundary(x):
            return np.isclose(x[1], 1.0)

        def bottom_left_boundary(x):
            return np.logical_and(np.isclose(x[0], 0.0), x[1] <= 0.5 + 1e-14)

        def half(x):
            return x[1] <= 0.5 + 1e-14

        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle
        )

        # Split domain in half and set an interface tag of 5
        tdim = mesh.topology.dim
        fdim = tdim - 1
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
        bottom_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, bottom_boundary
        )
        bot_left_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, bottom_left_boundary
        )
        num_facets_local = (
            mesh.topology.index_map(fdim).size_local
            + mesh.topology.index_map(fdim).num_ghosts
        )
        facets = np.arange(num_facets_local, dtype=np.int32)
        values = np.full_like(facets, 0, dtype=np.int32)
        values[top_facets] = 1
        values[bottom_facets] = 2
        values[bot_left_facets] = 3

        bottom_cells = dolfinx.mesh.locate_entities(mesh, tdim, half)
        num_cells_local = (
            mesh.topology.index_map(tdim).size_local
            + mesh.topology.index_map(tdim).num_ghosts
        )
        cells = np.full(num_cells_local, 5, dtype=np.int32)
        cells[bottom_cells] = 4
        ct = dolfinx.mesh.meshtags(
            mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
        )
        all_b_facets = dolfinx.mesh.compute_incident_entities(
            mesh.topology, ct.find(4), tdim, fdim
        )
        all_t_facets = dolfinx.mesh.compute_incident_entities(
            mesh.topology, ct.find(5), tdim, fdim
        )
        interface = np.intersect1d(all_b_facets, all_t_facets)
        values[interface] = 6

        mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)
        return mesh, mt, ct

    mesh, mt, ct = generate_mesh()

    # create velocity field
    def create_velocity_field():
        def velocity_func(x):
            values = np.zeros((2, x.shape[1]))  # Initialize with zeros
            scalar_value = -500 * x[1] * (x[1] - 0.5)  # Compute the scalar function
            values[0] = scalar_value  # Assign to first component
            values[1] = 0  # Second component remains zero
            return values

        mesh2, mt2, ct2 = generate_mesh(n=10)
        submesh, submesh_to_mesh, v_map = dolfinx.mesh.create_submesh(
            mesh2, ct2.dim, ct2.find(4)
        )[0:3]
        v_cg = basix.ufl.element(
            "Lagrange", submesh.topology.cell_name(), 2, shape=(submesh.geometry.dim,)
        )
        V_velocity = dolfinx.fem.functionspace(submesh, v_cg)
        u = dolfinx.fem.Function(V_velocity)
        u.interpolate(velocity_func)
        return u

    u = create_velocity_field()

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_bottom = F.Material(D_0=1, E_D=0 * 0.1)
    material_top = F.Material(D_0=1, E_D=0 * 0.1)

    material_bottom.K_S_0 = 2.0
    material_bottom.E_K_S = 0
    material_top.K_S_0 = 600
    material_top.E_K_S = 0

    top_domain = F.VolumeSubdomain(5, material=material_top)
    bottom_domain = F.VolumeSubdomain(4, material=material_bottom)

    top_surface = F.SurfaceSubdomain(id=1)
    bottom_surface = F.SurfaceSubdomain(id=2)
    inlet_surf = F.SurfaceSubdomain(id=3)
    my_model.subdomains = [
        bottom_domain,
        top_domain,
        top_surface,
        bottom_surface,
        inlet_surf,
    ]

    # we should be able to automate this
    my_model.interfaces = [F.Interface(6, (bottom_domain, top_domain))]
    my_model.surface_to_volume = {
        top_surface: top_domain,
        bottom_surface: bottom_domain,
        inlet_surf: bottom_domain,
    }

    H = F.Species("H", mobile=True)

    my_model.species = [H]

    for species in my_model.species:
        species.subdomains = [bottom_domain, top_domain]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(top_surface, value=0.0, species=H),
        F.FixedConcentrationBC(inlet_surf, value=1.0, species=H),
        F.FixedConcentrationBC(bottom_surface, value=0.0, species=H),
    ]

    my_model.advection_terms = [
        F.AdvectionTerm(velocity=u, species=[H], subdomain=bottom_domain)
    ]

    my_model.temperature = 500.0

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

import festim as F
import numpy as np
from dolfinx import fem
import ufl
from tools import error_L2
from dolfinx.mesh import meshtags, create_unit_square, create_unit_cube, locate_entities
from mpi4py import MPI


test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
test_mesh_2d = create_unit_square(MPI.COMM_WORLD, 50, 50)
test_mesh_3d = create_unit_cube(MPI.COMM_WORLD, 20, 20, 20)
x_1d = ufl.SpatialCoordinate(test_mesh_1d.mesh)
x_2d = ufl.SpatialCoordinate(test_mesh_2d)
x_3d = ufl.SpatialCoordinate(test_mesh_3d)


def test_1_mobile_1_trap_MMS_steady_state():
    """
    MMS test with one mobile species and one trap at steady state
    """

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0])

    def v_exact(mod):
        return lambda x: mod.sin(3 * mod.pi * x[0])

    mobile_analytical_ufl = u_exact(ufl)
    mobile_analytical_np = u_exact(np)
    trapped_analytical_ufl = v_exact(ufl)
    trapped_analytical_np = v_exact(np)

    elements = ufl.FiniteElement("P", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)
    f = fem.Function(V)
    g = fem.Function(V)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(mobile_analytical_ufl(x_1d)))
        + k * mobile_analytical_ufl(x_1d) * (n_trap - trapped_analytical_ufl(x_1d))
        - p * trapped_analytical_ufl(x_1d)
    )

    g = p * trapped_analytical_ufl(x_1d) - k * mobile_analytical_ufl(x_1d) * (
        n_trap - trapped_analytical_ufl(x_1d)
    )

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=1)
    my_model.subdomains = [vol, left, right]

    mobile = F.Species("mobile")
    trapped = F.Species("trapped", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[trapped])
    my_model.species = [mobile, trapped]

    my_model.reactions = [
        F.Reaction(
            reactant1=mobile,
            reactant2=traps,
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_analytical_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_analytical_ufl, species=trapped),
    ]

    my_model.sources = [
        F.Source(value=f, volume=vol, species=mobile),
        F.Source(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_analytical_np)
    L2_error_trapped = error_L2(trapped_computed, trapped_analytical_np)

    assert L2_error_mobile < 2e-07
    assert L2_error_trapped < 1e-07


def test_1_mobile_1_trap_MMS_transient():
    """
    MMS test with one mobile species in 0.1s transient. Analytical solution
    """

    final_time = 0.1

    def u_exact(mod):
        return lambda x, t: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * t**2

    def u_exact_alt(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * final_time**2

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact_alt(np)

    elements = ufl.FiniteElement("P", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 600 + 50 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    # FESTIM model

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    init_value = lambda x: 1 + ufl.sin(2 * ufl.pi * x[0])
    my_model.initial_conditions = [F.InitialCondition(value=init_value, species=H)]

    f = lambda x, t: 4 * t - ufl.div(D * ufl.grad(H_analytical_ufl(x_1d, t)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=final_time)
    my_model.settings.stepsize = final_time / 50

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 5e-4



def test_1_mobile_1_trap_MMS_2D():
    """Tests a 2D MMS problem with one mobile species"""

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])
    
    def v_exact(mod):
        return lambda x: mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])


    mobile_analytical_ufl = u_exact(ufl)
    mobile_analytical_np = u_exact(np)
    trapped_analytical_ufl = v_exact(ufl)
    trapped_analytical_np = v_exact(np)

    elements = ufl.FiniteElement("P", test_mesh_3d.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_3d, elements)
    T = fem.Function(V)
    f = fem.Function(V)
    g = fem.Function(V)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(mobile_analytical_ufl(x_3d)))
        + k * mobile_analytical_ufl(x_3d) * (n_trap - trapped_analytical_ufl(x_3d))
        - p * trapped_analytical_ufl(x_3d)
    )

    g = p * trapped_analytical_ufl(x_3d) - k * mobile_analytical_ufl(x_3d) * (
        n_trap - trapped_analytical_ufl(x_3d)
    )

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_3d)

    # create facet meshtags 
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[0], 1))
    ]
    facet_indices, facet_markers = [], []
    fdim = test_mesh_3d.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(test_mesh_3d, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    my_facet_meshtags = meshtags(test_mesh_3d, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    # create volume meshtags
    vdim = test_mesh_3d.topology.dim
    num_cells = test_mesh_3d.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    tags_volumes = np.full(num_cells, 1, dtype=np.int32)
    my_volume_meshtags = meshtags(
        test_mesh_3d, vdim, mesh_cell_indices, tags_volumes
    )

    my_model.facet_meshtags = my_facet_meshtags
    my_model.volume_meshtags = my_volume_meshtags

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    left = F.SurfaceSubdomain(id=1)
    right = F.SurfaceSubdomain(id=2)

    my_model.subdomains = [vol, left, right]

    mobile = F.Species("mobile")
    trapped = F.Species("trapped", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[trapped])
    my_model.species = [mobile, trapped]

    my_model.reactions = [
        F.Reaction(
            reactant1=mobile,
            reactant2=traps,
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_analytical_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_analytical_ufl, species=trapped),
    ]

    my_model.sources = [
        F.Source(value=f, volume=vol, species=mobile),
        F.Source(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_analytical_np)
    L2_error_trapped = error_L2(trapped_computed, trapped_analytical_np)

    assert L2_error_mobile < 3e-02
    assert L2_error_trapped < 9e-03


def test_1_mobile_MMS_3D():
    """Tests a 3D MMS problem with one mobile species"""

    def u_exact(mod):
        return lambda x: 1.5 + mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])
    
    def v_exact(mod):
        return lambda x: mod.sin(3 * mod.pi * x[0]) + mod.cos(3 * mod.pi * x[1])


    mobile_analytical_ufl = u_exact(ufl)
    mobile_analytical_np = u_exact(np)
    trapped_analytical_ufl = v_exact(ufl)
    trapped_analytical_np = v_exact(np)

    elements = ufl.FiniteElement("P", test_mesh_2d.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_2d, elements)
    T = fem.Function(V)
    f = fem.Function(V)
    g = fem.Function(V)

    k_0 = 2
    E_k = 1.5
    p_0 = 0.2
    E_p = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    n_trap = 3
    E_D = 0.1
    D_0 = 2
    k_B = F.k_B
    D = D_0 * ufl.exp(-E_D / (k_B * T))
    k = k_0 * ufl.exp(-E_k / (k_B * T))
    p = p_0 * ufl.exp(-E_p / (k_B * T))

    f = (
        -ufl.div(D * ufl.grad(mobile_analytical_ufl(x_2d)))
        + k * mobile_analytical_ufl(x_2d) * (n_trap - trapped_analytical_ufl(x_2d))
        - p * trapped_analytical_ufl(x_2d)
    )

    g = p * trapped_analytical_ufl(x_2d) - k * mobile_analytical_ufl(x_2d) * (
        n_trap - trapped_analytical_ufl(x_2d)
    )

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_2d)

    # create facet meshtags 
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[0], 1))
    ]
    facet_indices, facet_markers = [], []
    fdim = test_mesh_2d.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(test_mesh_2d, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    my_facet_meshtags = meshtags(test_mesh_2d, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    # create volume meshtags
    vdim = test_mesh_2d.topology.dim
    num_cells = test_mesh_2d.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    tags_volumes = np.full(num_cells, 1, dtype=np.int32)
    my_volume_meshtags = meshtags(
        test_mesh_2d, vdim, mesh_cell_indices, tags_volumes
    )

    my_model.facet_meshtags = my_facet_meshtags
    my_model.volume_meshtags = my_volume_meshtags

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    left = F.SurfaceSubdomain(id=1)
    right = F.SurfaceSubdomain(id=2)

    my_model.subdomains = [vol, left, right]

    mobile = F.Species("mobile")
    trapped = F.Species("trapped", mobile=False)
    traps = F.ImplicitSpecies(n=n_trap, others=[trapped])
    my_model.species = [mobile, trapped]

    my_model.reactions = [
        F.Reaction(
            reactant1=mobile,
            reactant2=traps,
            product=trapped,
            k_0=k_0,
            E_k=E_k,
            p_0=p_0,
            E_p=E_p,
            volume=vol,
        )
    ]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=right, value=mobile_analytical_ufl, species=mobile),
        F.DirichletBC(subdomain=left, value=trapped_analytical_ufl, species=trapped),
        F.DirichletBC(subdomain=right, value=trapped_analytical_ufl, species=trapped),
    ]

    my_model.sources = [
        F.Source(value=f, volume=vol, species=mobile),
        F.Source(value=g, volume=vol, species=trapped),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    mobile_computed = mobile.post_processing_solution
    trapped_computed = trapped.post_processing_solution

    L2_error_mobile = error_L2(mobile_computed, mobile_analytical_np)
    L2_error_trapped = error_L2(trapped_computed, trapped_analytical_np)

    assert L2_error_mobile < 4e-03
    assert L2_error_trapped < 2e-03


def test_1_mobile_MMS_multivolume():
    """Tests that a steady simulation can be run with multiple volumes"""

    def u_exact(mod):
        return lambda x: 1 + mod.cos(2 * mod.pi * x[0])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=my_mat)
    vol_2 = F.VolumeSubdomain1D(id=4, borders=[0.5, 1], material=my_mat)

    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol_1, vol_2, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_1d)))
    my_model.sources = [
        F.Source(value=f, volume=vol_1, species=H),
        F.Source(value=f, volume=vol_2, species=H),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [F.XDMFExport(filename="test_output.xdmf", field=H)]

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 6e-4



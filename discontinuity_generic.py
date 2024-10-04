from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx.cpp.fem import compute_integration_domains

import ufl
import numpy as np
import basix
import dolfinx.fem.petsc

class NewtonSolver:
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec

    def __init__(
        self,
        F: list[dolfinx.fem.form],
        J: list[list[dolfinx.fem.form]],
        w: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        problem_prefix="newton",
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def solve(self, tol=1e-6, beta=1.0):
        i = 0

        while i < self.max_iterations:
            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )
            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_local:
                b_local.set(0.0)

            constants_L = [
                form and dolfinx.cpp.fem.pack_constants(form._cpp_object)
                for form in self.F
            ]
            coeffs_L = [
                dolfinx.cpp.fem.pack_coefficients(form._cpp_object) for form in self.F
            ]

            constants_a = [
                [
                    (
                        dolfinx.cpp.fem.pack_constants(form._cpp_object)
                        if form is not None
                        else np.array([], dtype=PETSc.ScalarType)
                    )
                    for form in forms
                ]
                for forms in self.J
            ]

            coeffs_a = [
                [
                    (
                        {}
                        if form is None
                        else dolfinx.cpp.fem.pack_coefficients(form._cpp_object)
                    )
                    for form in forms
                ]
                for forms in self.J
            ]

            dolfinx.fem.petsc.assemble_vector_block(
                self.b,
                self.F,
                self.J,
                bcs=self.bcs,
                x0=self.x,
                scale=-1.0,
                coeffs_a=coeffs_a,
                constants_a=constants_a,
                coeffs_L=coeffs_L,
                constants_L=constants_L,
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(
                self.A, self.J, bcs=self.bcs, constants=constants_a, coeffs=coeffs_a
            )
            self.A.assemble()

            self._solver.solve(self.b, self.dx)
            # self._solver.view()
            assert (
                self._solver.getConvergedReason() > 0
            ), "Linear solver did not converge"
            offset_start = 0
            for s in self.w:
                num_sub_dofs = (
                    s.function_space.dofmap.index_map.size_local
                    * s.function_space.dofmap.index_map_bs
                )
                s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                    beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
                )
                s.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                offset_start += num_sub_dofs
            # Compute norm of update

            correction_norm = self.dx.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1

    def __del__(self):
        self.A.destroy()
        self.b.destroy()
        self.dx.destroy()
        self._solver.destroy()
        self.x.destroy()


def transfer_meshtags_to_submesh(
    mesh, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent
):
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.
    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


# ---------------- Generate a mesh ----------------
def generate_mesh():
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    def left_domain(x):
        return x[0] <= 0.5 + 1e-14
    
    def right_domain(x):
        return x[0] >= 0.7 - 1e-14

    interface_1 = 0.5
    interface_2 = 0.7

    # with N = 2000 I've never had the error but with N = 10 I had it
    N = 3
    vertices = np.concatenate(
        [
            np.linspace(0, interface_1, num=N),
            np.linspace(interface_1, interface_2, num=N),
            np.linspace(interface_2, 1, num=N),
        ]
    )

    vertices = np.sort(np.unique(vertices)).astype(float)
    degree = 1
    domain = ufl.Mesh(
        basix.ufl.element(basix.ElementFamily.P, "interval", degree, shape=(1,))
    )
    mesh_points = np.reshape(vertices, (len(vertices), 1))
    indexes = np.arange(vertices.shape[0])
    cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)

    # Split domain in half and set an interface tag of 5
    tdim = mesh.topology.dim
    fdim = tdim - 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left_boundary)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right_boundary)
    num_facets_local = (
        mesh.topology.index_map(fdim).size_local
        + mesh.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[left_facets] = 1
    values[right_facets] = 2

    left_cells = dolfinx.mesh.locate_entities(mesh, tdim, left_domain)
    right_cells = dolfinx.mesh.locate_entities(mesh, tdim, right_domain)
    num_cells_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    cells = np.full(num_cells_local, 4, dtype=np.int32)
    cells[left_cells] = 3
    cells[right_cells] = 5
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
    )
    all_l_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(3), tdim, fdim
    )
    all_m_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(4), tdim, fdim
    )
    all_r_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(5), tdim, fdim
    )
    interface1 = np.intersect1d(all_l_facets, all_m_facets)
    interface2 = np.intersect1d(all_m_facets, all_r_facets)
    values[interface1] = 5
    values[interface2] = 6

    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)
    return mesh, mt, ct


class VolumeSubdomain:
    id: int
    submesh: dolfinx.mesh.Mesh
    submesh_to_mesh: np.ndarray
    parent_mesh: dolfinx.mesh.Mesh
    parent_to_submesh: np.ndarray
    v_map: np.ndarray
    facet_to_parent: np.ndarray
    ft: dolfinx.mesh.MeshTags
    padded:bool
    def __init__(self, id):
        self.id = id

    def create_subdomain(self, mesh, marker):
        assert marker.dim == mesh.topology.dim
        self.parent_mesh = mesh
        self.submesh, self.submesh_to_mesh, self.v_map = (
            dolfinx.mesh.create_submesh(mesh, marker.dim, marker.find(self.id))[0:3]
        )
        num_cells_local = (
            mesh.topology.index_map(marker.dim).size_local
            + mesh.topology.index_map(marker.dim).num_ghosts
        )
        self.parent_to_submesh = np.full(num_cells_local, -1, dtype=np.int32)
        self.parent_to_submesh[self.submesh_to_mesh] = np.arange(
            len(self.submesh_to_mesh), dtype=np.int32
        )
        self.padded=False

    def transfer_meshtag(self, tag):
        # Transfer meshtags to submesh
        assert self.submesh is not None, "Need to call create_subdomain first"
        self.ft, self.facet_to_parent = transfer_meshtags_to_submesh(
            mesh, tag, self.submesh, self.v_map, self.submesh_to_mesh
        )

class Interface():
    id: int
    subdomains: tuple[VolumeSubdomain, VolumeSubdomain]
    parent_mesh: dolfinx.mesh.Mesh
    restriction: [str, str] = ("+", "-")
    padded: bool
    def __init__(self, parent_mesh, mt,  id, subdomains):
        self.id = id
        self.subdomains = tuple(subdomains)
        self.mt = mt
        self.parent_mesh = parent_mesh
    def pad_parent_maps(self):
        """Workaround to make sparsity-pattern work without skips
        """ 

        integration_data = compute_integration_domains(
                dolfinx.fem.IntegralType.interior_facet, self.parent_mesh.topology, self.mt.find(self.id), self.mt.dim).reshape(-1, 4)
        for i in range(2):
            # We pad the parent to submesh map to make sure that sparsity pattern is correct
            mapped_cell_0 = self.subdomains[i].parent_to_submesh[integration_data[:, 0]]
            mapped_cell_1 = self.subdomains[i].parent_to_submesh[integration_data[:, 2]]
            max_cells = np.maximum(mapped_cell_0, mapped_cell_1)
            self.subdomains[i].parent_to_submesh[integration_data[:, 0]] = max_cells
            self.subdomains[i].parent_to_submesh[integration_data[:, 2]] = max_cells
            self.subdomains[i].padded = True

mesh, mt, ct = generate_mesh()

left_domain = VolumeSubdomain(3)
mid_domain = VolumeSubdomain(4)
right_domain = VolumeSubdomain(5)
list_of_subdomains = [left_domain, mid_domain, right_domain]


gdim = mesh.geometry.dim
tdim = mesh.topology.dim
fdim = tdim - 1
num_cells_local = (
    mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
)



for subdomain in list_of_subdomains:
    subdomain.create_subdomain(mesh, ct)
    subdomain.transfer_meshtag(mt)





i0 = Interface(mesh, mt, 5, (left_domain, mid_domain))
i1 = Interface(mesh, mt, 6, (mid_domain, right_domain))
interfaces = [i0, i1]

def define_interior_eq(mesh, degree, submesh, submesh_to_mesh, value):
    element_CG = basix.ufl.element(
        basix.ElementFamily.P,
        submesh.basix_cell(),
        degree,
        basix.LagrangeVariant.equispaced,
    )
    element = basix.ufl.mixed_element([element_CG, element_CG])
    V = dolfinx.fem.functionspace(submesh, element)
    u = dolfinx.fem.Function(V, name=f"u_{value}")
    us = list(ufl.split(u))
    vs = list(ufl.TestFunctions(V))
    ct_r = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        submesh_to_mesh,
        np.full_like(submesh_to_mesh, 1, dtype=np.int32),
    )
    val = dolfinx.fem.Constant(submesh, value)
    dx_r = ufl.Measure("dx", domain=mesh, subdomain_data=ct_r, subdomain_id=1)
    F = ufl.inner(ufl.grad(us[0]), ufl.grad(vs[0])) * dx_r - val * vs[0] * dx_r
    k = 2
    p = 0.1
    n = 0.5
    F += k * us[0] * (n - us[1]) * vs[1] * dx_r - p * us[1] * vs[1] * dx_r
    return u, vs, F


# for each subdomain, define the interior equation
for subdomain in list_of_subdomains:
    degree = 1
    subdomain.u, subdomain.vs, subdomain.F = define_interior_eq(
        mesh, degree, subdomain.submesh, subdomain.submesh_to_mesh, 0.0
    )
    subdomain.u.name = f"u_{subdomain.id}"


def compute_mapped_interior_facet_data(interface: Interface):
    """
    Compute integration data for interface integrals.
    We define the first domain on an interface as the "+" restriction,
    meaning that we must sort all integration entities in this order
    
    Parameters
        interface: Interface between two subdomains
    Returns
        integration_data: Integration data for interior facets
    """
    assert (not interface.subdomains[0].padded) and (not interface.subdomains[1].padded)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    integration_data = compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet, mesh.topology, interface.mt.find(interface.id), interface.mt.dim)

    ordered_integration_data = integration_data.reshape(-1, 4).copy()

    mapped_cell_0 = interface.subdomains[0].parent_to_submesh[integration_data[0::4]]
    mapped_cell_1 = interface.subdomains[0].parent_to_submesh[integration_data[2::4]]

    switch = mapped_cell_1 > mapped_cell_0
    # Order restriction on one side        
    if True in switch:
        ordered_integration_data[switch, [0, 1, 2, 3]] = ordered_integration_data[
            switch, [2, 3, 0, 1]
        ]
    
    # Check that other restriction lies in other interface
    domain1_cell = interface.subdomains[1].parent_to_submesh[ordered_integration_data[:, 2]]
    assert (domain1_cell >=0).all()

    return (interface.id, ordered_integration_data.reshape(-1))



integral_data = [compute_mapped_interior_facet_data(interface) for interface in interfaces]
[interface.pad_parent_maps() for interface in interfaces]
dInterface = ufl.Measure(
        "dS", domain=mesh, subdomain_data=integral_data
    )

def mixed_term(u, v, n):
    return ufl.dot(ufl.grad(u), n) * v

n = ufl.FacetNormal(mesh)
cr = ufl.Circumradius(mesh)

gamma = 10.0


entity_maps = {sd.submesh: sd.parent_to_submesh for sd in list_of_subdomains}
for interface in interfaces:
    subdomain_1, subdomain_2 = interface.subdomains
    b_res, t_res = interface.restriction
    n_b = n(b_res)
    n_t = n(t_res)
    h_b = 2 * cr(b_res)
    h_t = 2 * cr(t_res)


    v_b = subdomain_1.vs[0](b_res)
    v_t = subdomain_2.vs[0](t_res)

    u_bs = list(ufl.split(subdomain_1.u))
    u_ts = list(ufl.split(subdomain_2.u))
    u_b = u_bs[0](b_res)
    u_t = u_ts[0](t_res)
    # fabricate K
    W_0 = dolfinx.fem.functionspace(subdomain_1.submesh, ("DG", 0))
    K_0 = dolfinx.fem.Function(W_0, name=f"K_{subdomain_1.id}")
    K_0.x.array[:] = 2
    W_1 = dolfinx.fem.functionspace(subdomain_2.submesh, ("DG", 0))
    K_1 = dolfinx.fem.Function(W_1, name=f"K_{subdomain_2.id}")
    K_1.x.array[:] = 4

    K_b = K_0(b_res)
    K_t = K_1(t_res)

    F_0 = (
        -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface(interface.id)
        - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_b) * dInterface(interface.id)
    )

    F_1 = (
        +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface(interface.id)
        - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_b) * dInterface(interface.id)
    )
    F_0 += 2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_b * dInterface(interface.id)
    F_1 += -2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_t * dInterface(interface.id)

    subdomain_1.F += F_0
    subdomain_2.F += F_1


J = []
forms = []
for i,subdomain1 in enumerate(list_of_subdomains):
    jac = []
    form = subdomain1.F
    for j, subdomain2 in enumerate(list_of_subdomains):
        jac.append(
            dolfinx.fem.form(
                ufl.derivative(form, subdomain2.u), entity_maps=entity_maps
            )
        )

    J.append(jac)
    forms.append(dolfinx.fem.form(subdomain1.F, entity_maps=entity_maps))

# boundary conditions
b_bc = dolfinx.fem.Function(left_domain.u.function_space)
b_bc.x.array[:] = 0.2
left_domain.submesh.topology.create_connectivity(
    left_domain.submesh.topology.dim - 1, left_domain.submesh.topology.dim
)
bc_b = dolfinx.fem.dirichletbc(
    b_bc,
    dolfinx.fem.locate_dofs_topological(
        left_domain.u.function_space.sub(0), fdim, left_domain.ft.find(1)
    ),
)

t_bc = dolfinx.fem.Function(right_domain.u.function_space)
t_bc.x.array[:] = 0.05
right_domain.submesh.topology.create_connectivity(
    right_domain.submesh.topology.dim - 1, right_domain.submesh.topology.dim
)
bc_t = dolfinx.fem.dirichletbc(
    t_bc,
    dolfinx.fem.locate_dofs_topological(
        right_domain.u.function_space.sub(0), fdim, right_domain.ft.find(2)
    ),
)
bcs = [bc_b, bc_t]


solver = NewtonSolver(
    forms,
    J,
    [subdomain.u for subdomain in list_of_subdomains],
    bcs=bcs,
    max_iterations=10,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
solver.solve(1e-5)

# for subdomain in list_of_subdomains:
#     u_sub_0 = subdomain.u.sub(0).collapse()
#     u_sub_0.name = "u_sub_0"

#     u_sub_1 = subdomain.u.sub(1).collapse()
#     u_sub_1.name = "u_sub_1"
#     bp = dolfinx.io.VTXWriter(
#         mesh.comm, f"u_{subdomain.id}.bp", [u_sub_0, u_sub_1], engine="BP4"
#     )
#     bp.write(0)
#     bp.close()


# # derived quantities
# V = dolfinx.fem.functionspace(mesh, ("CG", 1))
# T = dolfinx.fem.Function(V)
# T.interpolate(lambda x: 200 + x[1])


# T_b = dolfinx.fem.Function(right_domain.u.sub(0).collapse().function_space)
# T_b.interpolate(T)

# ds_b = ufl.Measure("ds", domain=right_domain.submesh)
# dx_b = ufl.Measure("dx", domain=right_domain.submesh)
# dx = ufl.Measure("dx", domain=mesh)

# n_b = ufl.FacetNormal(left_domain.submesh)

# form = dolfinx.fem.form(left_domain.u.sub(0) * dx_b, entity_maps=entity_maps)
# print(dolfinx.fem.assemble_scalar(form))

# form = dolfinx.fem.form(T_b * ufl.dot(ufl.grad(left_domain.u.sub(0)), n_b) * ds_b, entity_maps=entity_maps)
# print(dolfinx.fem.assemble_scalar(form))
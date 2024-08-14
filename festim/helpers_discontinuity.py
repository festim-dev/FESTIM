from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix


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
            dolfinx.fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
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

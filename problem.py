import festim as F
import dolfinx
from dolfinx import fem
import numpy as np
import basix
import ufl
from festim.helpers_discontinuity import NewtonSolver, transfer_meshtags_to_submesh


def K_S_fun(T, K_S_0, E_K_S):
    k_B = 8.6173303e-5
    return K_S_0 * ufl.exp(-E_K_S / k_B / T)


class HTransportProblemDiscontinuous(F.HydrogenTransportProblem):

    def initialise(self):
        self.create_submeshes()
        self.create_species_from_traps()
        self.define_temperature()

        self.entity_maps = {
            subdomain.submesh: subdomain.parent_to_submesh
            for subdomain in self.volume_subdomains
        }

        for subdomain in self.volume_subdomains:
            self.define_function_spaces(subdomain)
            ct_r = dolfinx.mesh.meshtags(
                self.mesh.mesh,
                self.mesh.mesh.topology.dim,
                subdomain.submesh_to_mesh,
                np.full_like(subdomain.submesh_to_mesh, 1, dtype=np.int32),
            )
            subdomain.dx = ufl.Measure(
                "dx", domain=self.mesh.mesh, subdomain_data=ct_r, subdomain_id=1
            )
            self.create_subdomain_formulation(subdomain)
            subdomain.u.name = f"u_{subdomain.id}"

        self.define_meshtags_and_measures()
        # self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = F.as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.define_boundary_conditions()
        self.create_source_values_fenics()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def create_dirichletbc_form(self, bc):
        fdim = self.mesh.mesh.topology.dim - 1
        volume_subdomain = self.surface_to_volume[bc.subdomain]
        bc_function = dolfinx.fem.Function(volume_subdomain.u.function_space)
        bc_function.x.array[:] = bc.value
        volume_subdomain.submesh.topology.create_connectivity(
            volume_subdomain.submesh.topology.dim - 1,
            volume_subdomain.submesh.topology.dim,
        )
        form = dolfinx.fem.dirichletbc(
            bc_function,
            dolfinx.fem.locate_dofs_topological(
                volume_subdomain.u.function_space.sub(0),
                fdim,
                volume_subdomain.ft.find(bc.subdomain.id),
            ),
        )
        return form

    def create_submeshes(self):
        mesh = self.mesh.mesh
        ct = self.volume_meshtags
        mt = self.facet_meshtags

        gdim = mesh.geometry.dim
        tdim = mesh.topology.dim
        fdim = tdim - 1

        num_facets_local = (
            mesh.topology.index_map(fdim).size_local
            + mesh.topology.index_map(fdim).num_ghosts
        )

        for subdomain in self.volume_subdomains:
            subdomain.submesh, subdomain.submesh_to_mesh, subdomain.v_map = (
                dolfinx.mesh.create_submesh(mesh, tdim, ct.find(subdomain.id))[0:3]
            )

            subdomain.parent_to_submesh = np.full(num_facets_local, -1, dtype=np.int32)
            subdomain.parent_to_submesh[subdomain.submesh_to_mesh] = np.arange(
                len(subdomain.submesh_to_mesh), dtype=np.int32
            )

            # We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
            # We use the entity on the same side to fix this (as all restrictions are one-sided)

            # Transfer meshtags to submesh
            subdomain.ft, subdomain.facet_to_parent = transfer_meshtags_to_submesh(
                mesh, mt, subdomain.submesh, subdomain.v_map, subdomain.submesh_to_mesh
            )

        # Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
        # TODO ask Jorgen what this is for
        mesh.topology.create_connectivity(fdim, tdim)
        f_to_c = mesh.topology.connectivity(fdim, tdim)
        for interface in self.interfaces:
            for facet in mt.find(interface):
                cells = f_to_c.links(facet)
                assert len(cells) == 2
                for domain in self.interfaces[interface]:
                    map = domain.parent_to_submesh[cells]
                    domain.parent_to_submesh[cells] = max(map)

        self.f_to_c = f_to_c

    def define_function_spaces(self, subdomain: F.VolumeSubdomain):
        # get number of species defined in the subdomain
        all_species = [
            species for species in self.species if subdomain in species.subdomains
        ]

        # instead of using the set function we use a list to keep the order
        unique_species = []
        for species in all_species:
            if species not in unique_species:
                unique_species.append(species)
        nb_species = len(unique_species)

        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            subdomain.submesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )
        element = basix.ufl.mixed_element([element_CG] * nb_species)
        V = dolfinx.fem.functionspace(subdomain.submesh, element)
        u = dolfinx.fem.Function(V)
        u_n = dolfinx.fem.Function(V)

        us = list(ufl.split(u))
        u_ns = list(ufl.split(u_n))
        vs = list(ufl.TestFunctions(V))
        for i, species in enumerate(unique_species):
            species.subdomain_to_solution[subdomain] = us[i]
            species.subdomain_to_prev_solution[subdomain] = u_ns[i]
            species.subdomain_to_test_function[subdomain] = vs[i]
        subdomain.u = u

    def create_subdomain_formulation(self, subdomain: F.VolumeSubdomain):
        form = 0
        # add diffusion and time derivative for each species
        for spe in self.species:
            u = spe.subdomain_to_solution[subdomain]
            u_n = spe.subdomain_to_prev_solution[subdomain]
            v = spe.subdomain_to_test_function[subdomain]
            dx = subdomain.dx

            D = subdomain.material.get_diffusion_coefficient(
                self.mesh.mesh, self.temperature_fenics, spe
            )
            if self.settings.transient:
                raise NotImplementedError("Transient not implemented")

            if spe.mobile:
                form += ufl.inner(D * ufl.grad(u), ufl.grad(v)) * dx

        for reaction in self.reactions:
            if reaction.volume != subdomain:
                continue
            for species in reaction.reactant + reaction.product:
                if isinstance(species, F.Species):
                    # TODO remove
                    # temporarily overide the solution and test function to the one of the subdomain
                    species.solution = species.subdomain_to_solution[subdomain]
                    species.test_function = species.subdomain_to_test_function[
                        subdomain
                    ]

            for reactant in reaction.reactant:
                if isinstance(reactant, F.Species):
                    form += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.subdomain_to_test_function[subdomain]
                        * dx
                    )

            # product
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]
            for product in products:
                form += (
                    -reaction.reaction_term(self.temperature_fenics)
                    * product.subdomain_to_test_function[subdomain]
                    * dx
                )

        subdomain.F = form

    def create_formulation(self):
        # Add coupling term to the interface
        # Get interface markers on submesh b
        mesh = self.mesh.mesh
        ct = self.volume_meshtags
        mt = self.facet_meshtags
        f_to_c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        for interface in self.interfaces:

            dInterface = ufl.Measure(
                "dS", domain=mesh, subdomain_data=mt, subdomain_id=interface
            )
            b_res = "+"
            t_res = "-"

            # look at the first facet on interface
            # and get the two cells that are connected to it
            # and get the material properties of these cells
            first_facet_interface = mt.find(interface)[0]
            c_plus, c_minus = (
                f_to_c.links(first_facet_interface)[0],
                f_to_c.links(first_facet_interface)[1],
            )
            id_minus, id_plus = ct.values[c_minus], ct.values[c_plus]

            for subdomain in self.interfaces[interface]:
                if subdomain.id == id_plus:
                    subdomain_1 = subdomain
                if subdomain.id == id_minus:
                    subdomain_2 = subdomain

            all_mobile_species = [spe for spe in self.species if spe.mobile]
            if len(all_mobile_species) > 1:
                raise NotImplementedError("Multiple mobile species not implemented")
            H = all_mobile_species[0]

            v_b = H.subdomain_to_test_function[subdomain_1](b_res)
            v_t = H.subdomain_to_test_function[subdomain_2](t_res)

            u_b = H.subdomain_to_solution[subdomain_1](b_res)
            u_t = H.subdomain_to_solution[subdomain_2](t_res)

            def mixed_term(u, v, n):
                return ufl.dot(ufl.grad(u), n) * v

            n = ufl.FacetNormal(mesh)
            n_b = n(b_res)
            n_t = n(t_res)
            cr = ufl.Circumradius(mesh)
            h_b = 2 * cr(b_res)
            h_t = 2 * cr(t_res)
            gamma = 400.0  # this needs to be "sufficiently large"

            K_b = K_S_fun(
                self.temperature_fenics(b_res),
                subdomain_1.material.K_S_0,
                subdomain_1.material.E_K_S,
            )
            K_t = K_S_fun(
                self.temperature_fenics(t_res),
                subdomain_2.material.K_S_0,
                subdomain_2.material.E_K_S,
            )

            F_0 = (
                -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface
                - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_b) * dInterface
            )

            F_1 = (
                +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface
                - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_b) * dInterface
            )
            F_0 += 2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_b * dInterface
            F_1 += -2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_t * dInterface

            subdomain_1.F += F_0
            subdomain_2.F += F_1

        J = []
        forms = []
        for subdomain1 in self.volume_subdomains:
            jac = []
            form = subdomain1.F
            for subdomain2 in self.volume_subdomains:
                jac.append(
                    dolfinx.fem.form(
                        ufl.derivative(form, subdomain2.u), entity_maps=self.entity_maps
                    )
                )
            J.append(jac)
            forms.append(dolfinx.fem.form(subdomain1.F, entity_maps=self.entity_maps))
        self.forms = forms
        self.J = J

    def create_solver(self):
        self.solver = NewtonSolver(
            self.forms,
            self.J,
            [subdomain.u for subdomain in self.volume_subdomains],
            bcs=self.bc_forms,
            max_iterations=10,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

    def run(self):
        if self.settings.transient:
            raise NotImplementedError("Transient not implemented")
        else:
            # Solve steady-state
            self.solver.solve(1e-5)
            self.post_processing()

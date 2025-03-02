from festim import (
    Concentration,
    FluxBC,
    k_B,
    RadioactiveDecay,
    SurfaceKinetics,
    festim_print,
)
from fenics import *


class Mobile(Concentration):
    """
    The mobile concentration.

    If conservation of chemical potential, this will be c_m/S.
    If not, Mobile represents c_m.

    Attributes:
        sources (list): list of festim.Source objects.
            The volumetric source terms
        F (fenics.Form): the variational formulation for mobile
    """

    def __init__(self):
        """Inits festim.Mobile"""
        super().__init__()
        self.sources = []
        self.boundary_conditions = []

    def create_form(self, materials, mesh, T, dt=None, traps=None, soret=False):
        """Creates the variational formulation.

        Args:
            materials (festim.Materials): the materials
            mesh (festim.Mesh): the mesh of the simulation
            T (festim.Temperature): the temperature
            dt (festim.Stepsize, optional): the stepsize. Defaults to None.
            traps (festim.Traps, optional): the traps. Defaults to None.
            chemical_pot (bool, optional): if True, conservation of chemical
                potential is assumed. Defaults to False.
            soret (bool, optional): If True, Soret effect is assumed. Defaults
                to False.
        """
        self.F = 0
        self.create_diffusion_form(materials, mesh, T, dt=dt, traps=traps, soret=soret)
        self.create_source_form(mesh.dx)
        self.create_fluxes_form(T, mesh.ds, dt)

    def create_diffusion_form(
        self, materials, mesh, T, dt=None, traps=None, soret=False
    ):
        """Creates the variational formulation for the diffusive part.

        Args:
            materials (festim.Materials): the materials
            mesh (festim.Mesh): the mesh
            T (festim.Temperature): the temperature
            dt (festim.Stepsize, optional): the stepsize. Defaults to None.
            traps (festim.Traps, optional): the traps. Defaults to None.
            chemical_pot (bool, optional): if True, conservation of chemical
                potential is assumed. Defaults to False.
            soret (bool, optional): If True, Soret effect is assumed. Defaults
                to False.
        """

        F = 0
        for material in materials:
            D_0 = material.D_0
            E_D = material.E_D
            c_0, c_0_n = self.get_concentration_for_a_given_material(material, T)

            subdomains = material.id  # list of subdomains with this material
            if not isinstance(subdomains, list):
                subdomains = [subdomains]  # make sure subdomains is a list

            # add to the formulation F for every subdomain
            for subdomain in subdomains:
                dx = mesh.dx(subdomain)
                # transient form
                if dt is not None:
                    F += ((c_0 - c_0_n) / dt.value) * self.test_function * dx
                D = D_0 * exp(-E_D / k_B / T.T)
                if mesh.type == "cartesian":
                    F += dot(D * grad(c_0), grad(self.test_function)) * dx
                    if soret:
                        Q = material.Q
                        if callable(Q):
                            Q = Q(T.T)
                        F += (
                            dot(
                                D * Q * c_0 / (k_B * T.T**2) * grad(T.T),
                                grad(self.test_function),
                            )
                            * dx
                        )

                # see https://fenicsproject.discourse.group/t/method-of-manufactured-solution-cylindrical/7963
                elif mesh.type == "cylindrical":
                    r = SpatialCoordinate(mesh.mesh)[0]
                    F += r * dot(D * grad(c_0), grad(self.test_function / r)) * dx

                    if soret:
                        Q = material.Q
                        if callable(Q):
                            Q = Q(T.T)
                        F += (
                            r
                            * dot(
                                D * Q * c_0 / (k_B * T.T**2) * grad(T.T),
                                grad(self.test_function / r),
                            )
                            * dx
                        )

                elif mesh.type == "spherical":
                    r = SpatialCoordinate(mesh.mesh)[0]
                    F += (
                        D
                        * r
                        * r
                        * dot(grad(c_0), grad(self.test_function / r / r))
                        * dx
                    )

                    if soret:
                        Q = material.Q
                        if callable(Q):
                            Q = Q(T.T)
                        F += (
                            D
                            * r
                            * r
                            * dot(
                                Q * c_0 / (k_B * T.T**2) * grad(T.T),
                                grad(self.test_function / r / r),
                            )
                            * dx
                        )

        # add the trapping terms
        F_trapping = 0
        if traps is not None:
            for trap in traps:
                for i, mat in enumerate(trap.materials):
                    if type(trap.k_0) is list:
                        k_0 = trap.k_0[i]
                        E_k = trap.E_k[i]
                        p_0 = trap.p_0[i]
                        E_p = trap.E_p[i]
                        density = trap.density[i]
                    else:
                        k_0 = trap.k_0
                        E_k = trap.E_k
                        p_0 = trap.p_0
                        E_p = trap.E_p
                        density = trap.density[0]
                    c_m, _ = self.get_concentration_for_a_given_material(mat, T)
                    F_trapping += (
                        -k_0
                        * exp(-E_k / k_B / T.T)
                        * c_m
                        * (density - trap.solution)
                        * self.test_function
                        * dx(mat.id)
                    )
                    F_trapping += (
                        p_0
                        * exp(-E_p / k_B / T.T)
                        * trap.solution
                        * self.test_function
                        * dx(mat.id)
                    )
        F += -F_trapping

        self.F_diffusion = F
        self.F += F

    def create_source_form(self, dx):
        """Creates the variational form for the volumetric source term parts.

        Args:
            dx (fenics.Measure): the measure dx
        """
        F_source = 0
        expressions_source = []

        festim_print("Defining source terms")
        for source in self.sources:
            if type(source.volume) is list:
                volumes = source.volume
            else:
                volumes = [source.volume]
            if isinstance(source, RadioactiveDecay):
                source.value = source.form(self.mobile_concentration())

            for volume in volumes:
                F_source += -source.value * self.test_function * dx(volume)
            if isinstance(source.value, (Expression, UserExpression)):
                expressions_source.append(source.value)

        self.F_source = F_source
        self.F += F_source
        self.sub_expressions += expressions_source

    def create_fluxes_form(self, T, ds, dt=None):
        """Modifies the formulation and adds fluxes based
        on parameters in self.boundary_conditions
        """

        expressions_fluxes = []
        F = 0

        solute = self.mobile_concentration()

        for bc in self.boundary_conditions:
            if bc.field != "T":
                if isinstance(bc, FluxBC):
                    if isinstance(bc, SurfaceKinetics):
                        bc.create_form(
                            solute,
                            self.previous_solution,
                            self.test_function,
                            T,
                            ds,
                            dt,
                        )
                        F += bc.form
                    else:
                        bc.create_form(T.T, solute)
                        for surf in bc.surfaces:
                            F += -self.test_function * bc.form * ds(surf)
                    # TODO : one day we will get rid of this huge expressions list
                    expressions_fluxes += bc.sub_expressions

        self.F_fluxes = F
        self.F += F
        self.sub_expressions += expressions_fluxes

    def get_concentration_for_a_given_material(self, material, T):
        return self.solution, self.previous_solution

    def mobile_concentration(self):
        return self.solution

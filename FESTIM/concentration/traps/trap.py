from FESTIM import Concentration, k_B
from fenics import *
import sympy as sp
import numpy as np


class Trap(Concentration):
    def __init__(
            self, k_0, E_k, p_0, E_p, materials, density, id=None):
        """Inits Trap

        Args:
            k_0 (float): trapping pre-exponential factor
            E_k (float): trapping activation energy
            p_0 (float): detrapping pre-exponential factor
            E_p (float): detrapping activation energy
            materials (list or int): the materials ids the trap is living in
            density (sp.Add, float): the trap density
            id (int, optional): The trap id. Defaults to None.

        Raises:
            ValueError: if duplicates are found in materials
        """
        super().__init__()
        self.id = id
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        self.materials = materials
        if not isinstance(self.materials, list):
            self.materials = [self.materials]
        if len(self.materials) != len(np.unique(self.materials)):
            raise ValueError("Duplicate materials in trap")

        self.density = []
        self.make_density(density)
        self.sources = []

    def make_density(self, densities):
        if type(densities) is not list:
            densities = [densities]

        for i, density in enumerate(densities):
            if density is not None:
                density_expr = sp.printing.ccode(density)
                self.density.append(Expression(density_expr, degree=2, t=0, name="density_{}_{}".format(self.id, i)))

    def create_form(
            self, mobile, materials, T, dx, dt=None,
            chemical_pot=False):
        """Creates the general form associated with the trap
        d ct/ dt = k c_m (n - c_t) - p c_t + S

        Args:
            mobile (FESTIM.Mobile): the mobile concentration of the simulation
            materials (FESTIM.Materials): the materials of the simulation
            T (FESTIM.Temperature): the temperature of the simulation
            dx (fenics.Measure): the dx measure of the sim
            dt (FESTIM.Stepsize, optional): If None assuming steady state.
                Defaults to None.
            chemical_pot (bool, optional): If True, continuity of chemical
                potential is assumed. Defaults to False.
        """
        self.F = 0
        self.create_trapping_form(mobile, materials, T, dx, dt, chemical_pot)
        if self.sources is not None:
            self.create_source_form(dx)

    def create_trapping_form(self, mobile, materials, T, dx, dt=None, chemical_pot=False):
        """d ct/ dt = k c_m (n - c_t) - p c_t

        Args:
            mobile (FESTIM.Mobile): the mobile concentration of the simulation
            materials (FESTIM.Materials): the materials of the simulation
            T (FESTIM.Temperature): the temperature of the simulation
            dx (fenics.Measure): the dx measure of the sim
            dt (FESTIM.Stepsize, optional): If None assuming steady state.
                Defaults to None.
            chemical_pot (bool, optional): If True, continuity of chemical
                potential is assumed. Defaults to False.
        """
        solution = self.solution
        prev_solution = self.previous_solution
        test_function = self.test_function
        trap_materials = self.materials

        T = T.T
        c_0 = mobile.solution
        if chemical_pot:
            theta = c_0

        expressions_trap = []
        F_trapping = 0  # initialise the form

        if dt is not None:
            # d(c_t)/dt in trapping equation
            F_trapping += ((solution - prev_solution) / dt.value) * test_function * dx
        else:
            # if the sim is steady state and
            # if a trap is not defined in one subdomain
            # add c_t = 0 to the form in this subdomain
            all_mat_ids = [mat.id for mat in materials.materials]
            for mat_id in all_mat_ids:
                if mat_id not in trap_materials:
                    F_trapping += solution*test_function*dx(mat_id)

        for i, mat_id in enumerate(trap_materials):
            if type(self.k_0) is list:
                k_0 = self.k_0[i]
                E_k = self.E_k[i]
                p_0 = self.p_0[i]
                E_p = self.E_p[i]
                density = self.density[i]
            else:
                k_0 = self.k_0
                E_k = self.E_k
                p_0 = self.p_0
                E_p = self.E_p
                density = self.density[0]

            # add the density to the list of
            # expressions to be updated
            expressions_trap.append(density)

            corresponding_material = \
                materials.find_material_from_id(mat_id)
            if chemical_pot:
                # TODO this needs changing for Henry
                # change of variable
                S_0 = corresponding_material.S_0
                E_S = corresponding_material.E_S
                c_0 = theta*S_0*exp(-E_S/k_B/T)

            # k(T)*c_m*(n - c_t) - p(T)*c_t
            F_trapping += - k_0 * exp(-E_k/k_B/T) * c_0 \
                * (density - solution) * \
                test_function*dx(mat_id)
            F_trapping += p_0*exp(-E_p/k_B/T)*solution * \
                test_function*dx(mat_id)

        self.F_trapping = F_trapping
        self.F += self.F_trapping
        self.sub_expressions += expressions_trap

    def create_source_form(self, dx):
        """Create the source form for the trap

        Args:
            dx (fenics.Measure): the dx measure of the sim
        """
        for source in self.sources:
            self.F_source = -source.value*self.test_function*dx(source.volume)
            self.F += self.F_source
            self.sub_expressions.append(source.value)

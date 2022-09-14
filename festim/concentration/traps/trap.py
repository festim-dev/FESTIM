from festim import Concentration, k_B, Material, Theta
from fenics import *
import sympy as sp
import numpy as np


class Trap(Concentration):
    """
    Args:
        k_0 (float, list): trapping pre-exponential factor (m3 s-1)
        E_k (float, list): trapping activation energy (eV)
        p_0 (float, list): detrapping pre-exponential factor (s-1)
        E_p (float, list): detrapping activation energy (eV)
        materials (list, str, festim.Material): the materials the
            trap is living in. The material's name.
        density (sp.Add, float, list, fenics.Expresion, fenics.UserExpression):
            the trap density (m-3)
        id (int, optional): The trap id. Defaults to None.

    Raises:
        ValueError: if duplicates are found in materials

    Notes:
        Should multiple traps in muliple materials be used, to save on
        dof's, traps can be conglomerated and described in lists in the
        format::

            festim.Trap(
                k_0=[1, 2],
                E_k=[1, 2],
                p_0=[1, 2],
                E_p=[1, 2],
                materials=[1, 2]
                density=[1, 2])

        This will act as a singular trap but with seperate properties for
        respective materials. Parameters k_0, E_k, p_0, E_p, materials and
        density MUST have the same length for this method to be valid.
    """

    def __init__(self, k_0, E_k, p_0, E_p, materials, density, id=None):

        super().__init__()
        self.id = id
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        self.materials = materials

        self.density = []
        self.make_density(density)
        self.sources = []

    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, value):
        if not isinstance(value, list):
            value = [value]
        for entry in value:
            if not isinstance(entry, (str, Material)):
                raise TypeError(
                    "Accepted types for materials are str or festim.Material"
                )
        self._materials = value

    def make_materials(self, materials):
        """Ensure all entries in self.materials are of type festim.Material

        Args:
            materials (festim.Materials): the materials

        Raises:
            ValueError: if some duplicates are found
        """
        new_materials = []

        for material in self.materials:
            new_materials.append(materials.find_material(material))

        self.materials = new_materials

        if len(self.materials) != len(list(set(self.materials))):
            raise ValueError("Duplicate materials in trap")

    def make_density(self, densities):
        if type(densities) is not list:
            densities = [densities]

        for i, density in enumerate(densities):
            if density is not None:
                # if density is already a fenics Expression, use it as is
                if isinstance(density, (Expression, UserExpression)):
                    self.density.append(density)
                # else assume it's a sympy expression
                else:
                    density_expr = sp.printing.ccode(density)
                    self.density.append(
                        Expression(
                            density_expr,
                            degree=2,
                            t=0,
                            name="density_{}_{}".format(self.id, i),
                        )
                    )

    def create_form(self, mobile, materials, T, dx, dt=None):
        """Creates the general form associated with the trap
        d ct/ dt = k c_m (n - c_t) - p c_t + S

        Args:
            mobile (festim.Mobile): the mobile concentration of the simulation
            materials (festim.Materials): the materials of the simulation
            T (festim.Temperature): the temperature of the simulation
            dx (fenics.Measure): the dx measure of the sim
            dt (festim.Stepsize, optional): If None assuming steady state.
                Defaults to None.
        """
        self.F = 0
        self.create_trapping_form(mobile, materials, T, dx, dt)
        if self.sources is not None:
            self.create_source_form(dx)

    def create_trapping_form(self, mobile, materials, T, dx, dt=None):
        """d ct/ dt = k c_m (n - c_t) - p c_t

        Args:
            mobile (festim.Mobile): the mobile concentration of the simulation
            materials (festim.Materials): the materials of the simulation
            T (festim.Temperature): the temperature of the simulation
            dx (fenics.Measure): the dx measure of the sim
            dt (festim.Stepsize, optional): If None assuming steady state.
                Defaults to None.
        """
        solution = self.solution
        prev_solution = self.previous_solution
        test_function = self.test_function

        if not all(isinstance(mat, Material) for mat in self.materials):
            self.make_materials(materials)

        expressions_trap = []
        F_trapping = 0  # initialise the form

        if dt is not None:
            # d(c_t)/dt in trapping equation
            F_trapping += ((solution - prev_solution) / dt.value) * test_function * dx
        else:
            # if the sim is steady state and
            # if a trap is not defined in one subdomain
            # add c_t = 0 to the form in this subdomain
            for mat in materials.materials:
                if mat not in self.materials:
                    F_trapping += solution * test_function * dx(mat.id)

        for i, mat in enumerate(self.materials):
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

            if isinstance(mobile, Theta) and mat.solubility_law == "henry":
                raise NotImplementedError(
                    "Henry law of solubility is not implemented with traps"
                )

            c_0, c_0_n = mobile.get_concentration_for_a_given_material(mat, T)

            # k(T)*c_m*(n - c_t) - p(T)*c_t
            F_trapping += (
                -k_0
                * exp(-E_k / k_B / T.T)
                * c_0
                * (density - solution)
                * test_function
                * dx(mat.id)
            )
            F_trapping += (
                p_0 * exp(-E_p / k_B / T.T) * solution * test_function * dx(mat.id)
            )

        self.F_trapping = F_trapping
        self.F += self.F_trapping
        self.sub_expressions += expressions_trap

    def create_source_form(self, dx):
        """Create the source form for the trap

        Args:
            dx (fenics.Measure): the dx measure of the sim
        """
        for source in self.sources:
            self.F_source = -source.value * self.test_function * dx(source.volume)
            self.F += self.F_source
            self.sub_expressions.append(source.value)

from festim import Mobile, k_B
import fenics as f


class Theta(Mobile):
    """Class representing the "chemical potential" c/S where S is the
    solubility of the metal
    """

    def __init__(self):
        """Inits Theta"""
        super().__init__()
        self.S = None
        self.F = None

    def initialise(self, V, value, label=None, time_step=None):
        """Assign a value to self.previous_solution

        Args:
            V (fenics.FunctionSpace): the function space
            value (sp.Add, float, int, str): the value of the initialisation.
            label (str, optional): the label in the XDMF file. Defaults to
                None.
            time_step (int, optional): the time step to read in the XDMF file.
                Defaults to None.
        """
        comp = self.get_comp(V, value, label=label, time_step=time_step)

        prev_sol = f.Function(V)
        v = f.TestFunction(V)
        dx = f.Measure("dx", subdomain_data=self.volume_markers)
        F = 0
        for mat in self.materials.materials:
            S = mat.S_0 * f.exp(-mat.E_S / k_B / self.T.T)
            F += -prev_sol * v * dx(mat.id)
            if mat.solubility_law == "sievert":
                F += comp / S * v * dx(mat.id)
            elif mat.solubility_law == "henry":
                F += (comp / S) ** 0.5 * v * dx(mat.id)
        f.solve(F == 0, prev_sol, bcs=[])

        f.assign(self.previous_solution, prev_sol)

    def get_concentration_for_a_given_material(self, material, T):
        """Returns the concentration (and previous concentration) for a given
        material

        Args:
            material (festim.Material): the material with attributes S_0 and
                E_S
            T (festim.Temperature): the temperature with attributest T and T_n

        Returns:
            fenics.Product, fenics.Product: the current concentration and
                previous concentration
        """
        E_S = material.E_S
        S_0 = material.S_0
        S = S_0 * f.exp(-E_S / k_B / T.T)
        S_n = S_0 * f.exp(-E_S / k_B / T.T_n)
        if material.solubility_law == "sievert":
            c_0 = self.solution * S
            c_0_n = self.previous_solution * S_n
        elif material.solubility_law == "henry":
            c_0 = (self.solution) ** 2 * S
            c_0_n = self.previous_solution**2 * S_n
        return c_0, c_0_n

    def mobile_concentration(self):
        """Returns the hydrogen concentration as c=theta*K_S or c=theta**2*K_H
        This is needed when adding robin BCs (eg RecombinationFlux).

        Returns:
            ufl.algebra.Sum: the hydrogen mobile concentration
        """
        henry_to_concentration = self.solution**2 * self.S
        sieverts_to_concentration = self.solution * self.S
        # henry_marker is equal to 1 in Henry materials and 0 elsewhere
        return (
            self.materials.henry_marker * henry_to_concentration
            + self.materials.sievert_marker * sieverts_to_concentration
        )

    def post_processing_solution_to_concentration(self):
        """Converts the post_processing_solution from theta to mobile
        concentration.
        c = theta * S.
        The attribute post_processing_solution is fenics.Product (if self.S is
        festim.ArheniusCoeff)
        """
        du = f.TrialFunction(self.post_processing_solution.function_space())
        J = f.derivative(self.form_post_processing, self.post_processing_solution, du)
        problem = f.NonlinearVariationalProblem(
            self.form_post_processing, self.post_processing_solution, [], J
        )
        solver = f.NonlinearVariationalSolver(problem)
        # TODO these prms should be the same as in Simulation.settings I think
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-10
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-10
        solver.parameters["newton_solver"]["maximum_iterations"] = 50
        solver.solve()

    def create_form_post_processing(self, V, materials, dx):
        F = 0
        v = f.TestFunction(V)
        self.post_processing_solution = f.Function(V)
        F += -self.post_processing_solution * v * dx
        for mat in materials.materials:
            if mat.solubility_law == "sievert":
                F += self.solution * self.S * v * dx(mat.id)
            elif mat.solubility_law == "henry":
                F += self.solution**2 * self.S * v * dx(mat.id)
        self.form_post_processing = F

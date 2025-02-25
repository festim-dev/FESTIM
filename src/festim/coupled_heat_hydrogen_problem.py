import tqdm.autonotebook

from festim.heat_transfer_problem import HeatTransferProblem
from festim.helpers import as_fenics_constant
from festim.hydrogen_transport_problem import HydrogenTransportProblem


class CoupledHeatTransferHydrogenTransport:
    """
    Coupled heat transfer and hydrogen transport problem

    Args:
        hydrogen_problem: the hydrogen transport problem
        heat_problem: the heat transfer problem

    Attributes:
        hydrogen_problem: the hydrogen transport problem
        heat_problem: the heat transfer problem

    Examples:
        .. highlight:: python
        .. code-block:: python

            import festim as F

            my_h_transport_model = F.HydrogenTransportProblem(
                mesh=F.Mesh(...),
                subdomains=[F.Subdomain(...)],
                species=[F.Species(name="H"), F.Species(name="Trap")],
                ...
            )

            my_heat_transfer_model = F.HeatTransferProblem(
                mesh=F.Mesh(...),
                subdomains=[F.Subdomain(...)],
                ...
            )

            coupled_problem = F.CoupledHeatTransferHydrogenTransport(
                hydrogen_problem=my_h_transport_model,
                heat_problem=my_heat_transfer_model,
            )


    """

    hydrogen_problem: HydrogenTransportProblem
    heat_problem: HeatTransferProblem

    def __init__(
        self,
        hydrogen_problem: HydrogenTransportProblem,
        heat_problem: HeatTransferProblem,
    ) -> None:
        self.hydrogen_problem = hydrogen_problem
        self.heat_problem = heat_problem

    @property
    def hydrogen_problem(self):
        return self._hydrogen_problem

    @hydrogen_problem.setter
    def hydrogen_problem(self, value):
        if not isinstance(value, HydrogenTransportProblem):
            raise TypeError(
                "hydrogen_problem must be a festim.HydrogenTransportProblem object"
            )
        self._hydrogen_problem = value

    @property
    def heat_problem(self):
        return self._heat_problem

    @heat_problem.setter
    def heat_problem(self, value):
        if not isinstance(value, HeatTransferProblem):
            raise TypeError("heat_problem must be a festim.HeatTransferProblem object")
        self._heat_problem = value

    def initialise(self):
        if (
            self.heat_problem.settings.transient
            and self.hydrogen_problem.settings.transient
        ):
            # make sure both problems have the same initial time step and final time,
            # use minimal initial value of the two and maximal final time of the two
            min_initial_dt = min(
                float(self.heat_problem.settings.stepsize.initial_value),
                float(self.hydrogen_problem.settings.stepsize.initial_value),
            )
            self.heat_problem.settings.stepsize.initial_value = min_initial_dt
            self.hydrogen_problem.settings.stepsize.initial_value = min_initial_dt

            max_final_time = max(
                float(self.heat_problem.settings.final_time),
                float(self.hydrogen_problem.settings.final_time),
            )
            self.heat_problem.settings.final_time = max_final_time
            self.hydrogen_problem.settings.final_time = max_final_time

        self.heat_problem.initialise()

        self.heat_problem.show_progress_bar = False

        if self.heat_problem.mesh.mesh == self.hydrogen_problem.mesh.mesh:
            self.hydrogen_problem.temperature = self.heat_problem.u
        else:
            raise ValueError(
                "The meshes of the heat transfer and hydrogen transport problems must be the same"
            )

        self.hydrogen_problem.initialise()

    def iterate(self):
        self.heat_problem.iterate()
        self.hydrogen_problem.iterate()

        # use the same time step for both problems, use minimum of the two
        next_dt_value = min(
            float(self.hydrogen_problem.dt), float(self.heat_problem.dt)
        )
        self.heat_problem.dt = as_fenics_constant(
            value=next_dt_value, mesh=self.heat_problem.mesh.mesh
        )
        self.hydrogen_problem.dt = as_fenics_constant(
            value=next_dt_value, mesh=self.hydrogen_problem.mesh.mesh
        )

    def run(self):
        if (
            self.heat_problem.settings.transient
            and self.hydrogen_problem.settings.transient
        ):
            if self.hydrogen_problem.show_progress_bar:
                self.hydrogen_problem.progress_bar = tqdm.autonotebook.tqdm(
                    desc=f"Solving {self.__class__.__name__}",
                    total=self.hydrogen_problem.settings.final_time,
                    unit_scale=True,
                )

            while (
                self.hydrogen_problem.t.value
                < self.hydrogen_problem.settings.final_time
            ):
                self.iterate()

            if self.hydrogen_problem.show_progress_bar:
                self.hydrogen_problem.progress_bar.refresh()  # refresh progress bar to show 100%
                self.hydrogen_problem.progress_bar.close()
        else:
            # Solve steady-state
            self.heat_problem.solver.solve(self.heat_problem.u)
            self.heat_problem.post_processing()

            self.hydrogen_problem.temperature = self.heat_problem.u
            self.hydrogen_problem.initialise()
            self.hydrogen_problem.run()

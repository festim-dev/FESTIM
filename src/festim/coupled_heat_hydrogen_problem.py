import tqdm.autonotebook
from dolfinx import fem

from festim.heat_transfer_problem import HeatTransferProblem
from festim.helpers import nmm_interpolate
from festim.hydrogen_transport_problem import (
    HydrogenTransportProblem,
    HydrogenTransportProblemDiscontinuous,
    HydrogenTransportProblemDiscontinuousChangeVar,
)


class CoupledTransientHeatTransferHydrogenTransport:
    """
    Coupled heat transfer and hydrogen transport transient problem

    Args:
        heat_problem: the heat transfer problem
        hydrogen_problem: the hydrogen transport problem

    Attributes:
        heat_problem: the heat transfer problem
        hydrogen_problem: the hydrogen transport problem
        non_matching_meshes: True if the meshes in the heat_problem and hydrogen_problem
            are not matching

    Examples:
        .. code:: python

            import festim as F

            my_heat_transfer_model = F.HeatTransferProblem(...)

            my_h_transport_model = F.HydrogenTransportProblem(...)

            coupled_problem = F.CoupledTransientHeatTransferHydrogenTransport(
                heat_problem=my_heat_transfer_model,
                hydrogen_problem=my_h_transport_model,
            )


    """

    heat_problem: HeatTransferProblem
    hydrogen_problem: HydrogenTransportProblem

    non_matching_meshes: bool

    def __init__(
        self,
        heat_problem: HeatTransferProblem,
        hydrogen_problem: HydrogenTransportProblem,
    ) -> None:
        self.heat_problem = heat_problem
        self.hydrogen_problem = hydrogen_problem

        if (
            not self.heat_problem.settings.transient
            or not self.hydrogen_problem.settings.transient
        ):
            raise TypeError(
                "Both the heat and hydrogen problems must be set to transient"
            )

    @property
    def heat_problem(self):
        return self._heat_problem

    @heat_problem.setter
    def heat_problem(self, value):
        if not isinstance(value, HeatTransferProblem):
            raise TypeError("heat_problem must be a festim.HeatTransferProblem object")
        value.show_progress_bar = False
        self._heat_problem = value

    @property
    def hydrogen_problem(self):
        return self._hydrogen_problem

    @hydrogen_problem.setter
    def hydrogen_problem(self, value):
        if isinstance(
            value,
            HydrogenTransportProblemDiscontinuous
            | HydrogenTransportProblemDiscontinuousChangeVar,
        ):
            raise NotImplementedError(
                "Coupled heat transfer - hydrogen transport simulations with "
                "HydrogenTransportProblemDiscontinuousChangeVar or"
                "HydrogenTransportProblemDiscontinuous"
                "not currently supported"
            )
        elif not isinstance(value, HydrogenTransportProblem):
            raise TypeError(
                "hydrogen_problem must be a festim.HydrogenTransportProblem object"
            )
        self._hydrogen_problem = value

    @property
    def non_matching_meshes(self):
        return self.heat_problem.mesh.mesh != self.hydrogen_problem.mesh.mesh

    def initialise(self):
        # make sure both problems have the same initial time step and final time,
        # use minimal initial value of the two and maximal final time of the two
        min_initial_dt = min(
            self.heat_problem.settings.stepsize.initial_value,
            self.hydrogen_problem.settings.stepsize.initial_value,
        )
        self.heat_problem.settings.stepsize.initial_value = min_initial_dt
        self.hydrogen_problem.settings.stepsize.initial_value = min_initial_dt

        if (
            self.heat_problem.settings.final_time
            != self.hydrogen_problem.settings.final_time
        ):
            raise ValueError(
                "Final time values in the heat transfer and hydrogen transport "
                "model must be the same"
            )

        self.heat_problem.initialise()

        if self.non_matching_meshes:
            V = fem.functionspace(self.hydrogen_problem.mesh.mesh, ("P", 1))
            T_func = fem.Function(V)

            nmm_interpolate(T_func, self.heat_problem.u)

            self.hydrogen_problem.temperature = T_func
        else:
            self.hydrogen_problem.temperature = self.heat_problem.u

        self.hydrogen_problem.initialise()

    def iterate(self):
        self.heat_problem.iterate()

        if self.non_matching_meshes:
            nmm_interpolate(
                self.hydrogen_problem.temperature_fenics, self.heat_problem.u
            )

        self.hydrogen_problem.iterate()

        # use the same time step for both problems, use minimum of the two
        next_dt_value = min(
            float(self.hydrogen_problem.dt), float(self.heat_problem.dt)
        )
        self.heat_problem.dt.value = next_dt_value
        self.hydrogen_problem.dt.value = next_dt_value

    def run(self):
        if self.hydrogen_problem.show_progress_bar:
            self.hydrogen_problem.progress_bar = tqdm.autonotebook.tqdm(
                desc=f"Solving {self.__class__.__name__}",
                total=self.hydrogen_problem.settings.final_time,
                unit_scale=True,
            )

        while self.hydrogen_problem.t.value < self.hydrogen_problem.settings.final_time:
            self.iterate()

        if self.hydrogen_problem.show_progress_bar:
            self.hydrogen_problem.progress_bar.refresh()
            self.hydrogen_problem.progress_bar.close()

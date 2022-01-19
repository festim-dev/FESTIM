import sys
import fenics as f


class Stepsize:
    def __init__(self, initial_value=0, stepsize_change_ratio=None, t_stop=None, stepsize_stop_max=None, dt_min=None) -> None:
        self.adaptive_stepsize = {}
        if stepsize_change_ratio is not None:
            self.adaptive_stepsize = {
                        "stepsize_change_ratio": stepsize_change_ratio,
                        "t_stop": t_stop,
                        "stepsize_stop_max": stepsize_stop_max,
                        "dt_min": dt_min
                        }
        self.value = f.Constant(initial_value, name="dt")

    def adapt(self, nb_it, converged):
        if not converged:
            self.value.assign(float(self.value)/self.stepsize_change_ratio)
            if float(self.value) < self.dt_min:
                sys.exit('Error: stepsize reached minimal value')
        if nb_it < 5:
            self.value.assign(float(self.value)*self.stepsize_change_ratio)
        else:
            self.value.assign(float(self.value)/self.stepsize_change_ratio)
        return

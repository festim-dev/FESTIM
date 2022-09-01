import fenics as f
import numpy as np
import festim
import warnings
import os

warnings.simplefilter("always", DeprecationWarning)


class TXTExport(festim.Export):
    """

    Args:
        field (str): the exported field ("solute", "1", "retention",
            "T"...)
        times (list): times of export. The stepsize will be modified to
            ensure these timesteps are hit.
        label (str): label of the field. Will also be the filename.
        folder (str): the export folder
    """

    def __init__(self, field, times, label, folder) -> None:

        super().__init__(field=field)
        self.times = sorted(times)
        self.label = label
        self.folder = folder

    def is_it_time_to_export(self, current_time):
        for time in self.times:
            if current_time == time:
                return True

        return False

    def when_is_next_time(self, current_time):
        for time in self.times:
            if current_time < time:
                return time
        return None

    def write(self, current_time, dt):

        # create a DG1 functionspace
        V_DG1 = f.FunctionSpace(self.function.function_space().mesh(), "DG", 1)

        solution = f.project(self.function, V_DG1)
        if self.is_it_time_to_export(current_time):
            filename = "{}/{}_{}s.txt".format(self.folder, self.label, current_time)
            busy = True
            x = f.interpolate(f.Expression("x[0]", degree=1), V_DG1)
            # if the directory doesn't exist
            # create it
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            np.savetxt(filename, np.transpose([x.vector()[:], solution.vector()[:]]))

        #  TODO maybe this should be in another method
        next_time = self.when_is_next_time(current_time)
        if next_time is not None:
            if current_time + float(dt.value) > next_time:
                dt.value.assign(next_time - current_time)


class TXTExports:
    def __init__(self, fields=[], times=[], labels=[], folder=None) -> None:
        self.fields = fields
        if len(self.fields) != len(labels):
            raise ValueError(
                "Number of fields to be exported "
                "doesn't match number of labels in txt exports"
            )
        self.times = sorted(times)
        self.labels = labels
        self.folder = folder
        self.exports = []
        for function, label in zip(self.fields, self.labels):
            self.exports.append(TXTExport(function, times, label, folder))

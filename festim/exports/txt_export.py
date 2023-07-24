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
        label (str): label of the field. Will also be the filename.
        folder (str): the export folder
        times (list, optional): if provided, the stepsize will be modified to
            ensure these timesteps are exported. Otherwise exports at all
            timesteps. Defaults to None.
    """

    def __init__(self, field, label, folder, times=None) -> None:
        super().__init__(field=field)
        if times:
            self.times = sorted(times)
        else:
            self.times = times
        self.label = label
        self.folder = folder

    def is_it_time_to_export(self, current_time):
        if self.times is None:
            return True
        for time in self.times:
            if current_time == time:
                return True

        return False

    def when_is_next_time(self, current_time):
        if self.times is None:
            return None
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
            if dt is None:
                filename = "{}/{}_steady.txt".format(self.folder, self.label)
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
            self.exports.append(TXTExport(function, label, folder, times))

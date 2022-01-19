import fenics as f
import numpy as np
import FESTIM
import warnings


class TXTExport(FESTIM.Export):
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
            x = f.interpolate(f.Expression('x[0]', degree=1), V_DG1)
            while busy is True:
                try:
                    np.savetxt(filename, np.transpose(
                                [x.vector()[:], solution.vector()[:]]))
                    break
                except OSError as err:
                    print("OS error: {0}".format(err))
                    print("The file " + filename + ".txt might currently be busy."
                          "Please close the application then press any key.")
                    input()

        #  TODO maybe this should be in another method
        next_time = self.when_is_next_time(current_time)
        if next_time is not None:
            if current_time + float(dt.value) > next_time:
                dt.value.assign(next_time - current_time)


class TXTExports:
    def __init__(self, fields=[], times=[], labels=[], folder=None, functions=[]) -> None:
        self.fields = fields
        if functions != []:
            self.fields = functions
            msg = "functions key will be deprecated. Please use fields instead"
            warnings.warn(msg, DeprecationWarning)

        if len(self.fields) != len(labels):
            raise ValueError("Number of fields to be exported "
                             "doesn't match number of labels in txt exports")
        self.times = sorted(times)
        self.labels = labels
        self.folder = folder
        self.exports = []
        for function, label in zip(self.fields, self.labels):
            self.exports.append(TXTExport(function, times, label, folder))

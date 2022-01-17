import fenics as f
import numpy as np


class TXTExport:
    def __init__(self, function, times, label, folder) -> None:
        self.function = function
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

    def write(self, label_to_function, current_time, dt):
        solution = label_to_function[self.function]

        filename = "{}/{}_{}".format(self.folder, self.label, current_time)
        busy = True
        x = f.interpolate(f.Expression('x[0]', degree=1), solution.functionspace())
        while busy is True:
            try:
                np.savetxt(filename + '.txt', np.transpose(
                            [x.vector()[:], solution.vector()[:]]))
                return True
            except OSError as err:
                print("OS error: {0}".format(err))
                print("The file " + filename + ".txt might currently be busy."
                      "Please close the application then press any key.")
                input()

        next_time = self.when_is_next_time(current_time)
        if next_time is not None:
            if current_time + float(dt) > next_time:
                dt.assign(next_time - current_time)


class TXTExports:
    def __init__(self, functions, times, labels, folder) -> None:
        if len(functions) != len(labels):
            raise NameError("Number of functions to be exported "
                            "doesn't match number of labels in txt exports")
        self.functions = functions
        self.times = sorted(times)
        self.labels = labels
        self.folder = folder
        self.exports = []
        for function, label in zip(self.functions, self.labels):
            self.exports.append(TXTExport(function, times, label, folder))

    def write(self, label_to_function, current_time, dt):
        if self.is_it_time_to_export(current_time):
            for export in self.exports:
                export.write(label_to_function, current_time, dt)

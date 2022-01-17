from FESTIM import Export
import fenics as f


class XDMFExport(Export):
    def __init__(self, functions, labels, folder, last_timestep_only=False, nb_iterations_between_exports=1, checkpoint=True) -> None:
        super().__init__()
        self.functions = functions
        self.labels = labels
        if len(self.functions) != len(self.labels):
            raise NameError("Number of functions to be exported "
                            "doesn't match number of labels in xdmf exports")
        self.folder = folder
        if self.folder == "":
            raise ValueError("folder value cannot be an empty string")
        if type(self.folder) is not str:
            raise TypeError("folder value must be of type str")
        self.files = None
        self.define_xdmf_files()
        self.last_time_step_only = last_timestep_only
        self.nb_iterations_between_exports = nb_iterations_between_exports
        self.checkpoint = checkpoint
        if type(self.checkpoint) != bool:
            raise TypeError(
                "Unknown value for XDMF checkpoint (True or False)")

        self.append = False

    def define_xdmf_files(self):

        files = list()
        for i in range(0, len(self.functions)):
            u_file = f.XDMFFile(self.folder + '/' +
                                self.labels[i] + '.xdmf')
            u_file.parameters["flush_output"] = True
            u_file.parameters["rewrite_function_mesh"] = False
            files.append(u_file)
        self.files = files

    def write(self, functions, t):
        if len(self.functions) > len(functions):
            raise NameError("Too many functions to export "
                            "in xdmf exports")
        solution_dict = {
            'solute': functions[0],
            'retention': functions[-2],
            'T': functions[-1],
        }
        for label, fun, file in zip(self.labels, self.functions, self.files):
            if type(fun) is int:
                if fun <= len(functions):
                    solution = functions[fun]
                else:
                    raise ValueError(
                        "The value " + str(fun) +
                        " is unknown.")
            elif type(fun) is str:
                if fun.isdigit():
                    fun = int(fun)
                    if fun <= len(functions):
                        solution = functions[fun]
                    else:
                        raise ValueError(
                            "The value " + str(fun) +
                            " is unknown.")
                elif fun in solution_dict.keys():
                    solution = solution_dict[fun]
                else:
                    raise ValueError(
                        "The value " + fun +
                        " is unknown.")
            else:
                raise TypeError('Unexpected' + str(type(fun)) + 'type')
            solution.rename(label, "label")

            if self.checkpoint:
                file.write_checkpoint(
                    solution, label, t, f.XDMFFile.Encoding.HDF5,
                    append=self.append)
            else:
                file.write(solution, t)

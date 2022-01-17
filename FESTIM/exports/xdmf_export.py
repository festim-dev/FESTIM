from FESTIM import Export
import fenics as f


class XDMFExport(Export):
    def __init__(self, function, label, folder, last_timestep_only=False, nb_iterations_between_exports=1, checkpoint=True) -> None:
        super().__init__()
        self.function = function
        self.label = label
        self.folder = folder
        if self.folder == "":
            raise ValueError("folder value cannot be an empty string")
        if type(self.folder) is not str:
            raise TypeError("folder value must be of type str")
        self.files = None
        self.define_xdmf_file()
        self.last_time_step_only = last_timestep_only
        self.nb_iterations_between_exports = nb_iterations_between_exports
        self.checkpoint = checkpoint
        if type(self.checkpoint) != bool:
            raise TypeError(
                "Unknown value for XDMF checkpoint (True or False)")

        self.append = False

    def define_xdmf_file(self):

        self.file = f.XDMFFile(self.folder + '/' +
                               self.label + '.xdmf')
        self.file.parameters["flush_output"] = True
        self.file.parameters["rewrite_function_mesh"] = False

    def write(self, label_to_function, t):
        solution = label_to_function[self.function]
        solution.rename(self.label, "label")

        if self.checkpoint:
            self.file.write_checkpoint(
                solution, self.label, t, f.XDMFFile.Encoding.HDF5,
                append=self.append)
        else:
            self.file.write(solution, t)


class XDMFExports:
    def __init__(self, functions, labels, folder, last_timestep_only=False, nb_iterations_between_exports=1, checkpoint=True) -> None:
        if len(functions) != len(labels):
            raise ValueError("Number of functions to be exported "
                             "doesn't match number of labels in xdmf exports")
        self.xdmf_exports = [
            XDMFExport(
                function, label, folder,
                last_timestep_only=last_timestep_only,
                nb_iterations_between_exports=nb_iterations_between_exports,
                checkpoint=checkpoint)
            for function, label in zip(functions, labels)
        ]

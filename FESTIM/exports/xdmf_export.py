import warnings
from FESTIM import Export
import fenics as f
warnings.simplefilter('always', DeprecationWarning)


class XDMFExport(Export):
    def __init__(self, field, label, folder, last_timestep_only=False, nb_iterations_between_exports=1, checkpoint=True) -> None:
        """Inits XDMFExport

        Args:
            field (str): [description]
            label (str): [description]
            folder (str): [description]
            last_timestep_only (bool, optional): If set to True, will only
                export at the last time step. Defaults to False.
            nb_iterations_between_exports (int, optional): Number of
                iterations between each export. Defaults to 1.
            checkpoint (bool, optional): If set to True,
                fenics.XDMFFile.write_checkpoint will be use, else
                fenics.XDMFFile.write. Defaults to True.

        Raises:
            ValueError: if folder is ""
            TypeError: if folder is not str
            TypeError: if checkpoint is not bool
        """
        super().__init__(field=field)
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
                "checkpoint should be a bool")

        self.append = False

    def define_xdmf_file(self):
        """Creates the file
        """

        self.file = f.XDMFFile(self.folder + '/' +
                               self.label + '.xdmf')
        self.file.parameters["flush_output"] = True
        self.file.parameters["rewrite_function_mesh"] = False

    def write(self, t):
        """Writes to file

        Args:
            t (float): current time
        """
        self.function.rename(self.label, "label")

        if self.checkpoint:
            self.file.write_checkpoint(
                self.function, self.label, t, f.XDMFFile.Encoding.HDF5,
                append=self.append)
        else:
            self.file.write(self.function, t)


class XDMFExports:
    def __init__(self, fields=[], labels=[], folder=None, last_timestep_only=False, nb_iterations_between_exports=1, checkpoint=True, functions=[]) -> None:
        self.fields = fields
        self.labels = labels
        if functions != []:
            self.fields = functions
            msg = "functions key will be deprecated. Please use fields instead"
            warnings.warn(msg, DeprecationWarning)

        if len(self.fields) != len(self.labels):
            raise ValueError("Number of fields to be exported "
                             "doesn't match number of labels in xdmf exports")
        self.xdmf_exports = [
            XDMFExport(
                function, label, folder,
                last_timestep_only=last_timestep_only,
                nb_iterations_between_exports=nb_iterations_between_exports,
                checkpoint=checkpoint)
            for function, label in zip(self.fields, self.labels)
        ]

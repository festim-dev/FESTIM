import warnings
from FESTIM import Export
import fenics as f


class XDMFExport(Export):
    def __init__(self, field, label, folder, mode=1, checkpoint=True) -> None:
        """Inits XDMFExport

        Args:
            field (str): the exported field ("solute", "1", "retention", "T"...)
            label (str): label of the field in the written file
            folder (str): path of the export folder
            mode (int, str, optional): if "last" only the last
                timestep will be exported. Otherwise the number of
                iterations between each export can be provided as an integer.
                Defaults to 1.
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
        self.mode = mode
        self.checkpoint = checkpoint
        if type(self.checkpoint) != bool:
            raise TypeError(
                "checkpoint should be a bool")

        self.append = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        accepted_values = "accepted values for mode are int and 'last'"
        if not isinstance(value, (str, int)):
            raise ValueError(accepted_values)
        if isinstance(value, int) and value <= 0:
            raise ValueError("mode must be positive")
        if isinstance(value, str) and value != "last":
            raise ValueError(accepted_values)

        self._mode = value

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

            # warn users if checkpoint is True and 1D
            dimension = self.function.function_space().mesh().topology().dim()
            if dimension == 1:
                msg = "in 1D, checkpointing is needed to visualise the XDMF "
                msg += "file in Paraview (see issue "
                msg += "https://github.com/RemDelaporteMathurin/FESTIM/issues/134)"
                warnings.warn(msg)

            self.file.write_checkpoint(
                self.function, self.label, t, f.XDMFFile.Encoding.HDF5,
                append=self.append)
        else:
            self.file.write(self.function, t)

    def is_export(self, t, final_time, nb_iterations):
        """Checks if export should be exported.

        Args:
            t (float): the current time
            final_time (float): the final time of the simulation
            nb_iterations (int): the current number of time steps

        Returns:
            bool: True if export should be exported, else False
        """
        if (self.mode == "last" and
                t >= final_time):
            return True
        elif isinstance(self.mode, int):
            if nb_iterations % self.mode == 0:
                return True

        return False


class XDMFExports:
    def __init__(self, fields=[], labels=[], folder=None, mode=1, checkpoint=True, functions=[]) -> None:
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
                mode=mode,
                checkpoint=checkpoint)
            for function, label in zip(self.fields, self.labels)
        ]

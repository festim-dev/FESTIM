import warnings
from festim import Export
import fenics as f


field_to_label = {
    "solute": "mobile_concentration",
    "T": "temperature",
    "retention": "retention",
    "trap": "trap_i_concentration",
}


class XDMFExport(Export):
    """
    Args:
        field (str): the exported field ("solute", "1", "retention", "T"...)
        label (str, optional): label of the field in the written file.
            If None, an automatic label will be given. Defaults to None.
        filename (str, optional): the file path, needs to end with '.xdmf'.
            If None, the label will be used. Defaults to None.
        mode (int, str, optional): if "last" only the last
            timestep will be exported. Otherwise the number of
            iterations between each export can be provided as an integer.
            Defaults to 1.
        checkpoint (bool, optional): If set to True,
            fenics.XDMFFile.write_checkpoint will be use, else
            fenics.XDMFFile.write. Defaults to True.
        folder (str, optional): path of the export folder. Defaults to None.
    """

    def __init__(
        self, field, label=None, filename=None, mode=1, checkpoint=True, folder=None
    ) -> None:

        super().__init__(field=field)
        self.label = label
        self.folder = folder
        self.filename = filename

        self.files = None
        self.define_xdmf_file()
        self.mode = mode
        self.checkpoint = checkpoint
        if type(self.checkpoint) != bool:
            raise TypeError("checkpoint must be a bool")

        self.append = False

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if value is None:
            if self.field in field_to_label.keys():
                self._label = field_to_label[self.field]
            elif self.field.isdigit():
                self._label = field_to_label["trap"].replace("i", self.field, 1)
        else:
            self._label = value

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

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("folder must be a string")

        self._folder = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("filename must be a string")
            if not value.endswith(".xdmf"):
                raise ValueError("filename must end with .xdmf")
            self._filename = value
        else:
            self._filename = "{}.xdmf".format(self.label)

    def define_xdmf_file(self):
        """Creates the file"""
        if self.folder is None:
            filename = self.filename
        else:
            filename = "{}/{}".format(self.folder, self.filename)
        self.file = f.XDMFFile(filename)
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
                msg = "in 1D, checkpoint needs to be set to False to "
                msg += "visualise the XDMF file in Paraview (see issue "
                msg += "https://github.com/RemDelaporteMathurin/festim/issues/134)"
                warnings.warn(msg)

            self.file.write_checkpoint(
                self.function,
                self.label,
                t,
                f.XDMFFile.Encoding.HDF5,
                append=self.append,
            )
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
        if self.mode == "last" and t >= final_time:
            return True
        elif isinstance(self.mode, int):
            if nb_iterations % self.mode == 0:
                return True

        return False


# TODO should we get rid of XDMFExports?


class XDMFExports:
    def __init__(
        self, fields=[], labels=[], folder=None, mode=1, checkpoint=True, functions=[]
    ) -> None:
        self.fields = fields
        self.labels = labels
        if functions != []:
            self.fields = functions
            msg = "functions key will be deprecated. Please use fields instead"
            warnings.warn(msg, DeprecationWarning)

        if len(self.fields) != len(self.labels):
            raise ValueError(
                "Number of fields to be exported "
                "doesn't match number of labels in xdmf exports"
            )
        self.xdmf_exports = [
            XDMFExport(function, label, folder=folder, mode=mode, checkpoint=checkpoint)
            for function, label in zip(self.fields, self.labels)
        ]

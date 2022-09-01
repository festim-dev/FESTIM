import warnings


class InitialCondition:
    """
    Args:
        field (int, str, optional): the field
            ("0", "solute", "T", "1",...). Defaults to 0.
        value (float, str, optional): the value of the initial condition.
            Defaults to 0.
        component (int, str, optional): the field
            ("0", "solute", "T", "1",...). Soon to be deprecated. Defaults to None.
        label (str, optional): label in the XDMF file. Defaults to None.
        time_step ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: if XDMF and label is None
        ValueError: if XDMF and time_step is None
    """

    def __init__(
        self, field=0, value=0.0, component=None, label=None, time_step=None
    ) -> None:

        # TODO make an inherited class InitialConditionXDMF
        self.field = field
        self.value = value
        if component is not None:
            self.field = component
            warnings.warn("components key will be deprecated", DeprecationWarning)
        self.label = label
        self.time_step = time_step
        if type(self.value) == str:
            if self.value.endswith(".xdmf"):
                if self.label is None:
                    raise ValueError("label cannot be None")
                if self.time_step is None:
                    raise ValueError("time_step cannot be None")

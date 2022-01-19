import warnings


class InitialCondition:
    def __init__(self, field=0, value=0, component=None, label=None, time_step=None) -> None:
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

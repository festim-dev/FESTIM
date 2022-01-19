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

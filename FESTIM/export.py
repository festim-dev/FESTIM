class Export:
    def __init__(self, field=None) -> None:
        self.field = field
        self.function = None


class Exports:
    def __init__(self, exports=[]) -> None:
        self.exports = exports

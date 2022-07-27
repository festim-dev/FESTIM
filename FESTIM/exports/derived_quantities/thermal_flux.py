from festim import SurfaceFlux


class ThermalFlux(SurfaceFlux):
    def __init__(self, surface) -> None:
        super().__init__(field="T", surface=surface)

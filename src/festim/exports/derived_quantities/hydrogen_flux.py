from festim import SurfaceFlux


class HydrogenFlux(SurfaceFlux):
    def __init__(self, surface) -> None:
        super().__init__(field="solute", surface=surface)

import dolfinx
from festim.subdomain import VolumeSubdomain
from festim.species import Species


class AdvectionTerm:
    velocity: dolfinx.fem.Function
    subdomain: VolumeSubdomain
    species: Species

    def __init__(
        self,
        velocity: dolfinx.fem.Function,
        subdomain: VolumeSubdomain,
        species: Species,
    ):
        self.velocity = velocity
        self.subdomain = subdomain
        self.species = species

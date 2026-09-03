from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class OutflowBC:
    r"""Lets a drift term carry the species out through a boundary.

    Adds :math:`+ \int_{\partial\Omega} c\, (\mathbf{v} \cdot \mathbf{n})\, w`,
    cancelling the boundary term the divergence form leaves behind, so the natural
    condition on this surface becomes zero *diffusive* flux -- the standard "do-nothing"
    outflow of advection-diffusion. Without it an untagged outlet is a closed end and
    the species backs up against it. See :ref:`outflow` in the user guide.

    It is a no-op on a surface where no drift term acts on ``species``.

    Args:
        subdomain: the surface subdomain the species flows out through. On a
            codimensional problem this may be the boundary of a manifold -- the outlet
            of a 1D fluid, for instance
        species: the species carried out

    Attributes:
        subdomain: the surface subdomain the species flows out through
        species: the species carried out

    Examples:
        .. code-block:: python

            import festim as F

            F.OutflowBC(subdomain=outlet, species=H)
    """

    subdomain: SurfaceSubdomain
    species: Species

    def __init__(self, subdomain: SurfaceSubdomain, species: Species):
        self.subdomain = subdomain
        self.species = species

    @property
    def species(self) -> Species:
        return self._species

    @species.setter
    def species(self, value):
        if not isinstance(value, Species):
            raise TypeError(
                f"species must be of type festim.Species, not {type(value)}"
            )
        self._species = value

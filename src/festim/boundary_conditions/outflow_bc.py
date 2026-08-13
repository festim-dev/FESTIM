from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class OutflowBC:
    r"""Lets a drift term carry the species out through a boundary.

    Drift terms are assembled in divergence form, so the boundary term they leave
    behind makes the flux conditions of the problem constrain the **total** flux. A
    boundary with no condition on it is then a wall: nothing leaves, and the drift is
    balanced by back-diffusion. That is what you want at an actual wall, and wrong at an
    outlet, where whatever the flow reaches should be carried out of the domain.

    This boundary condition adds

    .. math::

        + \int_{\partial\Omega} c\, (\mathbf{v} \cdot \mathbf{n})\, w

    which cancels exactly that term, so the natural condition on this surface becomes
    zero *diffusive* flux, :math:`D \nabla c \cdot \mathbf{n} = 0` -- the standard
    "do-nothing" outflow condition of advection-diffusion. The species leaves at the
    rate the drift carries it.

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

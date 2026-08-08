"""Geometry-agnostic derived quantities.

The historical ``*Surface``/``*Volume`` pairs differ only in which measure and which
meshtags they are handed, which is information the *problem* holds, not the export. The
classes here take a ``domain`` -- a volume or a surface subdomain -- and let the problem
resolve it (see ``measure_for``, ``entities_for`` and ``solution_for`` on
:class:`~festim.HydrogenTransportProblem`).

That split matters for codimensional subdomains, where "surface" and "volume" stop
being exclusive: a codim-1 volume subdomain is a manifold, geometrically a surface,
which the parent-mesh volume meshtags do not tag at all.

Quantities are grouped by *what they compute*, which is the axis along which they
actually differ:

* :class:`IntegralQuantity` assembles a form and needs a measure and entity maps.
* :class:`ExtremumQuantity` reduces over degrees of freedom and needs meshtags and an
  entity dimension, but no measure.
"""

from abc import abstractmethod

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl
from scifem import assemble_scalar

from festim.exports.derived_quantity import DerivedQuantity
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain
from festim.subdomain.volume_subdomain import VolumeSubdomain


class FieldQuantity(DerivedQuantity):
    """A derived quantity of one field over one subdomain.

    Args:
        field: species the quantity is computed for
        domain: the volume or surface subdomain it is computed over
        filename: name of the file to write the quantity to

    Attributes:
        field: species the quantity is computed for
        domain: the volume or surface subdomain it is computed over
        filename: name of the file to write the quantity to
        t: list of time values
        data: list of values of the quantity
    """

    field: Species
    domain: VolumeSubdomain | SurfaceSubdomain
    filename: str | None

    t: list[float]
    data: list[float]

    #: the noun used in :attr:`title`, eg. ``"Total H volume 1"``
    _quantity_name: str = ""

    def __init__(
        self,
        field: Species | str,
        domain: VolumeSubdomain | SurfaceSubdomain,
        filename: str | None = None,
    ) -> None:
        super().__init__(filename=filename)
        self.field = field
        self.domain = domain

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        if not isinstance(value, Species | str):
            raise TypeError("field must be of type F.Species or str")
        self._field = value

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        if not isinstance(value, VolumeSubdomain | SurfaceSubdomain):
            raise TypeError(
                "domain must be of type F.VolumeSubdomain or F.SurfaceSubdomain"
            )
        self._domain = value

    @property
    def domain_name(self) -> str:
        """``"volume"`` or ``"surface"``, for titles.

        Taken from the *declared* type rather than the geometry, so that the title of a
        quantity on a codim-1 volume subdomain still calls it a volume: that is how the
        user declared it, and it keeps csv headers stable across this refactor.
        """
        return "surface" if isinstance(self.domain, SurfaceSubdomain) else "volume"

    @property
    def title(self):
        return (
            f"{self._quantity_name} {self.field.name} "
            f"{self.domain_name} {self.domain.id}"
        )

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class IntegralQuantity(FieldQuantity):
    """A quantity obtained by assembling an integral over :attr:`domain`.

    Subclasses implement :meth:`integrate`, which receives the already-restricted
    measure. The unrestricted measure is what is passed in, so that a single
    ``problem.measure_for(domain)`` serves every integral quantity.
    """

    def compute(self, u, measure: ufl.Measure, entity_maps=None):
        """Assembles the quantity and appends it to :attr:`data`.

        Args:
            u: the field, as something that can appear in a form
            measure: the *unrestricted* measure of the domain, ie. before
                ``measure(domain.id)``. ``dx`` or ``ds`` on the parent mesh, or a ``dx``
                on a submesh for a codimensional subdomain
            entity_maps: entity maps relating the parent mesh and the submeshes
        """
        self.value = self.integrate(u, measure(self.domain.id), entity_maps)
        self.data.append(self.value)
        return self.value

    @abstractmethod
    def integrate(self, u, dmeasure, entity_maps):
        """The value of the quantity, given the restricted measure ``dmeasure``."""


class ExtremumQuantity(FieldQuantity):
    """A quantity obtained by reducing over the degrees of freedom on :attr:`domain`.

    No measure is involved: the dofs of the entities tagged in ``meshtags`` are located
    and reduced across ranks with :attr:`_mpi_op`.
    """

    #: the numpy reduction, eg. ``np.max``
    _reduce = None
    #: the matching MPI reduction, eg. ``MPI.MAX``
    _mpi_op = None

    def compute(self, u: dolfinx.fem.Function, meshtags, entity_dim: int):
        """Reduces the field over the domain and appends the result to :attr:`data`.

        Args:
            u: the field, as a collapsed function whose dofs can be indexed
            meshtags: tags, on the mesh of ``u``, in which the domain is tagged
            entity_dim: the dimension of the tagged entities
        """
        V = u.function_space
        mesh = V.mesh
        mesh.topology.create_connectivity(entity_dim, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=entity_dim, entities=meshtags.find(self.domain.id)
        )

        local = u.x.array[dofs]
        # a rank may hold none of the domain, and numpy has no identity for an empty
        # reduction -- let the MPI reduction supply it instead
        sentinel = np.inf if self._mpi_op is MPI.MIN else -np.inf
        local_value = self._reduce(local) if local.size else sentinel

        self.value = mesh.comm.allreduce(local_value, op=self._mpi_op)
        self.data.append(self.value)
        return self.value


class Total(IntegralQuantity):
    """The integral of a field over a subdomain.

    Args:
        field: species the total is computed for
        domain: the volume or surface subdomain to integrate over
        filename: name of the file to write the total to

    Examples:

        .. testsetup:: Total

            import festim as F
            my_mat = F.Material(D_0=1, E_D=0)
            my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
            H = F.Species("H")

        .. testcode:: Total

            F.Total(field=H, domain=my_vol)
    """

    _quantity_name = "Total"

    def integrate(self, u, dmeasure, entity_maps):
        return assemble_scalar(u * dmeasure, entity_maps=entity_maps)


class Average(IntegralQuantity):
    """The average of a field over a subdomain: its total divided by the measure of
    the subdomain.

    Args:
        field: species the average is computed for
        domain: the volume or surface subdomain to average over
        filename: name of the file to write the average to
    """

    _quantity_name = "Average"

    def integrate(self, u, dmeasure, entity_maps):
        return assemble_scalar(u * dmeasure, entity_maps=entity_maps) / assemble_scalar(
            1 * dmeasure, entity_maps=entity_maps
        )


class Maximum(ExtremumQuantity):
    """The maximum value a field takes on a subdomain.

    Args:
        field: species the maximum is computed for
        domain: the volume or surface subdomain to search
        filename: name of the file to write the maximum to
    """

    _quantity_name = "Maximum"
    _reduce = staticmethod(np.max)
    _mpi_op = MPI.MAX


class Minimum(ExtremumQuantity):
    """The minimum value a field takes on a subdomain.

    Args:
        field: species the minimum is computed for
        domain: the volume or surface subdomain to search
        filename: name of the file to write the minimum to
    """

    _quantity_name = "Minimum"
    _reduce = staticmethod(np.min)
    _mpi_op = MPI.MIN

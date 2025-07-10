from typing import Optional

import numpy as np
import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class SurfaceFlux(SurfaceQuantity):
    """Computes the flux of a field on a given surface

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    field: Species
    surface: SurfaceSubdomain
    filename: str

    title: str
    allowed_meshes: list[str]
    value: float
    data: list[float]

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    @property
    def allowed_meshes(self):
        return ["cartesian"]

    def __init__(
        self, field: Species, surface: SurfaceSubdomain, filename: str | None = None
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)

    def compute(
        self, u: fem.Function | ufl.indexed.Indexed, ds: ufl.Measure, entity_maps=None
    ):
        """Computes the value of the flux at the surface

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """

        # obtain mesh normal from field
        # if case multispecies, solution is an index, use sub_function_space
        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh
        n = ufl.FacetNormal(mesh)

        self.value = assemble_scalar(
            fem.form(
                -self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)


class SurfaceFluxCylindrical(SurfaceFlux):
    """Object to compute the flux J of a field u through a surface
    J = integral(-prop * grad(u) . n ds)
    where prop is the property of the field (D, thermal conductivity, etc)
    u is the field
    n is the normal vector of the surface
    ds is the surface measure in cylindrical coordinates.
    ds = r dr dtheta or ds = r dz dtheta

    .. note::
        For particle fluxes J is given in H/s, for heat fluxes J is given in W

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported
        azimuth_range: Range of the azimuthal angle (theta) needs to be between 0 and
            2 pi. Defaults to (0, 2 * np.pi).
    """

    azimuth_range: tuple[float, float] | None

    def __init__(
        self, field, surface, filename: str | None = None, azimuth_range=(0, 2 * np.pi)
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)
        self.azimuth_range = azimuth_range

    @property
    def allowed_meshes(self):
        return ["cylindrical"]

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > 2 * np.pi:
            raise ValueError("Azimuthal range must be between 0 and 2*pi")
        self._azimuth_range = value

    def compute(self, u, ds: ufl.Measure, entity_maps=None):
        """Computes the value of the flux at the cylindrical surface

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """

        # obtain mesh normal from field
        # if case multispecies, solution is an index, use sub_function_space
        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh
        n = ufl.FacetNormal(mesh)
        r = ufl.SpatialCoordinate(mesh)[0]

        # dS_z = r dr dtheta , assuming axisymmetry dS_z = theta r dr
        # dS_r = r dz dtheta , assuming axisymmetry dS_r = theta r dz
        # in both cases the expression with self.ds is the same

        self.value = assemble_scalar(
            fem.form(
                -r * self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.value *= self.azimuth_range[1] - self.azimuth_range[0]

        self.data.append(self.value)


class SurfaceFluxSpherical(SurfaceFlux):
    """
    Object to compute the flux J of a field u through a surface
    J = integral(-prop * grad(u) . n ds)
    where prop is the property of the field (D, thermal conductivity, etc)
    u is the field
    n is the normal vector of the surface
    ds is the surface measure in spherical coordinates.
    ds = r^2 sin(theta) dtheta dphi

    .. note::
        For particle fluxes J is given in H/s, for heat fluxes J is given in W

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported
        azimuth_range: Range of the azimuthal angle (phi) needs to be between 0 and 2
            pi. Defaults to (0, 2 * np.pi).
        polar_range: Range of the polar angle (theta) needs to be between 0 and pi.
            Defaults to (0, np.pi).
    """

    azimuth_range: tuple[float, float] | None
    polar_range: tuple[float, float] | None

    def __init__(
        self,
        field,
        surface,
        filename: str | None = None,
        azimuth_range=(0, 2 * np.pi),
        polar_range=(0, np.pi),
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)
        self.azimuth_range = azimuth_range
        self.polar_range = polar_range

    @property
    def allowed_meshes(self):
        return ["spherical"]

    @property
    def polar_range(self):
        return self._polar_range

    @polar_range.setter
    def polar_range(self, value):
        if value[0] < 0 or value[1] > np.pi:
            raise ValueError("Polar range must be between 0 and pi")
        self._polar_range = value

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > 2 * np.pi:
            raise ValueError("Azimuthal range must be between 0 and 2 pi")
        self._azimuth_range = value

    def compute(self, u, ds: ufl.Measure, entity_maps=None):
        """Computes the value of the flux at the spherical surface

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """

        # obtain mesh normal from field
        # if case multispecies, solution is an index, use sub_function_space
        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh
        n = ufl.FacetNormal(mesh)
        r = ufl.SpatialCoordinate(mesh)[0]

        # dS_r = r^2 sin(theta) dtheta dphi
        # integral(f dS_r) = integral(f r^2 sin(theta) dtheta dphi)
        #                  = (phi2 - phi1) * (-cos(theta2) + cos(theta1)) * f r^2

        self.value = assemble_scalar(
            fem.form(
                -(r**2) * self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        dphi = self.azimuth_range[1] - self.azimuth_range[0]
        dtheta = -np.cos(self.polar_range[1]) + np.cos(self.polar_range[0])
        self.value *= dphi * dtheta

        self.data.append(self.value)

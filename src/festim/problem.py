import warnings
from typing import Any

from petsc4py import PETSc

import dolfinx
import numpy as np
import tqdm.auto
import ufl

import festim as F
from festim.mesh.mesh import Mesh as _Mesh
from festim.source import SourceBase as _SourceBase
from festim.subdomain.volume_subdomain import (
    VolumeSubdomain as _VolumeSubdomain,
)


class ProblemBase:
    """Base class for :py:class:`HeatTransferProblem
    <festim.heat_transfer_problem.HeatTransferProblem>` and
    :py:class:`HydrogenTransportProblem
    <festim.hydrogen_transport_problem.HydrogenTransportProblem>`.

    Attributes:
        show_progress_bar: If `True` a progress bar is displayed during the simulation
        progress_bar: the progress bar
    """

    mesh: _Mesh
    sources: list[_SourceBase]
    exports: list[Any]
    subdomains: list[_VolumeSubdomain]
    show_progress_bar: bool
    progress_bar: tqdm.auto.tqdm | None
    timesteps: list[float]

    def __init__(
        self,
        mesh: _Mesh = None,
        sources=None,
        exports=None,
        subdomains=None,
        boundary_conditions=None,
        settings=None,
        petsc_options=None,
    ) -> None:
        self.mesh = mesh
        # for arguments to initialise as empty list
        # if arg not None, assign arg, else assign empty list
        self.subdomains = subdomains or []
        self.boundary_conditions = boundary_conditions or []
        self.sources = sources or []
        self.exports = exports or []
        self.settings = settings

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_meshtags = None
        self.volume_meshtags = None
        self.formulation = None
        self.bc_forms = []
        self.show_progress_bar = True
        self.petsc_options = petsc_options
        self._timesteps = []

    @property
    def volume_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.VolumeSubdomain)]

    @property
    def surface_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.SurfaceSubdomain)]

    @property
    def manifold_subdomains(self):
        """The codim-1 volume subdomains: manifolds embedded in the mesh carrying their
        own transport equation. They are tagged in the facet meshtags and may be used
        wherever a surface subdomain is expected."""
        if self.mesh is None or self.mesh.mesh is None:
            return []
        return [s for s in self.volume_subdomains if s.codim(self.mesh.vdim) == 1]

    @property
    def facet_surface_subdomains(self):
        """The surface subdomains that bound an ordinary volume subdomain, ie. the ones
        living in the facet meshtags. These are all of them unless a surface bounds a
        manifold."""
        if self.mesh is None or self.mesh.mesh is None:
            return self.surface_subdomains
        return [s for s in self.surface_subdomains if s.codim(self.mesh.vdim) == 1]

    @property
    def manifold_boundary_subdomains(self):
        """The codim-2 surface subdomains: the boundary of a manifold subdomain (the
        endpoints of a line, the rim of a surface).

        They carry no meshtag -- their entities are located on the manifold's submesh
        when the boundary condition using them is created -- so they take no part in the
        facet-tag id namespace and in ``surface_to_volume``."""
        if self.mesh is None or self.mesh.mesh is None:
            return []
        return [s for s in self.surface_subdomains if s.codim(self.mesh.vdim) == 2]

    @property
    def dt(self):
        return self._dt

    @property
    def timesteps(self):
        return self._timesteps

    def define_meshtags_and_measures(self):
        """Defines the facet and volume meshtags of the model which are used to define
        the measures fo the model, dx and ds."""

        if isinstance(self.mesh, F.MeshFromXDMF):
            # TODO: fix naming inconsistency between facet and surface meshtags
            self.facet_meshtags = self.mesh.define_surface_meshtags()
            self.volume_meshtags = self.mesh.define_volume_meshtags()

        elif (
            isinstance(self.mesh, F.Mesh)
            and self.facet_meshtags is None
            and self.volume_meshtags is None
        ):
            self.facet_meshtags, self.volume_meshtags = self.mesh.define_meshtags(
                # a codim-2 surface bounds a manifold, not the mesh: it has no facet to
                # tag and is resolved on the manifold's submesh instead
                surface_subdomains=self.facet_surface_subdomains,
                volume_subdomains=self.volume_subdomains,
                # if self has attribute interfaces pass it
                interfaces=getattr(self, "interfaces", None),
            )

        # check volume ids are unique
        vol_ids = [vol.id for vol in self.volume_subdomains]
        if len(vol_ids) != len(np.unique(vol_ids)):
            raise ValueError("Volume ids are not unique")

        # manifold subdomains and interfaces live in the facet meshtags, so their ids
        # must not clash with a surface subdomain's: the facets carrying the tag would
        # then be a mix of the two, and neither would integrate over what it claims to
        manifold_ids = [s.id for s in self.manifold_subdomains]
        others = {s.id for s in self.facet_surface_subdomains} | {
            i.id for i in getattr(self, "interfaces", None) or []
        }
        clashing = set(manifold_ids) & others
        if clashing or len(manifold_ids) != len(np.unique(manifold_ids)):
            raise ValueError(
                f"Surface ids {sorted(clashing) or manifold_ids} are not unique. "
                "Codim-1 volume subdomains and interfaces are tagged in the facet "
                "meshtags and therefore share their ids with the surface subdomains."
            )

        # a codim-2 surface has nothing to bound without a manifold, and since it is
        # never tagged it would otherwise just be ignored
        if self.manifold_boundary_subdomains and not self.manifold_subdomains:
            ids = [s.id for s in self.manifold_boundary_subdomains]
            raise ValueError(
                f"Surface subdomains {ids} have dim={self.mesh.vdim - 2}, which bounds "
                f"a manifold volume subdomain (one declared with "
                f"dim={self.mesh.vdim - 1}), but no such subdomain was declared."
            )

        # define measures
        self.ds = ufl.Measure(
            "ds", domain=self.mesh.mesh, subdomain_data=self.facet_meshtags
        )
        self.dx = ufl.Measure(
            "dx", domain=self.mesh.mesh, subdomain_data=self.volume_meshtags
        )

    def define_boundary_conditions(self):
        """Defines the dirichlet boundary conditions of the model."""
        for bc in self.boundary_conditions:
            if isinstance(bc, F.DirichletBCBase):
                if getattr(bc, "_gas_species", None) is not None:
                    # the value depends on an enclosure pressure, which is an unknown
                    # living in a real function space and cannot be interpolated, so
                    # the value stays a ufl expression and the BC is weak-only
                    self.create_dirichletbc_value_ufl(bc)
                    continue
                form = self.create_dirichletbc_form(bc)
                if not bc.enforce_weakly:
                    self.bc_forms.append(form)

    def get_petsc_options(self) -> dict[str, Any]:
        """Gets the PETSc options to pass to the NewtonProblem solver. Default options
        are updated with user-provided options, if any.

        Returns:
            the petsc options to pass to the NewtonProblem solver.
        """
        petsc_options = get_default_petsc_options()

        # Update default PETSc options with user-provided options, if any
        if self.petsc_options:
            petsc_options.update(self.petsc_options)

        if self.petsc_options:
            if (
                "snes_atol" in self.petsc_options
                or "snes_rtol" in self.petsc_options
                or "snes_max_it" in self.petsc_options
            ):
                warnings.warn(
                    "You have set one of the following PETSc options: snes_atol, "
                    "snes_rtol or snes_max_it. These options will be overwritten by "
                    "the values in festim.Settings (atol, rtol and max_iterations) to "
                    "ensure consistency between different versions of dolfinx. If you "
                    "want to set these options manually, please set them in "
                    "festim.Settings and not in the petsc_options dictionary."
                )

        petsc_options.update(
            {
                "snes_atol": self.settings.atol,
                "snes_rtol": self.settings.rtol,
                "snes_max_it": self.settings.max_iterations,
            }
        )

        return petsc_options

    def create_solver(self):
        """Creates the solver of the model."""

        from dolfinx.fem.petsc import NonlinearProblem

        petsc_options = self.get_petsc_options()
        self.solver = NonlinearProblem(
            self.formulation,
            self.u,
            bcs=self.bc_forms,
            petsc_options=petsc_options,
            petsc_options_prefix="festim_solver",
        )

        self.solver.solver.setMonitor(F.helpers.SnesMonitor)
        self.solver.solver.getKSP().setMonitor(F.helpers.KSPMonitor)
        self.solver.solver.setConvergenceTest(F.helpers.convergenceTest)
        # Delete PETSc options post setting them, ref:
        # https://gitlab.com/petsc/petsc/-/issues/1201
        snes = self.solver.solver
        prefix = snes.getOptionsPrefix()
        opts = PETSc.Options()
        for k in petsc_options.keys():
            del opts[f"{prefix}{k}"]

    def run(self):
        """Runs the model."""

        if self.settings.transient:
            # Solve transient
            if self.show_progress_bar:
                self.progress_bar = tqdm.auto.tqdm(
                    desc=f"Solving {self.__class__.__name__}",
                    total=self.settings.final_time,
                    unit_scale=True,
                )
            while self.t.value < self.settings.final_time:
                self.iterate()
            if self.show_progress_bar:
                self.progress_bar.refresh()  # refresh progress bar to show 100%
                self.progress_bar.close()
        else:
            # Solve steady-state
            self.solver.solve()
            self.post_processing()

    def iterate(self):
        """Iterates the model for a given time step."""
        self._timesteps.append(float(self.t))

        if self.show_progress_bar:
            self.progress_bar.update(
                min(self.dt.value, abs(self.settings.final_time - self.t.value))
            )

        # update rtol if it's callable
        if callable(self.settings.rtol):
            self.solver.rtol = self.settings.rtol(self.t.value)
        # update rtol if it's callable
        if callable(self.settings.atol):
            self.solver.atol = self.settings.atol(self.t.value)

        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # solve main problem
        _ = self.solver.solve()
        converged_reason = self.solver.solver.getConvergedReason()
        assert converged_reason > 0, (
            f"Non-linear solver did not converge. Reason code: {converged_reason}. \n See https://petsc.org/release/manualpages/SNES/SNESConvergedReason/ for more information."  # noqa: E501
        )
        nb_its = self.solver.solver.getIterationNumber()

        # post processing
        self.post_processing()

        # update previous solution
        self.u_n.x.array[:] = self.u.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            new_stepsize = self.settings.stepsize.modify_value(
                value=self.dt.value, nb_iterations=nb_its, t=self.t.value
            )
            self.dt.value = new_stepsize

    def update_time_dependent_values(self):
        t = float(self.t)
        for bc in self.boundary_conditions:
            if bc.time_dependent:
                bc.update(t=t)

        for source in self.sources:
            if source.value.explicit_time_dependent:
                source.value.update(t=t)


# DEFAULT PETSC OPTIONS

# taken from https://github.com/FEniCS/dolfinx/blob/5fcb988c5b0f46b8f9183bc844d8f533a2130d6a/python/demo/demo_cahn-hilliard.py#L279C1-L286C28
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
sys = PETSc.Sys()  # type: ignore
if sys.hasExternalPackage("mumps") and not use_superlu:
    linear_solver = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    linear_solver = "superlu_dist"
else:
    linear_solver = "petsc"
_DEFAULT_PETSC_OPTS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(dolfinx.default_real_type).eps) * 1e-2,
    "snes_divergence_tolerance": "PETSC_UNLIMITED",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": linear_solver,
    "snes_error_if_not_converged": True,
    "ksp_error_if_not_converged": True,
}


def get_default_petsc_options():
    return _DEFAULT_PETSC_OPTS.copy()

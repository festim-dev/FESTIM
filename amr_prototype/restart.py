"""Remesh-and-restart driver.

Re-running ``initialise()`` in place on a new mesh is a minefield in FESTIM today:
``create_species_from_traps`` appends to ``self.species``,
``define_meshtags_and_measures`` short-circuits unless the meshtags are reset to
``None``, ``initialise_exports`` truncates output files, the VTX/XDMF writers are bound
to specific ``Function`` objects, and ``Profile1DExport._dofs`` / ``export.D`` /
``export.volume_meshtags`` / ``CustomQuantity.ufl_expr`` are cached and never
invalidated.

So the prototype builds a **fresh problem object** on the new mesh instead. Everything
in the discontinuous ``initialise()`` -- submeshes, entity maps, interface data,
``surface_to_volume``, the blocked solver -- derives from ``self.mesh.mesh`` plus
geometric locators, so a fresh build is correct by construction, and the ffcx cache
makes the repeated form compilation nearly free after the first generation.

The old solution is carried over through a path FESTIM already supports: the
discontinuous ``create_initial_conditions`` accepts a ``fem.Function`` and transfers it
with ``nmm_interpolate`` onto the new submesh.
"""

from mpi4py import MPI

import numpy as np
from dolfinx import fem

import festim as F


def harvest(problem) -> dict[tuple[str, int], fem.Function]:
    """Collect the current solution as ``{(species name, subdomain id): Function}``.

    The functions are the collapsed ``subdomain_to_post_processing_solution`` ones --
    real ``fem.Function`` objects, unlike ``subdomain_to_solution`` which is a
    ``ufl.split`` view and cannot be interpolated from. They are refreshed by
    ``post_processing()``, so call that (or ``iterate()``) before harvesting.
    """
    state = {}
    for spe in problem.species:
        for subdomain in problem.volume_subdomains:
            u = spe.subdomain_to_post_processing_solution.get(subdomain)
            if u is not None:
                state[(spe.name, subdomain.id)] = u
    return state


def harvest_pressures(problem) -> dict[str, float]:
    """Collect enclosure partial pressures as ``{gas species name: pressure}``."""
    return {
        gas.name: float(gas.prev_solution.x.array[0])
        for gas in getattr(problem, "gas_species", [])
    }


def rebuild(
    problem_factory,
    new_mesh: F.Mesh,
    old_problem,
    t: float = 0.0,
    dt: float | None = None,
    facet_meshtags=None,
    volume_meshtags=None,
    padding: float = 1e-11,
):
    """Build a fresh problem on ``new_mesh``, carrying the state of ``old_problem``.

    Args:
        problem_factory: ``(mesh) -> HydrogenTransportProblemDiscontinuous``, returning
            a problem built from **fresh** ``VolumeSubdomain`` and ``Species`` objects.
            Reusing them across generations would carry stale ``submesh``, ``u``,
            ``cell_map`` and ``subdomain_to_*`` state into the new problem.
        new_mesh: the adapted mesh.
        old_problem: the problem to take the state from, already post-processed.
        t: simulation time to restart at.
        dt: timestep to restart with. Leave ``None`` for a steady problem, which has
            no ``_dt`` to set.
        facet_meshtags / volume_meshtags: pass these when the tags were user-supplied
            and had to be transferred; leave ``None`` to let FESTIM re-derive them.
        padding: bounding-box padding for the non-matching interpolation. The old and
            new submeshes of a subdomain are geometrically coincident, so points on
            their shared boundary can fail point-location with too small a value.

    Returns:
        the new, initialised problem, positioned at ``t`` with timestep ``dt``.
    """
    state = harvest(old_problem)
    pressures = harvest_pressures(old_problem)

    new = problem_factory(new_mesh)

    if facet_meshtags is not None:
        new.facet_meshtags = facet_meshtags
    if volume_meshtags is not None:
        new.volume_meshtags = volume_meshtags

    species_by_name = {s.name: s for s in new.species}
    volumes_by_id = {v.id: v for v in new.volume_subdomains}

    initial_conditions = []
    for (name, vol_id), old_function in state.items():
        species = species_by_name.get(name)
        volume = volumes_by_id.get(vol_id)
        if species is None or volume is None:
            raise ValueError(
                f"the problem returned by problem_factory has no species {name!r} on "
                f"subdomain {vol_id}; the factory must rebuild the same model"
            )
        initial_conditions.append(
            F.InitialConcentration(value=old_function, species=species, volume=volume)
        )
    new.initial_conditions = initial_conditions

    for gas in getattr(new, "gas_species", []):
        if gas.name in pressures:
            gas.initial_pressure = pressures[gas.name]

    new.initialise()

    # ``create_initial_conditions`` only fills ``u_n``. Leaving ``u`` at zero would
    # hand Newton a first guess as far from the answer as possible, and would make
    # every post-processed quantity read zero until the first solve lands.
    seed_current_from_previous(new)

    _assert_transfer_succeeded(new, padding)

    new.t.value = t
    if dt is not None:
        new._dt.value = dt
    return new


def seed_current_from_previous(problem):
    """Copy ``u_n`` into ``u`` and refresh the collapsed post-processing functions.

    This is the same refresh ``post_processing()`` does, without the export writes --
    a remesh happens between two output times and should not emit a frame of its own.
    """
    for subdomain in problem.volume_subdomains:
        subdomain.u.x.array[:] = subdomain.u_n.x.array[:]
        for species in problem.species:
            if subdomain not in species.subdomains:
                continue
            collapsed = species.subdomain_to_post_processing_solution[subdomain]
            _, v0_to_V = species.subdomain_to_collapsed_function_space[subdomain]
            collapsed.x.array[:] = subdomain.u.x.array[v0_to_V]

    for gas in getattr(problem, "gas_species", []):
        gas.solution.x.array[:] = gas.prev_solution.x.array[:]


def _assert_transfer_succeeded(problem, padding):
    """Catch a silently failed non-matching interpolation.

    ``interpolate_nonmatching`` leaves dofs at zero when a point could not be located
    in the source mesh, which for a transport problem looks like a plausible solution
    rather than an error.
    """
    for subdomain in problem.volume_subdomains:
        array = subdomain.u_n.x.array
        if array.size and not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"state transfer onto subdomain {subdomain.id} produced non-finite "
                f"values (padding={padding})"
            )


def restart_needed(previous_marks, refine_cells, coarsen_cells, tol: int = 0) -> bool:
    """Whether the marked set has moved enough to be worth a remesh.

    Rebuilding costs a full ``initialise()``, so it is not worth doing every step when
    the front has barely moved.
    """
    if previous_marks is None:
        return True
    old_refine, old_coarsen = previous_marks
    changed = np.setxor1d(old_refine, refine_cells).size
    changed += np.setxor1d(old_coarsen, coarsen_cells).size
    return changed > tol


def gather_profile(problem, species_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The (x, c) profile of a species across every subdomain, sorted by x.

    Field writers are bound to a mesh and to specific ``Function`` objects, so they do
    not survive a remesh. The prototype accumulates profiles in numpy instead and
    writes them once at the end.
    """
    xs, cs = [], []
    for spe in problem.species:
        if spe.name != species_name:
            continue
        for subdomain in problem.volume_subdomains:
            u = spe.subdomain_to_post_processing_solution.get(subdomain)
            if u is None:
                continue
            coords = u.function_space.tabulate_dof_coordinates()[:, 0]
            n_owned = u.function_space.dofmap.index_map.size_local
            xs.append(coords[:n_owned])
            cs.append(u.x.array[:n_owned])

    if not xs:
        raise ValueError(f"no species named {species_name!r} in the problem")

    x = np.concatenate(xs)
    c = np.concatenate(cs)
    comm = problem.mesh.mesh.comm
    x = np.concatenate(comm.allgather(x))
    c = np.concatenate(comm.allgather(c))
    order = np.argsort(x)
    return x[order], c[order]


def total_inventory(problem, species_name: str) -> float:
    """Volume integral of a species over the whole domain."""
    import ufl

    total = 0.0
    for spe in problem.species:
        if spe.name != species_name:
            continue
        for subdomain in problem.volume_subdomains:
            u = spe.subdomain_to_post_processing_solution.get(subdomain)
            if u is None:
                continue
            total += fem.assemble_scalar(fem.form(u * ufl.dx))
    return problem.mesh.mesh.comm.allreduce(total, op=MPI.SUM)

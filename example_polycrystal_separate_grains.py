"""A Voronoi polycrystal in which every grain is its own subdomain.

``example_voronoi_polycrystal.py`` models the same microstructure with a *single* bulk
subdomain: the lattice concentration is then one continuous field across the whole
specimen, and the grain-boundary network can only act as a short circuit -- a fast path
in parallel with the lattice, never a barrier. Giving each grain its own
:class:`festim.VolumeSubdomain` lets the concentration jump from one grain to the next,
so a boundary can also *resist*: transport between two grains then has to go through the
network, and a small exchange rate for one grain cuts it off from the rest.

What that costs, and what it does not:

* the mesh has to tag every Voronoi cell separately -- one gmsh physical group per grain
  instead of one for the whole specimen -- and each grain needs its own
  :class:`festim.Species`, so there are as many transport equations as grains;
* the charged surface has to be split per grain as well, since a
  :class:`festim.SurfaceSubdomain` belongs to exactly one volume subdomain;
* but the network is still declared as **one** codim-1 subdomain, so it keeps a single
  connected field and hydrogen still crosses triple junctions with no junction condition
  to write. FESTIM gives each adjacent grain an interior-facet integral of its own, and
  the coupling is written exactly as it is for two subdomains: one
  :class:`festim.ParticleFluxBC` and one :class:`festim.ParticleSource` per grain, each
  naming that grain's species.

The example solves the microstructure twice -- once with every grain coupled to the
network at the same rate, once with the largest interior grain nearly cut off -- and
reports how far hydrogen got in each case.

Requires ``pip install gmsh`` and scipy.

Run it with ``python example_polycrystal_separate_grains.py``. It writes:

* ``poly_<case>_grains.bp``  -- the lattice concentration of every grain gathered into
  one discontinuous (DG1) field on the parent mesh, so the jump across each boundary
  shows up in a single ParaView dataset;
* ``poly_<case>_network.bp`` -- the concentration on the boundary network, a 1D dataset
  lying on the grain boundaries;
* ``poly_grain_ids.bp``      -- the grain each cell belongs to, ie. the microstructure.

Open them together in ParaView. The network is a line mesh: give it a ``Tube`` filter,
or raise the line width, to see it over the grains.
"""

from mpi4py import MPI

import dolfinx
import gmsh
import numpy as np
from dolfinx.io.gmsh import model_to_mesh
from scipy.spatial import Voronoi

import festim as F

# ---------------------------------------------------------------- parameters
L = 1.0  # specimen size
N_SEEDS = 6  # Voronoi seeds. The tiled ridges cut the box into more grains than this
SEED = 3  # rng seed, so the microstructure is reproducible

D_B = 1e-3  # lattice diffusivity
D_GB = 30.0  # grain-boundary diffusivity
DELTA = 1e-3  # grain-boundary width
K_EX = 1.0  # bulk <-> grain-boundary exchange (see the units warning in the docs)
K_BLOCKED = 1e-6  # the rate given to the blocked grain in the second case
C0 = 1.0  # concentration imposed on the charged surface

H_GB, H_BULK = 0.008, 0.04  # mesh size at the boundaries / inside the grains
T_END, DT = 3.0, 0.05

NETWORK_ID = 1000  # above every grain id, which are 1..n_grains
SURFACE_ID_0 = 2000  # the per-grain charged surfaces are numbered from here


# ---------------------------------------------------------------- microstructure
def voronoi_segments(n_seeds, size, rng):
    """Voronoi ridges of a periodically tiled seed set, clipped to the box."""
    pts = np.column_stack(
        [rng.uniform(0, size, n_seeds), rng.uniform(0, size, n_seeds)]
    )
    tiled = np.vstack(
        [
            pts + np.array([dx * size, dy * size])
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]
    )
    vor = Voronoi(tiled)
    segments = []
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        clipped = clip_to_box(vor.vertices[a], vor.vertices[b], size)
        if clipped is not None and np.linalg.norm(clipped[1] - clipped[0]) > 1e-9:
            segments.append(clipped)
    return segments


def clip_to_box(p, q, size):
    """Liang-Barsky clip of the segment ``pq`` to ``[0, size]^2``."""
    d = q - p
    t0, t1 = 0.0, 1.0
    for num, den in (
        (-d[0], p[0]),
        (d[0], size - p[0]),
        (-d[1], p[1]),
        (d[1], size - p[1]),
    ):
        if abs(num) < 1e-15:
            if den < 0:
                return None
            continue
        r = den / num
        if num < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    return None if t0 > t1 else (p + t0 * d, p + t1 * d)


def near_segments(points, segments, tol=1e-7):
    """Vectorised test: is each point within ``tol`` of any segment?"""
    px, py = points[0], points[1]
    hit = np.zeros(px.shape, dtype=bool)
    for p, q in segments:
        d = q - p
        t = np.clip(((px - p[0]) * d[0] + (py - p[1]) * d[1]) / (d @ d), 0.0, 1.0)
        dx, dy = px - (p[0] + t * d[0]), py - (p[1] + t * d[1])
        hit |= dx * dx + dy * dy < tol * tol
    return hit


def build_mesh(segments, size):
    """A triangular mesh conforming to every grain boundary, with ONE PHYSICAL GROUP
    PER GRAIN.

    That single line is the whole difference from the single-subdomain version:
    ``occ.fragment`` splits the rectangle along every ridge either way, but tagging each
    resulting surface separately is what gives FESTIM one volume subdomain per grain.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polycrystal")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0, 0, 0, size, size)
    lines = [
        occ.addLine(occ.addPoint(p[0], p[1], 0), occ.addPoint(q[0], q[1], 0))
        for p, q in segments
    ]
    _, out_map = occ.fragment([(2, rect)], [(1, ln) for ln in lines])
    occ.synchronize()

    grain_surfaces = [tag for (dim, tag) in out_map[0] if dim == 2]
    for i, tag in enumerate(grain_surfaces):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)  # <-- one group per grain

    gb_curves = [tag for entry in out_map[1:] for (dim, tag) in entry if dim == 1]
    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "CurvesList", gb_curves)
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", H_GB)
    field.setNumber(2, "SizeMax", H_BULK)
    field.setNumber(2, "DistMin", 2 * H_GB)
    field.setNumber(2, "DistMax", 12 * H_GB)
    field.setAsBackgroundMesh(2)
    for opt in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(2)
    result = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    mesh = result.mesh if hasattr(result, "mesh") else result[0]
    cell_tags = result.cell_tags if hasattr(result, "cell_tags") else result[1]
    return mesh, cell_tags, len(grain_surfaces)


# ---------------------------------------------------------------- subdomains
class Grain(F.VolumeSubdomain):
    """One Voronoi cell, located from its gmsh physical group.

    A locator cannot separate one grain from the next -- they are the same material and
    have no analytical description -- so the cells are read straight from the tags the
    mesh was generated with.
    """

    def __init__(self, id, material, cell_tags):
        super().__init__(id=id, material=material)
        self.cell_tags = cell_tags

    def locate_subdomain_entities(self, mesh):
        return self.cell_tags.find(self.id).astype(np.int32)


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain.

    ``locate_subdomain_entities`` is overridden rather than passing a ``locator``:
    ``locate_entities`` marks a facet when *all its vertices* satisfy the locator, which
    near a triple junction also catches short facets that merely touch two different
    boundaries. Testing the facet midpoint instead selects the network exactly.
    """

    def __init__(self, id, material, segments):
        super().__init__(id=id, material=material, dim=1)
        self.segments = segments

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
        candidates = dolfinx.mesh.locate_entities(
            mesh, tdim - 1, lambda x: near_segments(x, self.segments)
        )
        x = mesh.geometry.x
        midpoints = np.array(
            [x[facet_to_vertex.links(f)].mean(axis=0) for f in candidates]
        )
        keep = near_segments(midpoints.T, self.segments)
        return candidates[keep].astype(np.int32)


class ChargedSurfaceOfGrain(F.SurfaceSubdomain):
    """The part of the charged surface belonging to one grain.

    A :class:`festim.SurfaceSubdomain` must belong to exactly one volume subdomain, so
    a surface crossing several grains has to be declared once per grain.
    """

    def __init__(self, id, grain_id, cell_tags, y):
        super().__init__(id=id)
        self.grain_id, self.cell_tags, self.y = grain_id, cell_tags, y

    def locate_boundary_facet_indices(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
        cells = set(self.cell_tags.find(self.grain_id).tolist())
        facets = dolfinx.mesh.locate_entities_boundary(
            mesh, tdim - 1, lambda x: np.isclose(x[1], self.y)
        )
        keep = [f for f in facets if any(c in cells for c in facet_to_cell.links(f))]
        return np.array(keep, dtype=np.int32)


# ---------------------------------------------------------------- visualisation
def parent_field(model, grains, species, name="c"):
    """One discontinuous field on the parent mesh holding every grain's solution.

    Each grain lives on its own submesh, so a plain export writes one file per grain --
    a dozen files to load. Interpolating them all into a single DG1 function on the
    parent mesh instead gives one ParaView dataset in which the jump across each grain
    boundary is visible.
    """
    V = dolfinx.fem.functionspace(model.mesh.mesh, ("DG", 1))
    field = dolfinx.fem.Function(V, name=name)

    def update():
        for spe, grain in zip(species, grains, strict=True):
            parent_cells = model.volume_meshtags.find(grain.id)
            sub_cells = grain.cell_map.sub_topology_to_topology(
                parent_cells, inverse=True
            )
            field.interpolate(
                spe.subdomain_to_post_processing_solution[grain],
                cells0=sub_cells,
                cells1=parent_cells,
            )

    return field, update


def write_grain_ids(model, grains, filename="poly_grain_ids.bp"):
    """A constant-per-cell field labelling the grains, ie. the microstructure."""
    V = dolfinx.fem.functionspace(model.mesh.mesh, ("DG", 0))
    ids = dolfinx.fem.Function(V, name="grain_id")
    for grain in grains:
        cells = model.volume_meshtags.find(grain.id)
        ids.x.array[V.dofmap.list[cells].reshape(-1)] = grain.id
    with dolfinx.io.VTXWriter(model.mesh.mesh.comm, filename, [ids], "BP5") as writer:
        writer.write(0.0)
    return filename


# ---------------------------------------------------------------- build and solve
def build(mesh, cell_tags, n_grains, segments, rates):
    """The polycrystal: one subdomain and one species per grain, one network, and one
    exchange per grain.

    ``rates[i]`` is the exchange rate of grain ``i + 1`` with the network.
    """
    grains = [
        Grain(id=i + 1, material=F.Material(D_0=D_B, E_D=0.0), cell_tags=cell_tags)
        for i in range(n_grains)
    ]
    network = GrainBoundaryNetwork(
        id=NETWORK_ID, material=F.Material(D_0=D_GB, E_D=0.0), segments=segments
    )
    # the charged surface, split per grain; grains that do not touch it are dropped
    charged = [
        ChargedSurfaceOfGrain(
            id=SURFACE_ID_0 + i, grain_id=i + 1, cell_tags=cell_tags, y=L
        )
        for i in range(n_grains)
    ]
    charged = [s for s in charged if len(s.locate_boundary_facet_indices(mesh)) > 0]

    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    # one exchange per grain. This is the same pair of objects the two-subdomain case
    # uses, written once per grain: each half names that grain's species, and that is
    # how FESTIM works out which side of the network the term belongs to
    sources = [
        F.ParticleSource(
            value=lambda c_gb_, c_bulk, k=k: (2.0 / DELTA) * k * (c_bulk - c_gb_),
            species=c_gb,
            volume=network,
            species_dependent_value={"c_bulk": spe, "c_gb_": c_gb},
        )
        for spe, k in zip(species, rates, strict=True)
    ]
    bcs = [
        F.ParticleFluxBC(
            subdomain=network,
            species=spe,
            value=lambda c_gb_, c_bulk, k=k: k * (c_gb_ - c_bulk),
            species_dependent_value={"c_bulk": spe, "c_gb_": c_gb},
        )
        for spe, k in zip(species, rates, strict=True)
    ]
    bcs += [
        F.FixedConcentrationBC(
            subdomain=surface, value=C0, species=species[surface.grain_id - 1]
        )
        for surface in charged
    ]

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[*species, c_gb],
        subdomains=[*grains, network, *charged],
        sources=sources,
        boundary_conditions=bcs,
        temperature=500,
        settings=F.Settings(
            atol=1e-14,
            rtol=1e-12,
            transient=True,
            final_time=T_END,
            stepsize=F.Stepsize(initial_value=DT),
        ),
    )
    model.show_progress_bar = False
    return model, grains, network, species, c_gb, {s.grain_id for s in charged}


def run(case, mesh, cell_tags, n_grains, segments, rates):
    """Solve one case, writing a frame of both files at every timestep."""
    model, grains, network, species, c_gb, charged_ids = build(
        mesh, cell_tags, n_grains, segments, rates
    )
    model.initialise()

    grains_field, update_grains = parent_field(model, grains, species)
    network_field = c_gb.subdomain_to_post_processing_solution[network]
    comm = mesh.comm

    with (
        dolfinx.io.VTXWriter(
            comm, f"poly_{case}_grains.bp", [grains_field], "BP5"
        ) as grains_writer,
        dolfinx.io.VTXWriter(
            comm, f"poly_{case}_network.bp", [network_field], "BP5"
        ) as network_writer,
    ):
        update_grains()
        grains_writer.write(0.0)
        network_writer.write(0.0)
        while model.t.value < model.settings.final_time:
            model.iterate()
            update_grains()
            grains_writer.write(float(model.t))
            network_writer.write(float(model.t))

    means = {
        g.id: float(spe.subdomain_to_post_processing_solution[g].x.array.mean())
        for spe, g in zip(species, grains, strict=True)
    }
    return model, grains, network, c_gb, charged_ids, means


if __name__ == "__main__":
    segments = voronoi_segments(N_SEEDS, L, np.random.default_rng(SEED))
    mesh, cell_tags, n_grains = build_mesh(segments, L)
    areas = {i + 1: len(cell_tags.find(i + 1)) for i in range(n_grains)}
    print(
        f"microstructure: {N_SEEDS} seeds -> {n_grains} grains, "
        f"{len(segments)} boundary segments, "
        f"{mesh.topology.index_map(2).size_global} cells"
    )

    model, grains, network, c_gb, charged_ids, open_means = run(
        "open", mesh, cell_tags, n_grains, segments, [K_EX] * n_grains
    )
    n_sides = len(model.manifold_to_volumes[network])
    print(f"the network is adjacent to {n_sides} grains")
    print(f"  -> {write_grain_ids(model, grains)}")
    print("  -> poly_open_grains.bp, poly_open_network.bp")

    # cut off the largest grain that is not charged directly, so that the only way in
    # would have been through the network
    blocked = max((g for g in areas if g not in charged_ids), key=areas.get)
    rates = [K_BLOCKED if i + 1 == blocked else K_EX for i in range(n_grains)]
    *_, blocked_means = run("blocked", mesh, cell_tags, n_grains, segments, rates)
    print("  -> poly_blocked_grains.bp, poly_blocked_network.bp")

    print(f"\nmean concentration after t = {T_END}, by grain")
    print(f"{'grain':>6} {'cells':>7} {'charged':>8} {'open':>11} {'blocked':>11}")
    for gid in sorted(areas):
        mark = "yes" if gid in charged_ids else ""
        flag = "  <- blocked" if gid == blocked else ""
        print(
            f"{gid:6d} {areas[gid]:7d} {mark:>8} "
            f"{open_means[gid]:11.4e} {blocked_means[gid]:11.4e}{flag}"
        )
    ratio = open_means[blocked] / max(blocked_means[blocked], 1e-300)
    print(
        f"\ngrain {blocked} holds {ratio:.0f}x less hydrogen once its exchange with"
        " the network is cut,\nwhile the rest of the polycrystal is barely affected."
        " A single-subdomain polycrystal\ncannot represent that: there the lattice"
        " field is continuous across every boundary,\nso nothing can block it."
    )

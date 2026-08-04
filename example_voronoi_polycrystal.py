"""Short-circuit diffusion through the grain-boundary network of a Voronoi polycrystal.

Where ``example_fisher_grain_boundary.py`` models a *single* grain boundary and checks
it against the analytical Whipple/Le Claire solution, this one models the whole
connected network of a random polycrystal: many boundaries meeting at triple junctions,
charged from one surface.

The point of the example is that the entire network is declared as **one** codim-1
subdomain with **one** species. That is what makes triple junctions work: the submesh
built from all the grain-boundary facets is topologically connected, so a single
continuous field lives on it and hydrogen crosses from one boundary to another with no
junction condition to write. Declaring one subdomain per boundary instead would give
each its own function space, and the network would be disconnected at the junctions.

Because every grain is the same material, the grains are also a *single* volume
subdomain, and the network sits inside it rather than between two subdomains. FESTIM
decides that the coupling is an interior-facet integral from the mesh topology, not from
the number of subdomains the manifold separates.

Note that this is a demonstration, not a verification: a random network has no
closed-form solution. The quantitative check lives in the Fisher example.

Requires: ``pip install gmsh`` (also used through ``dolfinx.io.gmsh``) and scipy.
"""

from collections import Counter

from mpi4py import MPI

import dolfinx
import gmsh
import numpy as np
import ufl
from dolfinx.io.gmsh import model_to_mesh
from scipy.spatial import Voronoi

import festim as F

# ---------------------------------------------------------------- parameters
L = 1.0  # specimen size
N_SEEDS = 12  # number of grains
SEED = 3  # rng seed, so the microstructure is reproducible

D_B = 1e-3  # lattice diffusivity
D_GB = 30.0  # grain-boundary diffusivity, 1e4 x faster
DELTA = 1e-3  # grain-boundary width
K_EX = 1.0  # bulk <-> grain-boundary exchange (see the Fisher example on units)
C0 = 1.0  # surface concentration

H_GB, H_BULK = 0.008, 0.04  # mesh size at the boundaries / in the grain interiors
T_END, DT = 3.0, 0.02


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
    """A triangular mesh whose facets conform to every grain boundary, refined there."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polycrystal")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0, 0, 0, size, size)
    lines = [
        occ.addLine(occ.addPoint(p[0], p[1], 0), occ.addPoint(q[0], q[1], 0))
        for p, q in segments
    ]
    # fragment forces the surface to be split along every segment, so the generated
    # mesh has facets lying exactly on the grain boundaries
    out, out_map = occ.fragment([(2, rect)], [(1, ln) for ln in lines])
    occ.synchronize()

    gb_curves = [tag for entry in out_map[1:] for (dim, tag) in entry if dim == 1]
    gmsh.model.addPhysicalGroup(2, [t for (d, t) in out if d == 2], 1)

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
    return result.mesh if hasattr(result, "mesh") else result[0]


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


# ---------------------------------------------------------------- build and solve
segments = voronoi_segments(N_SEEDS, L, np.random.default_rng(SEED))
mesh = build_mesh(segments, L)

grains = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_B, E_D=0.0),
    locator=lambda x: np.full_like(x[0], True, dtype=bool),
)
network = GrainBoundaryNetwork(
    id=2, material=F.Material(D_0=D_GB, E_D=0.0), segments=segments
)
top = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], L))
# every point where a grain boundary meets the charged surface, in one object: the
# locator runs on the network itself, so dim = mesh dimension - 2
mouths = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[1], L))

c_b = F.Species("c_b", subdomains=[grains])
c_gb = F.Species("c_gb", subdomains=[network])


def solve(d_gb):
    """Run the problem, returning the two solutions. ``d_gb == D_B`` is the reference
    case in which the boundaries are not short circuits at all."""
    network.material = F.Material(D_0=d_gb, E_D=0.0)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_b, c_gb],
        subdomains=[grains, network, top, mouths],
        sources=[
            F.ParticleSource(
                value=lambda cb, cg: (2.0 / DELTA) * K_EX * (cb - cg),
                species=c_gb,
                volume=network,
                species_dependent_value={"cb": c_b, "cg": c_gb},
            )
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=network,
                species=c_b,
                value=lambda cb, cg: K_EX * (cg - cb),
                species_dependent_value={"cb": c_b, "cg": c_gb},
            ),
            F.FixedConcentrationBC(subdomain=top, value=C0, species=c_b),
            F.FixedConcentrationBC(subdomain=mouths, value=C0, species=c_gb),
        ],
        temperature=500,
        settings=F.Settings(
            atol=1e-14,
            rtol=1e-12,
            transient=True,
            final_time=T_END,
            stepsize=F.Stepsize(initial_value=DT),
        ),
        exports=[
            F.VTXSpeciesExport("voronoi_grains.bp", field=c_b, subdomain=grains),
            F.VTXSpeciesExport("voronoi_network.bp", field=c_gb, subdomain=network),
        ]
        if d_gb != D_B
        else [],
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()
    return (
        model,
        c_b.subdomain_to_post_processing_solution[grains],
        (c_gb.subdomain_to_post_processing_solution[network]),
    )


model, cb_fast, cgb_fast = solve(D_GB)


# ---------------------------------------------------------------- what we built
def connected_components(segs):
    """Union-find over shared segment endpoints."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p, q in segs:
        a, b = tuple(np.round(p, 9)), tuple(np.round(q, 9))
        parent[find(a)] = find(b)
    return len({find(k) for k in parent})


ends = Counter(tuple(np.round(p, 9)) for seg in segments for p in seg)
junctions = [
    k
    for k, n in ends.items()
    if n >= 3 and 1e-9 < k[0] < L - 1e-9 and 1e-9 < k[1] < L - 1e-9
]
ridge_length = sum(float(np.linalg.norm(q - p)) for p, q in segments)

sub = network.submesh
sub.topology.create_connectivity(1, 0)
f2v = sub.topology.connectivity(1, 0)
facet_length = sum(
    float(np.linalg.norm(np.diff(sub.geometry.x[f2v.links(f)], axis=0)))
    for f in range(sub.topology.index_map(1).size_local)
)

print(f"microstructure: {N_SEEDS} grains, {len(segments)} boundary segments")
print(f"  triple junctions inside the box : {len(junctions)}")
print(f"  connected components            : {connected_components(segments)}")
print(
    f"  mesh                            : "
    f"{mesh.topology.index_map(2).size_global} cells"
)
print(
    f"  network captured by the submesh  : {facet_length:.4f} of {ridge_length:.4f}"
    f" ({100 * facet_length / ridge_length:.2f} %)"
)
print(f"  interior facets                 : {model.manifold_is_interior(network)}")


# ---------------------------------------------------------------- effect of the network
def inventory(cb, cgb):
    """Total hydrogen: the grains plus the boundary slabs."""
    dx_bulk = ufl.Measure("dx", domain=cb.function_space.mesh)
    dx_gb = ufl.Measure("dx", domain=cgb.function_space.mesh)
    total = dolfinx.fem.assemble_scalar(dolfinx.fem.form(cb * dx_bulk))
    total += DELTA * dolfinx.fem.assemble_scalar(dolfinx.fem.form(cgb * dx_gb))
    return mesh.comm.allreduce(total, op=MPI.SUM)


fast = inventory(cb_fast, cgb_fast)
# The deepest a boundary touching the charged surface reaches. Below this depth nothing
# is fed directly, so whatever arrives has crossed at least one triple junction.
touching = [(p, q) for p, q in segments if max(p[1], q[1]) > L - 1e-9]
junction_only_below = min((min(p[1], q[1]) for p, q in touching), default=L)

gb_y = cgb_fast.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < junction_only_below
c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

_, cb_ref, cgb_ref = solve(D_B)
ref = inventory(cb_ref, cgb_ref)

print(
    f"\nafter t = {T_END} (lattice diffusion alone reaches "
    f"~{2 * np.sqrt(D_B * T_END):.3g})"
)
print(f"  inventory with fast boundaries : {fast:.4e}")
print(f"  inventory with D_gb = D_b      : {ref:.4e}")
print(f"  enhancement                    : x {fast / ref:.1f}")

# For scale only: Hart's effective diffusivity is the upper bound you would get if every
# boundary ran straight along the gradient. A real network is tortuous and only partly
# connected to the source, so the observed enhancement is well below it.
f_gb = DELTA * ridge_length / L**2
print(f"  boundary area fraction f       : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : {f_gb * D_GB + (1 - f_gb) * D_B:.3e}"
    f"  (vs D_b = {D_B:.3e})"
)

beta = DELTA * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * T_END))
print(f"  type-B parameter beta          : {beta:.0f}  (short circuit needs beta >> 1)")

bulk_y = cb_fast.function_space.tabulate_dof_coordinates()[:, 1]
c_grain_deep = cb_fast.x.array[bulk_y < junction_only_below].mean()

print("\njunction transport: no boundary touching the charged surface reaches below")
print(
    f"y = {junction_only_below:.3f}, so everything the network holds there has crossed"
)
print("at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")

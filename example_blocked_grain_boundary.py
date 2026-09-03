"""A grain boundary that blocks one grain, in a three-grain triple junction.

Three grains meet at a T-shaped junction in a unit square, each declared as its own
:class:`festim.VolumeSubdomain`, and the whole boundary network -- the line y=0.5 plus
the branch x=0.5 below it -- is one codim-1 subdomain carrying its own transport
equation::

           grain 1 (top)
    ---------------------------      <- network, y = 0.5
      grain 2   |   grain 3
     (charged)  |                    <- network, x = 0.5
                |

Only grain 2 is charged, so grains 1 and 3 are fed entirely through the network. The
exchange rate is declared per grain, and the example runs the problem twice: once with
all three grains coupled normally, and once with grain 3 nearly blocked off. Because the
grains are separate subdomains, the concentration is free to jump from one to the
next -- in the blocked case grain 3 stays empty while its neighbours fill up, which is
exactly what a single-subdomain polycrystal cannot represent.

Run it with ``python example_blocked_grain_boundary.py``. It writes, for each case:

* ``<case>_grains.bp``  -- the bulk concentration of all three grains gathered into one
  discontinuous (DG1) field on the parent mesh, so the jumps across the boundaries are
  visible in a single ParaView dataset;
* ``<case>_network.bp`` -- the concentration on the boundary network itself, a 1D
  dataset lying on the grain boundaries;
* ``grain_ids.bp``      -- which grain each cell belongs to, to see the microstructure.

Open them together in ParaView. The network is a line mesh, so give it a ``Tube`` filter
(or just increase the line width) to see it on top of the grains.
"""

from mpi4py import MPI

import dolfinx
import numpy as np

import festim as F

# ---------------------------------------------------------------- parameters
N = 40  # mesh divisions per side
D_BULK = 1.5  # lattice diffusivity of every grain
D_GB = 20.0  # diffusivity along the boundary network -- a short circuit
K = 1.0  # bulk <-> boundary exchange rate (see the units warning in the docs)
K_BLOCKED = 1e-4  # the rate given to the blocked grain instead
C0 = 1.0  # concentration imposed on the charged surface
T_END, DT = 0.4, 0.01

TOP, BOTTOM_LEFT, BOTTOM_RIGHT = 1, 2, 3
NETWORK_ID, CHARGED_ID = 100, 200


def on_network(x):
    """The T: the line y=0.5, plus the branch x=0.5 below it."""
    return np.isclose(x[1], 0.5) | (np.isclose(x[0], 0.5) & (x[1] <= 0.5 + 1e-14))


class Network(F.VolumeSubdomain):
    """The whole T as one codim-1 subdomain.

    ``locate_subdomain_entities`` is overridden rather than passing ``on_network`` as a
    plain ``locator``: ``locate_entities`` marks a facet when *all its vertices* satisfy
    the locator, so near the junction it also catches the diagonal running from a vertex
    of the vertical branch to a vertex of the horizontal line -- a facet lying on
    neither. That stray diagonal shows up in ParaView as a little fork, and looks like
    extra triple junctions. Testing the facet midpoint instead selects the T exactly.
    """

    def __init__(self, id, material, dim):
        super().__init__(id=id, material=material, dim=dim)

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
        candidates = dolfinx.mesh.locate_entities(mesh, tdim - 1, on_network)
        x = mesh.geometry.x
        midpoints = np.array(
            [x[facet_to_vertex.links(f)].mean(axis=0) for f in candidates]
        )
        return candidates[on_network(midpoints.T)].astype(np.int32)


def build(rates):
    """The three grains, the network, and one exchange per grain.

    ``rates`` gives the exchange rate of each grain with the network, in the order
    (top, bottom left, bottom right).
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    grains = [
        F.VolumeSubdomain(
            id=TOP,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: x[1] >= 0.5 - 1e-14,
        ),
        F.VolumeSubdomain(
            id=BOTTOM_LEFT,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: (x[1] <= 0.5 + 1e-14) & (x[0] <= 0.5 + 1e-14),
        ),
        F.VolumeSubdomain(
            id=BOTTOM_RIGHT,
            material=F.Material(D_0=D_BULK, E_D=0.0),
            locator=lambda x: (x[1] <= 0.5 + 1e-14) & (x[0] >= 0.5 - 1e-14),
        ),
    ]
    # the whole network is ONE codim-1 subdomain, so it carries a single connected
    # field and hydrogen crosses the triple junction with no junction condition to write
    network = Network(
        id=NETWORK_ID,
        material=F.Material(D_0=D_GB, E_D=0.0),
        dim=mesh.topology.dim - 1,
    )
    charged = F.SurfaceSubdomain(
        id=CHARGED_ID,
        locator=lambda x: np.isclose(x[1], 0.0) & (x[0] <= 0.5 + 1e-14),
    )

    # one species per grain: separate subdomains means separate function spaces, which
    # is what lets the concentration jump across a boundary
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    # one exchange per grain. Each half names that grain's species, and that is how
    # FESTIM works out which side of the network the term belongs to -- nothing else
    # has to be specified, and there may be as many of them as there are grains
    sources = [
        F.ParticleSource(
            value=lambda c_gb_, c_bulk, k=k: k * (c_bulk - c_gb_),
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
    bcs.append(
        F.FixedConcentrationBC(subdomain=charged, value=C0, species=species[1]),
    )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[*species, c_gb],
        subdomains=[*grains, network, charged],
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
    return model, grains, network, species, c_gb


# ---------------------------------------------------------------- visualisation
def parent_field(model, grains, species, name="c"):
    """One discontinuous field on the parent mesh holding every grain's solution.

    Each grain lives on its own submesh, so a plain export writes one file per grain.
    Interpolating them all into a single DG1 function on the parent mesh instead gives
    one ParaView dataset in which the jump across each grain boundary is visible.
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


def write_grain_ids(model, grains, filename="grain_ids.bp"):
    """A constant-per-cell field labelling the grains, to see the microstructure."""
    V = dolfinx.fem.functionspace(model.mesh.mesh, ("DG", 0))
    ids = dolfinx.fem.Function(V, name="grain_id")
    for grain in grains:
        cells = model.volume_meshtags.find(grain.id)
        ids.x.array[V.dofmap.list[cells].reshape(-1)] = grain.id
    with dolfinx.io.VTXWriter(model.mesh.mesh.comm, filename, [ids], "BP5") as writer:
        writer.write(0.0)


def run(case, rates):
    """Solve one case, writing a frame of both files at every timestep."""
    model, grains, network, species, c_gb = build(rates)
    model.show_progress_bar = False
    model.initialise()

    grains_field, update_grains = parent_field(model, grains, species)
    network_field = c_gb.subdomain_to_post_processing_solution[network]
    comm = model.mesh.mesh.comm

    with (
        dolfinx.io.VTXWriter(
            comm, f"{case}_grains.bp", [grains_field], "BP5"
        ) as grains_writer,
        dolfinx.io.VTXWriter(
            comm, f"{case}_network.bp", [network_field], "BP5"
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

    print(f"\n{case}: exchange rates {rates}")
    for spe, grain in zip(species, grains, strict=True):
        c = spe.subdomain_to_post_processing_solution[grain].x.array
        print(f"  grain {grain.id}: max {c.max():.4e}   mean {c.mean():.4e}")
    gb = c_gb.subdomain_to_post_processing_solution[network].x.array
    print(f"  network: max {gb.max():.4e}   mean {gb.mean():.4e}")
    print(f"  -> {case}_grains.bp, {case}_network.bp")
    return model, grains


if __name__ == "__main__":
    model, grains = run("open", rates=(K, K, K))
    write_grain_ids(model, grains)
    print("  -> grain_ids.bp")

    # the same problem with grain 3 nearly cut off from the network
    run("blocked", rates=(K, K, K_BLOCKED))
    print(
        "\nCompare the two in ParaView: in the blocked case grain 3 stays empty while "
        "\ngrain 1, which is coupled normally, fills through the same network."
    )

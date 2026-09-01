import os
from unittest.mock import patch

import mpi4py.MPI as MPI

import dolfinx
import h5py
import io4dolfinx
import numpy as np
import pytest

import festim as F


def test_writing_and_reading_of_species_function_using_checkpoints(tmpdir):
    """Tests that a model can write a checkpoint file and another model can read it."""
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, nx=10, ny=10, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
    )
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh)

    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    surf = F.SurfaceSubdomain(id=1)
    my_model.subdomains = [vol, surf]

    H = F.Species("H")
    D = F.Species("D")
    my_model.species = [H, D]

    my_model.temperature = 500

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(
            subdomain=surf, value=lambda x: 2 * x[0] + x[1], species=H
        ),
        F.FixedConcentrationBC(
            subdomain=surf, value=lambda x: 2 * x[1] + x[0], species=D
        ),
    ]

    my_model.settings = F.Settings(
        atol=1e-12, rtol=1e-12, transient=True, final_time=10
    )
    my_model.settings.stepsize = F.Stepsize(1)

    my_model.exports = [
        F.VTXSpeciesExport(
            filename=tmpdir + "/out_checkpoint.bp",
            field=[H, D],
            checkpoint=True,
        ),
        F.VTXSpeciesExport(
            filename=tmpdir + "/model_1_out_h.bp",
            field=[H],
        ),
    ]

    my_model.initialise()
    my_model.run()

    my_model2 = F.HydrogenTransportProblem()
    my_model2.mesh = F.Mesh(mesh)
    my_model2.subdomains = [vol, surf]

    H = F.Species("H")
    D = F.Species("D")
    my_model2.species = [H, D]

    my_model2.temperature = 500

    my_model2.initial_conditions = [
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint.bp",
                name="H",
                timestamp=10,
                mesh=mesh,
            ),
            species=H,
            volume=vol,
        ),
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint.bp",
                name="D",
                timestamp=10,
                mesh=mesh,
            ),
            species=D,
            volume=vol,
        ),
    ]

    my_model2.settings = F.Settings(
        atol=1e-10, rtol=1e-10, transient=True, final_time=10
    )
    my_model2.settings.stepsize = F.Stepsize(0.1)

    my_model2.exports = [
        F.VTXSpeciesExport(
            filename=tmpdir + "/model_2_out_h.bp",
            field=[H],
        ),
    ]

    my_model2.initialise()
    my_model2.run()

    import numpy as np

    np.testing.assert_allclose(
        H.post_processing_solution.x.array,
        1.5,
        atol=1e-10,
    )


def test_VTXExport_times_added_to_milestones(tmpdir):
    """Creates a HydrogenTransportProblem object and checks that, if no
    stepsize.milestones are given and VTXExport.times are given, VTXExport.times are are
    added to stepsize.milestones by .initialise()

    Args:
        tmpdir (os.PathLike): path to the pytest temporary folder
    """
    H = F.Species("H", mobile=True)
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXSpeciesExport(
        field=H,
        filename=filename,
        times=[1, 2, 3],
    )

    # build
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D([1, 2, 3])
    my_model.species = [H]
    my_model.subdomains = [
        F.VolumeSubdomain1D(id=1, borders=[1, 3], material=F.Material(D_0=1, E_D=0))
    ]
    my_model.temperature = 100
    my_model.settings = F.Settings(
        atol=0.999,
        rtol=0.999,
        final_time=4,
        transient=True,
        stepsize=F.Stepsize(initial_value=3),
    )
    my_model.exports = [my_export]

    # run
    my_model.initialise()

    # test
    assert my_model.settings.stepsize.milestones == my_export.times


def test_vtx_writer_called_only_at_specified_times(tmpdir):
    """Test that the VTXWriter.write function is called the number of times specified in
    the export.times."""

    filename = str(tmpdir.join("my_export.bp"))

    my_model = F.HydrogenTransportProblem()

    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 10, 100))

    H = F.Species("H", mobile=True)
    my_model.species = [H]

    vol = F.VolumeSubdomain1D(
        id=1, borders=[0, 10], material=F.Material(D_0=1.0, E_D=0)
    )
    left = F.SurfaceSubdomain1D(id=2, x=0)
    my_model.subdomains = [vol, left]
    my_model.temperature = 500
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, value=5.0, species=H),
    ]
    my_model.exports = [F.VTXSpeciesExport(filename=filename, field=H, times=[2, 4, 6])]
    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=True,
        final_time=6,
        stepsize=F.Stepsize(initial_value=1),
    )

    with patch("dolfinx.io.VTXWriter") as MockWriter:
        # This is the mock instance of VTXWriter
        mock_writer_instance = MockWriter.return_value

        my_model.initialise()
        my_model.run()

        # Check number of write calls
        assert mock_writer_instance.write.call_count == 3

        # Check which times were passed to write
        actual_times = [
            call.args[0] for call in mock_writer_instance.write.call_args_list
        ]
        expected_times = [2, 4, 6]
        assert actual_times == expected_times


def test_writing_and_reading_of_species_function_using_checkpoints_discontinuous(
    tmpdir,
):
    """Tests that a model can write a checkpoint file and another model can read it."""
    my_model = F.HydrogenTransportProblemDiscontinuous()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    my_model.mesh = F.Mesh(mesh)

    mat = F.Material(name="mat", D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    vol1 = F.VolumeSubdomain(id=1, material=mat, locator=lambda x: x[0] <= 0.5)
    vol2 = F.VolumeSubdomain(id=2, material=mat, locator=lambda x: x[0] >= 0.5)
    surf1 = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 0.0))
    surf2 = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1.0))
    my_model.subdomains = [vol1, vol2, surf1, surf2]

    H = F.Species(name="H", subdomains=[vol1, vol2])
    D = F.Species(name="D", subdomains=[vol1, vol2])
    my_model.species = [H, D]

    my_model.temperature = 500

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=surf1, value=1.5, species=H),
        F.FixedConcentrationBC(subdomain=surf1, value=3.4, species=D),
        F.FixedConcentrationBC(subdomain=surf2, value=1.5, species=H),
        F.FixedConcentrationBC(subdomain=surf2, value=3.4, species=D),
    ]

    my_model.settings = F.Settings(
        atol=1e-12, rtol=1e-12, transient=True, final_time=10
    )
    my_model.settings.stepsize = F.Stepsize(1)

    my_model.exports = [
        F.VTXSpeciesExport(
            filename=tmpdir + "/out_checkpoint1.bp",
            field=[H, D],
            checkpoint=True,
            subdomain=vol1,
        ),
        F.VTXSpeciesExport(
            filename=tmpdir + "/out_checkpoint2.bp",
            field=[H, D],
            checkpoint=True,
            subdomain=vol2,
        ),
    ]

    my_model.initialise()
    my_model.run()

    my_model2 = F.HydrogenTransportProblemDiscontinuous()
    my_model2.mesh = F.Mesh(mesh)
    my_model2.subdomains = [vol1, vol2, surf1]

    H = F.Species("H", subdomains=[vol1, vol2])
    D = F.Species("D", subdomains=[vol1, vol2])
    my_model2.species = [H, D]

    my_model2.temperature = 500

    my_model2.initial_conditions = [
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint1.bp",
                name="H",
                timestamp=10,
                mesh=mesh,
            ),
            species=H,
            volume=vol1,
        ),
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint1.bp",
                name="D",
                timestamp=10,
                mesh=mesh,
            ),
            species=D,
            volume=vol1,
        ),
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint2.bp",
                name="H",
                timestamp=10,
                mesh=mesh,
            ),
            species=H,
            volume=vol2,
        ),
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint2.bp",
                name="D",
                timestamp=10,
                mesh=mesh,
            ),
            species=D,
            volume=vol2,
        ),
    ]

    my_model2.settings = F.Settings(
        atol=1e-10, rtol=1e-10, transient=True, final_time=10
    )
    my_model2.settings.stepsize = F.Stepsize(0.1)

    my_model2.exports = [
        F.VTXSpeciesExport(
            filename=tmpdir + "/model_2_out_h.bp",
            field=[H],
            subdomain=vol1,
        ),
    ]

    my_model2.initialise()
    my_model2.run()

    vol_1_H_solution = H.subdomain_to_post_processing_solution[vol1]
    vol_2_D_solution = D.subdomain_to_post_processing_solution[vol2]

    np.testing.assert_allclose(
        vol_1_H_solution.x.array,
        1.5,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        vol_2_D_solution.x.array,
        3.4,
        atol=1e-10,
    )


def _gather_sorted(func):
    """Gather a function's dof coordinates and values on rank 0, sorted by coordinate.

    Lets two functions on the same geometry be compared even when the mesh has been
    re-partitioned by a round trip through a file. Coordinates are rounded before
    sorting: the two partitions produce coordinates that differ in the last bits, which
    is enough to flip the order of neighbouring points and make the comparison fail
    spuriously.
    """
    space = func.function_space
    num_owned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
    coords = space.tabulate_dof_coordinates()[: space.dofmap.index_map.size_local]
    values = func.x.array[:num_owned]

    all_coords = MPI.COMM_WORLD.gather(coords, root=0)
    all_values = MPI.COMM_WORLD.gather(values, root=0)
    if MPI.COMM_WORLD.rank != 0:
        return None, None
    coords = np.round(np.concatenate(all_coords), 9)
    values = np.concatenate(all_values)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order], values[order]


def _bcast_from_root(value):
    """Share rank 0's verdict so that every rank asserts the same thing.

    Asserting only on rank 0 makes a *failure* hang instead of failing: rank 0 raises
    while the other ranks run on into the next collective call and wait forever.
    """
    return MPI.COMM_WORLD.bcast(value, root=0)


def _discontinuous_model(final_time=2.0):
    """A two-subdomain model with one species, used by the format tests below."""
    my_model = F.HydrogenTransportProblemDiscontinuous()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    my_model.mesh = F.Mesh(mesh)

    mat = F.Material(name="mat", D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    vol1 = F.VolumeSubdomain(id=1, material=mat, locator=lambda x: x[0] <= 0.5)
    vol2 = F.VolumeSubdomain(id=2, material=mat, locator=lambda x: x[0] >= 0.5)
    surf1 = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 0.0))
    surf2 = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1.0))
    my_model.subdomains = [vol1, vol2, surf1, surf2]

    H = F.Species(name="H", subdomains=[vol1, vol2])
    my_model.species = [H]
    my_model.temperature = 500
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=surf1, value=1.5, species=H),
        F.FixedConcentrationBC(subdomain=surf2, value=3.4, species=H),
    ]
    my_model.settings = F.Settings(
        atol=1e-12, rtol=1e-12, transient=True, final_time=final_time
    )
    my_model.settings.stepsize = F.Stepsize(1)
    return my_model, H, vol1, vol2


def test_vtkhdf_export_roundtrip(tmpdir):
    """A .vtkhdf export holds the solution at each timestep and reads back."""
    my_model, H, vol1, _ = _discontinuous_model()
    filename = str(tmpdir.join("out.vtkhdf"))
    my_model.exports = [
        F.SpeciesExport(filename=filename, field=[H], subdomain=vol1, format="vtkhdf")
    ]
    my_model.initialise()
    my_model.run()
    my_model.exports[0].close()

    assert os.path.exists(filename)

    # the export interpolates onto the geometry nodes, so compare there
    written = H.subdomain_to_post_processing_solution[vol1]
    geometry_space = dolfinx.fem.functionspace(
        written.function_space.mesh, ("Lagrange", 1)
    )
    expected = dolfinx.fem.Function(geometry_space)
    expected.interpolate(written)

    backend_args = {"name": "subdomain_1"}
    mesh_in = io4dolfinx.read_mesh(
        filename, comm=MPI.COMM_WORLD, backend="vtkhdf", backend_args=backend_args
    )
    read_back = io4dolfinx.read_point_data(
        filename,
        name="H",
        mesh=mesh_in,
        time=2.0,
        backend="vtkhdf",
        backend_args=backend_args,
    )

    exp_coords, exp_values = _gather_sorted(expected)
    got_coords, got_values = _gather_sorted(read_back)
    verdict = _bcast_from_root(
        None
        if MPI.COMM_WORLD.rank != 0
        else {
            "same_mesh": np.allclose(exp_coords, got_coords),
            "error": float(np.max(np.abs(exp_values - got_values))),
            "spread": float(got_values.max() - got_values.min()),
        }
    )
    assert verdict["same_mesh"], "meshes do not match"
    assert verdict["error"] < 1e-10, verdict["error"]
    # the field actually varies, so the comparison above means something
    # (its spread is well above the tolerance used for it)
    assert verdict["spread"] > 1e-3


def test_vtkhdf_multiblock_single_file(tmpdir):
    """Two subdomains exported to one filename become two blocks of one file."""
    my_model, H, vol1, vol2 = _discontinuous_model()
    filename = str(tmpdir.join("multi.vtkhdf"))
    my_model.exports = [
        F.SpeciesExport(filename, field=[H], subdomain=vol1, format="vtkhdf"),
        F.SpeciesExport(filename, field=[H], subdomain=vol2, format="vtkhdf"),
    ]
    my_model.initialise()
    my_model.run()
    for export in my_model.exports:
        export.close()

    assert os.path.exists(filename)

    structure = None
    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(filename, "r") as f:
            blocks = sorted(set(f["VTKHDF"].keys()) - {"Assembly"})
            structure = {
                "blocks": blocks,
                "fields": {b: list(f["VTKHDF"][b]["PointData"].keys()) for b in blocks},
                "times": {
                    b: f["VTKHDF"][b]["Steps"]["Values"][:].tolist() for b in blocks
                },
            }
    structure = _bcast_from_root(structure)

    assert structure["blocks"] == ["subdomain_1", "subdomain_2"]
    for block in structure["blocks"]:
        assert structure["fields"][block] == ["H"]
        assert np.allclose(structure["times"][block], [0.0, 1.0, 2.0])


@pytest.mark.parametrize("backend", ["adios2", "h5py"])
def test_checkpoint_of_temperature(tmpdir, backend):
    """Temperature can be checkpointed and read back; previously species-only.

    Runs on both checkpoint backends: the file an export writes has to be readable by
    :func:`festim.read_function_from_file` given the same backend.
    """
    my_model, _, _, _ = _discontinuous_model()
    filename = str(tmpdir.join("temperature"))
    my_model.temperature = lambda x: 500 + 100 * x[0]
    my_model.exports = [
        F.TemperatureExport(filename, format="checkpoint", backend=backend)
    ]
    my_model.initialise()
    my_model.run()
    my_model.exports[0].close()

    read_back = F.read_function_from_file(
        filename=my_model.exports[0].filename,
        name="temperature",
        timestamp=2.0,
        backend=backend,
    )
    expected = dolfinx.fem.Function(read_back.function_space)
    expected.interpolate(lambda x: 500 + 100 * x[0])

    exp_coords, exp_values = _gather_sorted(expected)
    got_coords, got_values = _gather_sorted(read_back)
    verdict = _bcast_from_root(
        None
        if MPI.COMM_WORLD.rank != 0
        else {
            "same_mesh": np.allclose(exp_coords, got_coords),
            "error": float(np.max(np.abs(exp_values - got_values))),
        }
    )
    assert verdict["same_mesh"]
    assert verdict["error"] < 1e-10, verdict["error"]

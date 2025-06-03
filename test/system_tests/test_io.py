from unittest.mock import patch

import mpi4py.MPI as MPI

import dolfinx
import numpy as np

import festim as F


def test_writing_and_reading_of_species_function_using_checkpoints(tmpdir):
    """
    Tests that a model can write a checkpoint file and another model can read it.
    """
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, nx=10, ny=10, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
    )
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh)

    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain(id=0, material=my_mat)
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
        F.InitialCondition(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint.bp",
                name="H",
                timestamp=10,
            ),
            species=H,
        ),
        F.InitialCondition(
            value=F.read_function_from_file(
                filename=tmpdir + "/out_checkpoint.bp",
                name="D",
                timestamp=10,
            ),
            species=D,
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
    stepsize.milestones are given and VTXExport.times are given, VTXExport.times are
    are added to stepsize.milestones by .initialise()

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
        atol=1e0,
        rtol=1e0,
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
    """test that the VTXWriter.write function is called the number of times specified in
    the export.times"""

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

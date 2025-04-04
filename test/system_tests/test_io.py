import festim as F

import dolfinx
import mpi4py.MPI as MPI


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

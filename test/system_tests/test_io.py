import festim as F

import dolfinx
import mpi4py.MPI as MPI


def test():
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, nx=10, ny=10, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
    )
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh)

    my_mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain(id=0, material=my_mat)
    surf = F.SurfaceSubdomain(id=0)
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
        atol=1e-12, rtol=1e-12, transient=True, final_time=100
    )
    my_model.settings.stepsize = F.Stepsize(1)

    my_model.exports = [
        F.VTXSpeciesExport(
            filename="H.bp",
            field=[H],
            checkpoint=True,
        ),
    ]

    my_model.initialise()
    my_model.run()

    my_model2 = F.HydrogenTransportProblem()
    my_model2.mesh = F.Mesh(mesh)
    my_model.subdomains = [vol, surf]

    H = F.Species("H")
    D = F.Species("D")
    my_model.species = [H, D]

    my_model.temperature = 500

    my_model.initial_conditions = [
        F.InitialConcentrationFromFile(
            filename="H.bp", species=H, name="H", timestamp=100
        ),
        # F.InitialConcentrationFromFile(
        #     filename="H.bp", species=D, name="D", timestamp=100
        # ),
    ]

    my_model.settings = F.Settings(
        atol=1e-12, rtol=1e-12, transient=True, final_time=100
    )
    my_model.settings.stepsize = F.Stepsize(1)

    my_model.initialise()
    my_model.run()

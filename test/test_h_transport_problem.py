import festim as F
import tqdm.autonotebook
import mpi4py.MPI as MPI
import dolfinx
import ufl
import numpy as np

# TODO test all the methods in the class


def test_iterate():
    """Test that the iterate method updates the solution and time correctly"""
    # BUILD
    my_model = F.HydrogenTransportProblem()

    my_model.settings = F.Settings(atol=1e-6, rtol=1e-6, final_time=10)

    my_model.progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )

    my_model.boundary_conditions = []

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    my_model.u = dolfinx.fem.Function(V)
    my_model.u_n = dolfinx.fem.Function(V)
    my_model.dt = dolfinx.fem.Constant(mesh, 2.0)
    v = ufl.TestFunction(V)

    source_value = 2.0
    form = (
        my_model.u - my_model.u_n
    ) / my_model.dt * v * ufl.dx - source_value * v * ufl.dx

    problem = dolfinx.fem.petsc.NonlinearProblem(form, my_model.u, bcs=[])
    my_model.solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    my_model.t = dolfinx.fem.Constant(mesh, 0.0)

    for i in range(10):
        # RUN
        my_model.iterate(skip_post_processing=True)

        # TEST

        # check that t evolves
        expected_t_value = (i + 1) * float(my_model.dt)
        assert np.isclose(float(my_model.t), expected_t_value)

        # check that u and u_n are updated
        expected_u_value = (i + 1) * float(my_model.dt) * source_value
        assert np.all(np.isclose(my_model.u.x.array, expected_u_value))

        assert np.all(np.isclose(my_model.u_n.x.array, expected_u_value))

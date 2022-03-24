from os import path
import FESTIM
import fenics as f
import numpy as np
from pathlib import Path
import timeit
import pytest


class TestPostProcessing:
    @pytest.fixture
    def my_sim(self):
        my_sim = FESTIM.Simulation({})
        my_sim.t = 0
        my_sim.mesh = FESTIM.MeshFromRefinements(10, 1)
        my_sim.settings = FESTIM.Settings(None, None, final_time=10)
        mat1 = FESTIM.Material(1, D_0=1, E_D=1)
        my_sim.materials = FESTIM.Materials([mat1])

        my_sim.mobile = FESTIM.Mobile()
        trap_1 = FESTIM.Trap(1, 1, 1, 1, 1, 1)
        my_sim.traps = FESTIM.Traps([trap_1])

        my_sim.mesh.define_measures(my_sim.materials)

        my_sim.V_DG1 = f.FunctionSpace(my_sim.mesh.mesh, "DG", 1)

        my_sim.T = FESTIM.Temperature(value=20)
        my_sim.T.create_functions(my_sim.mesh)
        my_sim.h_transport_problem = FESTIM.HTransportProblem(
            my_sim.mobile, my_sim.traps, my_sim.T, my_sim.settings,
            my_sim.initial_conditions
        )
        my_sim.h_transport_problem.define_function_space(my_sim.mesh)
        my_sim.h_transport_problem.initialise_concentrations()

        my_sim.materials.create_properties(
                my_sim.mesh.volume_markers, my_sim.T.T)
        my_sim.exports = FESTIM.Exports([])

        return my_sim

    def test_derived_quantities_size(self, my_sim):
        derived_quantities = FESTIM.DerivedQuantities()
        derived_quantities.derived_quantities = [
            FESTIM.SurfaceFlux("solute", 1),
            FESTIM.AverageVolume("T", 1),
            FESTIM.TotalVolume("1", 1)
        ]
        derived_quantities.assign_measures_to_quantities(my_sim.mesh.dx, my_sim.mesh.ds)
        derived_quantities.assign_properties_to_quantities(my_sim.materials)

        my_sim.exports.exports = [derived_quantities]
        t = 0
        dt = 1
        for i in range(1, 3):
            t += dt
            my_sim.t = t
            my_sim.run_post_processing()

        assert len(my_sim.exports.exports[0].data) == i + 1
        assert my_sim.exports.exports[0].data[i][0] == t

    def test_pure_diffusion(self, my_sim):
        my_sim.materials = FESTIM.Materials(
            [
                FESTIM.Material(1, D_0=1, E_D=1, borders=[0, 0.5]),
                FESTIM.Material(2, D_0=1, E_D=1, borders=[0.5, 1]),
            ]
        )

        my_sim.mesh.define_measures(my_sim.materials)

        f.assign(
            my_sim.h_transport_problem.u.sub(0),
            f.interpolate(f.Constant(10), my_sim.h_transport_problem.V.sub(0).collapse()))
        f.assign(
            my_sim.h_transport_problem.u.sub(1),
            f.interpolate(f.Constant(1), my_sim.h_transport_problem.V.sub(1).collapse()))

        derived_quantities = FESTIM.DerivedQuantities()
        derived_quantities.derived_quantities = [
            FESTIM.AverageVolume("solute", 2),
            FESTIM.AverageVolume("T", 2),
            FESTIM.AverageVolume("retention", 2),
            FESTIM.MinimumVolume("retention", 1)
        ]
        derived_quantities.assign_measures_to_quantities(my_sim.mesh.dx, my_sim.mesh.ds)

        my_sim.exports.exports = [derived_quantities]

        t = 0
        dt = 1
        for i in range(1, 3):
            t += dt
            my_sim.t = t
            my_sim.run_post_processing()
            data = derived_quantities.data

            assert len(data) == i + 1
            assert data[i][0] == t
            assert data[i][1] == 10
            assert data[i][2] == 20
            assert data[i][3] == pytest.approx(11)
            assert data[i][4] == pytest.approx(11)

    def test_fluxes(self, my_sim):

        my_sim.T = FESTIM.Temperature(100*FESTIM.x + 200)
        my_sim.T.create_functions(my_sim.mesh)

        my_sim.materials = FESTIM.Materials(
            [
                FESTIM.Material(1, D_0=5, E_D=0.4, thermal_cond=3, borders=[0, 0.5]),
                FESTIM.Material(2, D_0=6, E_D=0.5, thermal_cond=5, borders=[0.5, 1]),
            ]
        )

        my_sim.mesh.define_measures(my_sim.materials)

        my_sim.materials.create_properties(
                my_sim.mesh.volume_markers, my_sim.T.T)

        u_expr = f.Expression("2*x[0]", degree=1)
        f.assign(
            my_sim.h_transport_problem.u.sub(0),
            f.interpolate(u_expr, my_sim.h_transport_problem.V.sub(0).collapse()))

        derived_quantities = FESTIM.DerivedQuantities()
        derived_quantities.derived_quantities = [
            FESTIM.SurfaceFlux("solute", 1),
            FESTIM.SurfaceFlux("solute", 2),
            FESTIM.SurfaceFlux("T", 1),
            FESTIM.SurfaceFlux("T", 2),
        ]
        derived_quantities.assign_measures_to_quantities(my_sim.mesh.dx, my_sim.mesh.ds)
        derived_quantities.assign_properties_to_quantities(my_sim.materials)

        my_sim.exports.exports = [derived_quantities]

        my_sim.run_post_processing()
        data = derived_quantities.data
        D_x_0 = 5*f.exp(-0.4/FESTIM.k_B/my_sim.T.T(0))
        D_x_1 = 6*f.exp(-0.5/FESTIM.k_B/my_sim.T.T(1))
        lambda_x_0 = 3
        lambda_x_1 = 5
        grad_c = 2
        grad_T = 100
        assert np.isclose(data[1][1], -1*grad_c*D_x_0)
        assert np.isclose(data[1][2], -1*grad_c*D_x_1*-1)
        assert np.isclose(data[1][3], -1*grad_T*lambda_x_0)
        assert np.isclose(data[1][4], -1*grad_T*lambda_x_1*-1)

    def test_performance_xdmf_export_every_N_iterations(self, my_sim, tmpdir):
        d = tmpdir.mkdir("test_folder")
        my_sim.exports.exports = FESTIM.XDMFExports(
                fields=["solute", "T"],
                labels=['solute', 'temperature'],
                folder=str(Path(d))).xdmf_exports

        # export every 10 iterations
        for export in my_sim.exports.exports:
            export.nb_iterations_between_exports = 30
        my_sim.nb_iterations = 0
        start = timeit.default_timer()
        for i in range(40):
            my_sim.run_post_processing()

        stop = timeit.default_timer()
        short_time = stop - start
        print(short_time)

        # export every time
        my_sim.nb_iterations = 0
        for export in my_sim.exports.exports:
            export.nb_iterations_between_exports = 1
        start = timeit.default_timer()
        for i in range(40):
            my_sim.run_post_processing()

        stop = timeit.default_timer()
        long_time = stop - start
        print(long_time)

        assert short_time < long_time

    def test_xdmf_export_only_last_timestep(self, my_sim, tmpdir):
        d = tmpdir.mkdir("test_folder")
        my_sim.exports.exports = FESTIM.XDMFExports(
                fields=["solute", "T"],
                labels=['solute', 'temperature'],
                mode="last",
                folder=str(Path(d))).xdmf_exports
        my_sim.exports.final_time = 1
        my_sim.t = 0

        my_sim.run_post_processing()
        assert not path.exists(str(Path(d)) + '/solute.xdmf')

        my_sim.t = my_sim.exports.final_time
        my_sim.run_post_processing()
        assert path.exists(str(Path(d)) + '/solute.xdmf')

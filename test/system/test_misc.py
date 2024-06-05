import festim as F
import fenics as f
import numpy as np
import pytest
import os


def test_convective_flux(tmpdir):
    """
    Tests that convective boundary condition works correctly

    Args:
        tmpdir (os.PathLike): path to the pytest temporary folder
    """
    sim = F.Simulation()

    sim.mesh = F.MeshFromVertices(np.linspace(0, 1, num=50))

    sim.T = F.HeatTransferProblem(transient=False)

    T_external = 2
    sim.boundary_conditions = [
        F.ConvectiveFlux(h_coeff=10, T_ext=T_external, surfaces=1),
        F.DirichletBC(surfaces=2, value=T_external + 1, field="T"),
    ]

    sim.materials = F.Materials([F.Material(1, D_0=1, E_D=0, thermal_cond=2)])

    sim.exports = F.Exports(
        [
            F.XDMFExport(
                "T", checkpoint=False, filename="{}/temperature.xdmf".format(tmpdir)
            )
        ]
    )

    sim.settings = F.Settings(1e-10, 1e-10, transient=False)

    sim.initialise()
    sim.run()

    assert sim.T.T(0) > T_external


def test_error_steady_state_with_stepsize():
    """Checks that an error is raised when a stepsize is given for a steady state simulation"""
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromRefinements(1000, size=1)

    my_model.materials = F.Materials([F.Material(D_0=1, E_D=0, id=1)])

    my_model.T = F.Temperature(value=400)

    my_model.settings = F.Settings(
        transient=False,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )

    my_model.dt = F.Stepsize(initial_value=1)
    with pytest.raises(
        AttributeError, match="dt must be None in steady state simulations"
    ):
        my_model.initialise()


def test_error_transient_without_stepsize():
    """Checks that an error is raised when a stepsize is not given for a transient simulation"""
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromRefinements(1000, size=1)

    my_model.materials = F.Materials([F.Material(D_0=1, E_D=0, id=1)])

    my_model.T = F.Temperature(value=400)

    my_model.settings = F.Settings(
        transient=True,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        final_time=10,
    )

    my_model.dt = None
    with pytest.raises(
        AttributeError, match="dt must be provided in transient simulations"
    ):
        my_model.initialise()


def test_high_recombination_flux():
    """Added test that catches the bug #465
    Checks that with chemical potential and a high recombination coefficient
    the solver doesn't diverge
    """
    model = F.Simulation()

    model.mesh = F.MeshFromVertices(range(10))

    model.boundary_conditions = [
        F.DirichletBC(surfaces=1, value=1, field=0),
        F.RecombinationFlux(Kr_0=1000000, E_Kr=0, order=2, surfaces=2),
    ]

    model.T = F.Temperature(100)

    model.materials = F.Materials([F.Material(id=1, D_0=1, E_D=0, S_0=1, E_S=0)])

    model.settings = F.Settings(1e-10, 1e-10, transient=False, chemical_pot=True)

    model.initialise()
    model.run()


@pytest.mark.parametrize("field", ["coucou", "2", "-1", 2, -1])
def test_wrong_value_for_bc_field(field):
    """
    Tests that an error is raised when a wrong value of field is
    given for a boundary condition
    '2' is not a correct value because there is only one trap in the model
    """
    sim = F.Simulation()

    sim.mesh = F.MeshFromVertices(np.linspace(0, 1, num=50))

    sim.T = F.Temperature(500)

    sim.materials = F.Materials([F.Material(1, D_0=1, E_D=0)])

    sim.traps = F.Trap(1, 1, 1, 1, materials="1", density=1)

    sim.settings = F.Settings(1e-10, 1e-10, transient=False)

    with pytest.raises(ValueError):
        sim.boundary_conditions = [F.BoundaryCondition(surfaces=1, field=field)]
        sim.initialise()


@pytest.mark.parametrize("field", ["solute", "T"])
@pytest.mark.parametrize("surfaces", [1, 2, [1, 2]])
def test_error_DirichletBC_on_same_surface(field, surfaces):
    """
    Tests that an error is raised when a DiricheltBC is set on
    a surface together with another boundary condition for the
    same field
    """
    sim = F.Simulation()

    sim.mesh = F.MeshFromVertices(np.linspace(0, 1, num=10))

    sim.T = F.Temperature(500)

    sim.materials = F.Materials([F.Material(1, D_0=1, E_D=0)])

    sim.settings = F.Settings(1e-10, 1e-10, transient=False)

    with pytest.raises(ValueError):
        sim.boundary_conditions = [
            F.FluxBC(value=1, field=field, surfaces=1),
            F.DirichletBC(value=1, field=field, surfaces=2),
            F.DirichletBC(value=1, field=field, surfaces=surfaces),
        ]
        sim.initialise()


@pytest.mark.parametrize(
    "final_time,stepsize,export_times",
    [(1, 0.1, [0.2, 0.5]), (1e-7, 1e-9, [1e-8, 1.5e-8, 2e-8])],
)
def test_txt_export_desired_times(tmp_path, final_time, stepsize, export_times):
    """
    Tests that TXTExport can be exported at desired times
    Also catches the bug #682
    """
    my_model = F.Simulation()

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1))
    my_model.materials = F.Material(1, 1, 0)
    my_model.settings = F.Settings(1e-10, 1e-10, final_time=final_time)
    my_model.T = F.Temperature(500)
    my_model.dt = F.Stepsize(stepsize)

    my_export = F.TXTExport(
        "solute", times=export_times, filename="{}/mobile_conc.txt".format(tmp_path)
    )
    my_model.exports = [my_export]

    my_model.initialise()
    my_model.run()

    assert os.path.exists(my_export.filename)

    data = np.genfromtxt(
        my_export.filename,
        skip_header=1,
        delimiter=",",
    )
    assert len(data[0, :]) == len(my_export.times) + 1


def test_txt_export_all_times(tmp_path):
    """
    Tests that TXTExport can be exported at all timesteps

    Args:
        tmp_path (os.PathLike): path to a temporary folder
    """
    my_model = F.Simulation()

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1))
    my_model.materials = F.Material(1, 1, 0)
    my_model.settings = F.Settings(1e-10, 1e-10, final_time=1)
    my_model.T = F.Temperature(500)
    my_model.dt = F.Stepsize(0.1)

    my_export = F.TXTExport("solute", filename="{}/mobile_conc.txt".format(tmp_path))
    my_model.exports = [my_export]

    my_model.initialise()
    my_model.run()

    assert os.path.exists(my_export.filename)

    data = np.genfromtxt(
        my_export.filename,
        skip_header=1,
        delimiter=",",
    )
    assert len(data[0, :]) == 11


def test_txt_export_steady_state(tmp_path):
    """
    Tests that TXTExport can be exported in steady state

    Args:
        tmp_path (os.PathLike): path to a temporary folder
    """
    my_model = F.Simulation()

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1))
    my_model.materials = F.Material(1, 1, 0)
    my_model.settings = F.Settings(1e-10, 1e-10, transient=False)
    my_model.T = F.Temperature(500)

    my_export = F.TXTExport("solute", filename="{}/mobile_conc.txt".format(tmp_path))
    my_model.exports = [my_export]

    my_model.initialise()
    my_model.run()

    assert os.path.exists(my_export.filename)

    txt = open(my_export.filename)
    header = txt.readline().rstrip()
    txt.close()

    assert header == "x,t=steady"


def test_finaltime_overshoot():
    """Checks that the time doesn't overshoot the final time"""
    my_model = F.Simulation()

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1))
    my_model.materials = F.Material(1, 1, 0)
    my_model.settings = F.Settings(1e-10, 1e-10, final_time=1)
    my_model.T = F.Temperature(500)
    my_model.dt = F.Stepsize(0.1)

    my_model.initialise()
    my_model.run()

    assert np.isclose(my_model.t, my_model.settings.final_time)


def test_derived_quantities_exported_last_timestep_with_small_stepsize(tmp_path):
    """test to catch bug #566"""
    my_model = F.Simulation()

    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, num=100))

    my_model.materials = F.Material(id=1, D_0=1, E_D=0)

    my_model.T = F.Temperature(value=300)

    my_model.dt = F.Stepsize(
        initial_value=99.9999999,
    )

    my_model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, final_time=100
    )

    list_of_derived_quantities = [F.TotalVolume("solute", volume=1)]

    derived_quantities = F.DerivedQuantities(
        list_of_derived_quantities,
        filename=f"{tmp_path}/out.csv",
    )

    my_model.exports = [derived_quantities]

    my_model.initialise()
    my_model.run()

    assert os.path.exists(f"{tmp_path}/out.csv")


def test_small_timesteps_final_time_bug():
    """
    Test to catch the bug #682
    Runs a sim on small timescales and checks that the final time is reached
    """
    my_model = F.Simulation(log_level=40)
    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, 10))
    my_model.materials = F.Material(1, 1, 0.1)
    my_model.T = F.Temperature(1000)
    my_model.settings = F.Settings(1e-10, 1e-10, final_time=1e-7)
    my_model.dt = F.Stepsize(1e-9)
    my_model.initialise()
    my_model.run()

    assert np.isclose(my_model.t, my_model.settings.final_time, atol=0.0)


def test_materials_setter():
    """
    Checks that @materials.setter properly assigns F.Materials to F.Simulation.materials
    see #694 for the details
    """
    my_model = F.Simulation()
    test_materials = F.Materials([])
    my_model.materials = test_materials
    assert my_model.materials is test_materials


class TestFestimProblem:
    """Tests the methods of the festim.Problem class"""

    # define the FESTIM problem
    mesh = f.UnitIntervalMesh(8)
    V = f.FunctionSpace(mesh, "CG", 1)
    u = f.Function(V)
    v = f.TestFunction(V)
    s = u * v * f.dx + v * f.dx
    J = f.derivative(s, u)
    problem = F.Problem(J, s, [])

    # define the fenics assembler used in festim.Problem
    assembler = f.SystemAssembler(J, s, [])
    x = f.PETScVector()

    def test_F(self):
        """
        Creates two epty matrices and checks
        that the festim.Problem.J properly assembles the RHS of Ax=b
        """
        b1 = f.PETScVector()
        b2 = f.PETScVector()

        self.problem.F(b1, self.x)
        self.assembler.assemble(b2, self.x)

        assert (b1 == b2).all()

    def test_J(self):
        """
        Creates two epty matrices and checks
        that the festim.Problem.J properly assembles the LHS of Ax=b
        """
        A1 = f.PETScMatrix()
        A2 = f.PETScMatrix()

        self.problem.J(A1, self.x)
        self.assembler.assemble(A2)

        assert (A1.array() == A2.array()).all()


class TestWarningsCustomSolver:
    """
    Creates a simulation object and checks that a TypeError (UserWarning) is raised
    when the newton_solver attribute is given a value of the wrong type (solver is overwritten)
    """

    def sim(self):
        """Defines a model"""
        my_sim = F.Simulation()
        my_sim.mesh = F.MeshFromVertices([1, 2, 3])
        my_mat = F.Materials(
            [F.Material(id=1, D_0=1, E_D=0, thermal_cond=1, heat_capacity=1, rho=1)]
        )
        my_sim.materials = my_mat
        my_sim.T = F.HeatTransferProblem(
            transient=True, initial_condition=F.InitialCondition(field="T", value=1)
        )
        my_sim.traps = F.ExtrinsicTrap(
            1,
            1,
            1,
            1,
            my_mat,
            phi_0=1,
            n_amax=2,
            n_bmax=2,
            eta_a=3,
            eta_b=4,
            f_a=5,
            f_b=6,
        )
        # add source to the HeatTransferProblem
        my_sim.settings = F.Settings(
            transient=True,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            final_time=1,
        )
        my_sim.dt = F.Stepsize(1)
        return my_sim

    @pytest.mark.parametrize("value", ["coucou", [0, 0], -1.0])
    def test_wrong_type_solver_h_transport(self, value):
        """
        Checks that a TypeError is raised when the newton_solver attribute
        of the HTransportProblem class is given a value of the wrong type
        """
        problem = self.sim()
        problem.initialise()

        with pytest.raises(
            TypeError,
            match="accepted type for newton_solver is fenics.NewtonSolver",
        ):
            problem.h_transport_problem.newton_solver = value

    @pytest.mark.parametrize("value", ["coucou", [0, 0], -1.0])
    def test_wrong_type_solver_heat_transport(self, value):
        """
        Checks that a TypeError is raised when the newton_solver attribute
        of the HeatTransferProblem class is given a value of the wrong type
        """
        problem = self.sim()

        with pytest.raises(
            TypeError,
            match="accepted type for newton_solver is fenics.NewtonSolver",
        ):
            problem.T.newton_solver = value

    @pytest.mark.parametrize("value", ["coucou", [0, 0], -1.0])
    def test_wrong_type_solver_ex_trap(self, value):
        """
        Checks that a TypeError is raised when the newton_solver attribute
        of the ExtrinsicTrap class is given a value of the wrong type
        """
        problem = self.sim()

        with pytest.raises(
            TypeError,
            match="accepted type for newton_solver is fenics.NewtonSolver",
        ):
            problem.traps[0].newton_solver = value

    def test_warn_solver_h_transport(self):
        """
        Checks that a warning is raised when the newton_solver attribute
        of the HTransportProblem class is overwritten
        """
        problem = self.sim()
        problem.initialise()

        with pytest.warns(
            UserWarning,
            match="Settings for the Newton solver will be overwritten",
        ):
            problem.h_transport_problem.newton_solver = f.NewtonSolver()

    def test_warn_solver_heat_transport(self):
        """
        Checks that a warning is raised when the newton_solver attribute
        of the HeatTransferProblem class is overwritten
        """
        problem = self.sim()
        problem.initialise()

        with pytest.warns(
            UserWarning,
            match="Settings for the Newton solver will be overwritten",
        ):
            problem.T.newton_solver = f.NewtonSolver()

    def test_warn_solver_ex_trap(self):
        """
        Checks that a warning is raised when the newton_solver attribute
        of the ExtrinsicTrap class is overwritten
        """
        problem = self.sim()
        problem.initialise()

        with pytest.warns(
            UserWarning,
            match="Settings for the Newton solver will be overwritten",
        ):
            problem.traps[0].newton_solver = f.NewtonSolver()

            
def test_error_raised_when_no_IC_heat_transfer():
    """
    Checks that an error is raised when no initial condition is provided for
    transient heat transfer simulations
    """
    my_model = F.Simulation()

    my_model.T = F.HeatTransferProblem(transient=True, initial_condition=None)
    my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, 10))
    my_model.materials = F.Material(1, 1, 0.1)
    my_model.settings = F.Settings(1e-10, 1e-10, final_time=1e-7)
    my_model.dt = F.Stepsize(1e-9)

    with pytest.raises(
        AttributeError,
        match="Initial condition is required for transient heat transfer simulations",
    ):
        my_model.initialise()
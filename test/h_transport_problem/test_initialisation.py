import festim
import pytest
import fenics
from pathlib import Path


def test_initialisation_from_xdmf(tmpdir):
    mesh = fenics.UnitSquareMesh(5, 5)
    V = fenics.VectorFunctionSpace(mesh, "P", 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)
    ini_u = fenics.Expression("1", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(1).collapse())
    fenics.assign(u.sub(1), ini_u)

    d = tmpdir.mkdir("Initial solutions")
    file1 = d.join("u_1out.xdmf")
    file2 = d.join("u_2out.xdmf")
    print(Path(file1))
    with fenics.XDMFFile(str(Path(file1))) as f:
        f.write_checkpoint(
            u.sub(0), "1", 2, fenics.XDMFFile.Encoding.HDF5, append=False
        )
    with fenics.XDMFFile(str(Path(file2))) as f:
        f.write_checkpoint(
            u.sub(1), "2", 2, fenics.XDMFFile.Encoding.HDF5, append=False
        )
        f.write_checkpoint(u.sub(1), "2", 4, fenics.XDMFFile.Encoding.HDF5, append=True)

    initial_conditions = [
        festim.InitialCondition(
            field=0, value=str(Path(file1)), label="1", time_step=0
        ),
        festim.InitialCondition(
            field=1, value=str(Path(file2)), label="2", time_step=0
        ),
    ]
    my_trap = festim.Trap(1, 1, 1, 1, ["mat_name"], 1)

    my_problem = festim.HTransportProblem(
        festim.Mobile(),
        festim.Traps([my_trap]),
        festim.Temperature(300),
        festim.Settings(1e10, 1e-10),
        initial_conditions,
    )

    my_problem.V = V
    my_mats = festim.Materials()
    my_mats.S = None
    my_problem.initialise_concentrations()
    w = my_problem.u_n
    assert fenics.errornorm(u, w) == 0


def test_initialisation_with_expression():
    """
    Test that initialise_solutions interpolates correctly
    from an expression
    """
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, "P", 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1+x[0] + x[1]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)
    ini_u = fenics.Expression("1+x[0]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(1).collapse())
    fenics.assign(u.sub(1), ini_u)

    initial_conditions = [
        festim.InitialCondition(field=0, value=1 + festim.x + festim.y),
        festim.InitialCondition(field=1, value=1 + festim.x),
    ]
    my_trap = festim.Trap(1, 1, 1, 1, ["mat_name"], 1)

    my_problem = festim.HTransportProblem(
        festim.Mobile(),
        festim.Traps([my_trap]),
        festim.Temperature(300),
        festim.Settings(1e10, 1e-10),
        initial_conditions,
    )

    my_problem.V = V
    my_problem.initialise_concentrations()
    w = my_problem.u_n
    assert fenics.errornorm(u, w) == 0


def test_initialisation_with_expression_chemical_pot():
    """
    Test that initialise_solutions interpolates correctly
    from an expression with conservation of chemical potential
    """

    S = 2
    mesh = fenics.UnitSquareMesh(8, 8)
    vm = fenics.MeshFunction("size_t", mesh, 2, 1)
    V = fenics.VectorFunctionSpace(mesh, "P", 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("(1+x[0] + x[1])/S", S=S, degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)
    ini_u = fenics.Expression("1+x[0]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(1).collapse())
    fenics.assign(u.sub(1), ini_u)

    initial_conditions = [
        festim.InitialCondition(field=0, value=1 + festim.x + festim.y),
        festim.InitialCondition(field=1, value=1 + festim.x),
    ]
    my_trap = festim.Trap(1, 1, 1, 1, ["mat_name"], 1)

    my_theta = festim.Theta()
    my_theta.materials = festim.Materials([festim.Material(1, 1, 0, S_0=S, E_S=0)])
    my_theta.volume_markers = vm
    my_theta.T = festim.Temperature(10)
    my_theta.T.create_functions(festim.Mesh(mesh))
    my_theta.S = S

    my_problem = festim.HTransportProblem(
        my_theta,
        festim.Traps([my_trap]),
        festim.Temperature(300),
        festim.Settings(1e10, 1e-10),
        initial_conditions,
    )

    my_problem.V = V
    my_problem.initialise_concentrations()
    expected = u
    produced = my_problem.u_n
    assert fenics.errornorm(expected, produced) == pytest.approx(0)


def test_initialisation_default():
    """
    Test that initialise_solutions interpolates correctly
    if nothing is given (default is 0)
    """
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, "P", 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    initial_conditions = []
    my_trap = festim.Trap(1, 1, 1, 1, ["mat_name"], 1)

    my_problem = festim.HTransportProblem(
        festim.Mobile(),
        festim.Traps([my_trap]),
        festim.Temperature(300),
        festim.Settings(1e10, 1e-10),
        initial_conditions,
    )

    my_problem.V = V
    my_problem.initialise_concentrations()
    w = my_problem.u_n
    assert fenics.errornorm(u, w) == 0


def test_initialisation_solute_only():
    """
    Test that initialise_solutions interpolates correctly
    if solution has only 1 component (ie solute)
    """
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.FunctionSpace(mesh, "P", 1)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1 + x[0] + x[1]", degree=1)
    u = fenics.interpolate(ini_u, V)

    initial_conditions = [
        festim.InitialCondition(field=0, value=1 + festim.x + festim.y),
    ]
    my_trap = festim.Trap(1, 1, 1, 1, ["mat_name"], 1)

    my_problem = festim.HTransportProblem(
        festim.Mobile(),
        festim.Traps([my_trap]),
        festim.Temperature(300),
        festim.Settings(1e10, 1e-10),
        initial_conditions,
    )

    my_problem.V = V
    my_problem.initialise_concentrations()
    w = my_problem.u_n
    assert fenics.errornorm(u, w) == 0


def test_initialisation_no_component():
    """
    Test that initialise_solutions set component at 0
    by default
    """
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, "P", 1, 3)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1 + x[0] + x[1]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)

    initial_conditions = [
        festim.InitialCondition(value=1 + festim.x + festim.y),
    ]
    my_trap = festim.Trap(1, 1, 1, 1, ["mat_name"], 1)

    my_problem = festim.HTransportProblem(
        festim.Mobile(),
        festim.Traps([my_trap]),
        festim.Temperature(300),
        festim.Settings(1e10, 1e-10),
        initial_conditions,
    )

    my_problem.V = V
    my_problem.initialise_concentrations()
    w = my_problem.u_n
    assert fenics.errornorm(u, w) == 0

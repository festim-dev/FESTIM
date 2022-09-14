from attr import has
from sympy.matrices import expressions
import festim
import fenics
import pytest
import sympy as sp
import numpy as np


def test_define_dirichlet_bcs_theta():
    """
    Test the function apply_boundary_condition()
    when conservation of chemical potential is
    required.
    Meaning that the BC value given by the user must
    be multiplied by S(T) in apply_boundary_condition()
    - multi material
    - transient bc
    - transient T
    - space dependent T
    """
    S_01 = 2
    S_02 = 3
    E_S1 = 0.1
    E_S2 = 0.2

    mesh = fenics.UnitSquareMesh(4, 4)
    V = fenics.FunctionSpace(mesh, "P", 1)
    u = fenics.Function(V)
    v = fenics.TestFunction(V)

    vm = fenics.MeshFunction("size_t", mesh, 2, 1)
    left = fenics.CompiledSubDomain("x[0] < 0.5")
    right = fenics.CompiledSubDomain("x[0] >= 0.5")
    left.mark(vm, 1)
    right.mark(vm, 2)

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)
    left = fenics.CompiledSubDomain("x[0] < 0.0001")
    right = fenics.CompiledSubDomain("x[0] > 0.99999999")
    left.mark(sm, 1)
    right.mark(sm, 2)

    my_mesh = festim.Mesh(mesh, vm, sm)
    my_mesh.dx = fenics.dx()
    my_mesh.ds = fenics.ds()

    my_temp = festim.Temperature(value=200 + (festim.x + 1) * festim.t)
    my_temp.create_functions(my_mesh)

    mat1 = festim.Material(1, None, None, S_0=S_01, E_S=E_S1)
    mat2 = festim.Material(2, None, None, S_0=S_02, E_S=E_S2)
    my_mats = festim.Materials([mat1, mat2])

    my_bc = festim.DirichletBC([1, 2], value=200 + festim.t, field=0)
    my_bc.create_dirichletbc(
        V,
        my_temp.T,
        surface_markers=sm,
        chemical_pot=True,
        materials=my_mats,
        volume_markers=vm,
    )
    expressions = my_bc.sub_expressions + [my_bc.expression]
    bcs = my_bc.dirichlet_bc

    F = fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx

    for i in range(0, 3):
        my_temp.expression.t = i
        my_temp.T.assign(fenics.interpolate(my_temp.expression, V))
        expressions[0].t = i
        expressions[1].t = i

        # Test that the expression is correct at vertices
        expr = fenics.interpolate(expressions[1], V)
        assert np.isclose(
            expr(0.25, 0.5),
            (200 + i) / (S_01 * np.exp(-E_S1 / festim.k_B / my_temp.T(0.25, 0.5))),
        )
        assert np.isclose(
            expr(0.75, 0.5),
            (200 + i) / (S_02 * np.exp(-E_S2 / festim.k_B / my_temp.T(0.75, 0.5))),
        )

        # Test that the BCs can be applied to a problem
        # and gives the correct values
        fenics.solve(F == 0, u, bcs[0])
        assert np.isclose(
            u(0.25, 0.5),
            (200 + i) / (S_01 * np.exp(-E_S1 / festim.k_B / my_temp.T(0, 0.5))),
        )
        fenics.solve(F == 0, u, bcs[1])
        assert np.isclose(
            u(0.75, 0.5),
            (200 + i) / (S_02 * np.exp(-E_S2 / festim.k_B / my_temp.T(1, 0.5))),
        )


def test_bc_recomb():
    """Test the function boundary_conditions.define_dirichlet_bcs
    with bc type dc_imp
    """
    phi = 3 + 10 * festim.t
    R_p = 5 + festim.x
    D_0 = 2
    E_D = 0.5
    Kr_0 = 2
    E_Kr = 0.35

    mesh = fenics.UnitSquareMesh(4, 4)
    my_mesh = festim.Mesh(mesh)
    my_mesh.dx = fenics.dx()
    my_mesh.ds = fenics.ds()
    V = fenics.FunctionSpace(mesh, "P", 1)
    T_expr = 500 + (festim.x + 1) * 100 * festim.t

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)

    my_temp = festim.Temperature(value=T_expr)
    my_temp.create_functions(my_mesh)

    my_bc = festim.ImplantationDirichlet(
        [1, 2], phi=phi, R_p=R_p, D_0=D_0, E_D=E_D, Kr_0=Kr_0, E_Kr=E_Kr
    )
    my_bc.create_dirichletbc(V, my_temp.T, surface_markers=sm)
    expressions = my_bc.sub_expressions + [my_bc.expression]

    for current_time in range(0, 3):
        my_temp.expression.t = current_time
        my_temp.T.assign(fenics.interpolate(my_temp.expression, V))
        expressions[0].t = current_time
        expressions[1].t = current_time

        for x_ in [0, 1]:
            T = float(T_expr.subs(festim.t, current_time).subs(festim.x, x_))
            D = D_0 * np.exp(-E_D / festim.k_B / T)
            K = Kr_0 * np.exp(-E_Kr / festim.k_B / T)
            # Test that the expression is correct at vertices
            val_phi = phi.subs(festim.t, current_time).subs(festim.x, x_)
            val_R_p = R_p.subs(festim.t, current_time).subs(festim.x, x_)
            assert np.isclose(
                expressions[-1](x_, 0.5),
                float(val_phi * val_R_p / D + (val_phi / K) ** 0.5),
            )


def test_bc_recomb_instant_recomb():
    """Test the function boundary_conditions.define_dirichlet_bcs
    with bc type dc_imp (with instantaneous recombination)
    """
    phi = 3 + 10 * festim.t
    R_p = 5 + festim.x
    D_0 = 200
    E_D = 0.25

    # Set up
    mesh = fenics.UnitSquareMesh(4, 4)
    my_mesh = festim.Mesh(mesh)
    my_mesh.dx = fenics.dx()
    my_mesh.ds = fenics.ds()
    V = fenics.FunctionSpace(mesh, "P", 1)
    T_expr = 500 + (festim.x + 1) * 100 * festim.t

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)

    my_temp = festim.Temperature(value=T_expr)
    my_temp.create_functions(my_mesh)

    my_bc = festim.ImplantationDirichlet([1, 2], phi=phi, R_p=R_p, D_0=D_0, E_D=E_D)
    my_bc.create_dirichletbc(V, my_temp.T, surface_markers=sm)
    expressions = my_bc.sub_expressions + [my_bc.expression]

    for current_time in range(0, 3):
        my_temp.expression.t = current_time
        my_temp.T.assign(fenics.interpolate(my_temp.expression, V))
        for expr in expressions:
            expr.t = current_time

        for x_ in [0, 1]:
            T = float(T_expr.subs(festim.t, current_time).subs(festim.x, x_))
            D = D_0 * np.exp(-E_D / festim.k_B / T)
            # Test that the expression is correct at vertices
            val_phi = phi.subs(festim.t, current_time).subs(festim.x, x_)
            val_R_p = R_p.subs(festim.t, current_time).subs(festim.x, x_)
            assert np.isclose(expressions[-1](x_, 0.5), float(val_phi * val_R_p / D))


def test_bc_recomb_chemical_pot():
    """Tests the function boundary_conditions.define_dirichlet_bcs()
    with type dc_imp and conservation of chemical potential
    """
    phi = 3
    R_p = 5
    D_0 = 2
    E_D = 0.5
    Kr_0 = 2
    E_Kr = 0.5
    S_01 = 2
    S_02 = 3
    E_S1 = 0.1
    E_S2 = 0.2

    mesh = fenics.UnitSquareMesh(4, 4)
    my_mesh = festim.Mesh(mesh)
    my_mesh.dx = fenics.dx()
    my_mesh.ds = fenics.ds()
    V = fenics.FunctionSpace(mesh, "P", 1)

    vm = fenics.MeshFunction("size_t", mesh, 2, 1)
    left = fenics.CompiledSubDomain("x[0] < 0.5")
    right = fenics.CompiledSubDomain("x[0] >= 0.5")
    left.mark(vm, 1)
    right.mark(vm, 2)

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)
    left = fenics.CompiledSubDomain("x[0] < 0.0001")
    left.mark(sm, 1)
    right = fenics.CompiledSubDomain("x[0] > 0.99999999")
    right.mark(sm, 2)

    my_temp = festim.Temperature(value=200 + (festim.x + 1) * festim.t)
    my_temp.create_functions(my_mesh)

    mat1 = festim.Material(1, None, None, S_0=S_01, E_S=E_S1)
    mat2 = festim.Material(2, None, None, S_0=S_02, E_S=E_S2)
    my_mats = festim.Materials([mat1, mat2])

    # bcs, expressions = define_dirichlet_bcs(my_sim)
    V = fenics.FunctionSpace(mesh, "P", 1)
    my_bc = festim.ImplantationDirichlet(
        surfaces=[1, 2], phi=phi, R_p=R_p, D_0=D_0, E_D=E_D, Kr_0=Kr_0, E_Kr=E_Kr
    )
    my_bc.create_dirichletbc(
        V,
        my_temp.T,
        surface_markers=sm,
        chemical_pot=True,
        materials=my_mats,
        volume_markers=vm,
    )
    bcs = my_bc.dirichlet_bc
    expressions = my_bc.sub_expressions + [my_bc.expression]
    # Set up formulation
    u = fenics.Function(V)
    v = fenics.TestFunction(V)
    F = fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx

    for i in range(0, 3):
        my_temp.expression.t = i
        my_temp.T.assign(fenics.interpolate(my_temp.expression, V))
        for expr in expressions:
            expr.t = i

        T_left = 200 + i
        T_right = 200 + 2 * i
        D_left = D_0 * np.exp(-E_D / festim.k_B / T_left)
        D_right = D_0 * np.exp(-E_D / festim.k_B / T_right)
        K_left = Kr_0 * np.exp(-E_Kr / festim.k_B / T_left)
        K_right = Kr_0 * np.exp(-E_Kr / festim.k_B / T_right)
        S_left = S_01 * np.exp(-E_S1 / festim.k_B / my_temp.T(0, 0.5))
        S_right = S_02 * np.exp(-E_S2 / festim.k_B / my_temp.T(1, 0.5))

        # Test that the BCs can be applied to a problem
        # and gives the correct values
        fenics.solve(F == 0, u, bcs[0])
        expected = (phi * R_p / D_left + (phi / K_left) ** 0.5) / S_left
        computed = u(0.25, 0.5)
        assert np.isclose(expected, computed)

        fenics.solve(F == 0, u, bcs[1])
        expected = (phi * R_p / D_right + (phi / K_right) ** 0.5) / S_right
        computed = u(0.25, 0.5)
        assert np.isclose(expected, computed)


def test_sievert_bc_varying_time():
    """Checks the methode SievertsBC.create_expression produces the expected expression"""
    # build
    T = fenics.Constant(300)
    pressure_expr = 1e5 * (1 + festim.t)
    s_0_expr = 100
    E_S_expr = 0.5
    my_bc = festim.SievertsBC(
        surfaces=1, pressure=pressure_expr, S_0=s_0_expr, E_S=E_S_expr
    )

    pressure_expr = fenics.Expression(sp.printing.ccode(pressure_expr), t=0, degree=1)
    s_0_expr = fenics.Expression(sp.printing.ccode(s_0_expr), t=0, degree=1)
    E_S_expr = fenics.Expression(sp.printing.ccode(E_S_expr), t=0, degree=1)
    T_expr = fenics.Expression(sp.printing.ccode(T), t=0, degree=1)
    # run
    my_bc.create_expression(T)
    # test

    def sieverts(T, S_0, E_S, pressure):
        S = S_0 * fenics.exp(-E_S / festim.k_B / T)
        return S * pressure**0.5

    prms = {"S_0": s_0_expr, "E_S": E_S_expr, "pressure": pressure_expr}

    expected = festim.BoundaryConditionExpression(
        T_expr,
        eval_function=sieverts,
        pressure=pressure_expr,
        S_0=s_0_expr,
        E_S=E_S_expr,
    )
    assert my_bc.expression(0) == pytest.approx(expected(0))

    for prm in my_bc.sub_expressions:
        if hasattr(prm, "t"):
            prm.t += 10
    for prm in prms.values():
        prm.t += 10
    assert my_bc.expression(0) == pytest.approx(expected(0))


def test_sievert_bc_varying_temperature():
    """Checks the method SievertsBC.create_expression produces the expected expression"""
    # build
    T = fenics.Constant(300)
    pressure_expr = 1e5 * (1 + festim.t)
    s_0_expr = 100
    E_S_expr = 0.5
    my_bc = festim.SievertsBC(
        surfaces=1, pressure=pressure_expr, S_0=s_0_expr, E_S=E_S_expr
    )

    pressure_expr = fenics.Expression(sp.printing.ccode(pressure_expr), t=0, degree=1)
    s_0_expr = fenics.Expression(sp.printing.ccode(s_0_expr), t=0, degree=1)
    E_S_expr = fenics.Expression(sp.printing.ccode(E_S_expr), t=0, degree=1)

    # run
    my_bc.create_expression(T)
    # test

    def sieverts(T, S_0, E_S, pressure):
        S = S_0 * fenics.exp(-E_S / festim.k_B / T)
        return S * pressure**0.5

    expected = festim.BoundaryConditionExpression(
        T, eval_function=sieverts, S_0=s_0_expr, E_S=E_S_expr, pressure=pressure_expr
    )
    assert my_bc.expression(0) == pytest.approx(expected(0))

    T.assign(1000)
    assert my_bc.expression(0) == pytest.approx(expected(0))


def test_henry_bc_varying_time():
    """Checks the method HenrysBC.create_expression produces the expected
    expression
    """
    # build
    T = fenics.Constant(300)
    pressure_expr = 1e5 * (1 + festim.t)
    H_0_expr = 100
    E_H_expr = 0.5
    my_bc = festim.HenrysBC(
        surfaces=1, pressure=pressure_expr, H_0=H_0_expr, E_H=E_H_expr
    )
    pressure_expr = fenics.Expression(sp.printing.ccode(pressure_expr), t=0, degree=1)
    H_0_expr = fenics.Expression(sp.printing.ccode(H_0_expr), t=0, degree=1)
    E_H_expr = fenics.Expression(sp.printing.ccode(E_H_expr), t=0, degree=1)
    T_expr = fenics.Expression(sp.printing.ccode(T), t=0, degree=1)

    # run
    my_bc.create_expression(T)
    # test

    def henrys(T, H_0, E_H, pressure):
        H = H_0 * fenics.exp(-E_H / festim.k_B / T)
        return H * pressure

    prms = {"H_0": H_0_expr, "E_H": E_H_expr, "pressure": pressure_expr}

    expected = festim.BoundaryConditionExpression(
        T_expr, eval_function=henrys, pressure=pressure_expr, H_0=H_0_expr, E_H=E_H_expr
    )
    assert my_bc.expression(0) == pytest.approx(expected(0))

    for prm in my_bc.sub_expressions:
        if hasattr(prm, "t"):
            prm.t += 10
    for prm in prms.values():
        prm.t += 10
    assert my_bc.expression(0) == pytest.approx(expected(0))


def test_henry_bc_varying_temperature():
    """Checks the method HenrysBC.create_expression produces the expected
    expression
    """
    # build
    T = fenics.Constant(300)
    pressure_expr = 1e5 * (1 + festim.t)
    H_0_expr = 100
    E_H_expr = 0.5
    my_bc = festim.HenrysBC(
        surfaces=1, pressure=pressure_expr, H_0=H_0_expr, E_H=E_H_expr
    )

    pressure_expr = fenics.Expression(sp.printing.ccode(pressure_expr), t=0, degree=1)
    H_0_expr = fenics.Expression(sp.printing.ccode(H_0_expr), t=0, degree=1)
    E_H_expr = fenics.Expression(sp.printing.ccode(E_H_expr), t=0, degree=1)

    # run
    my_bc.create_expression(T)
    # test

    def henrys(T, H_0, E_H, pressure):
        H = H_0 * fenics.exp(-E_H / festim.k_B / T)
        return H * pressure

    expected = festim.BoundaryConditionExpression(
        T, eval_function=henrys, H_0=H_0_expr, E_H=E_H_expr, pressure=pressure_expr
    )
    assert my_bc.expression(0) == pytest.approx(expected(0))

    T.assign(1000)
    assert my_bc.expression(0) == pytest.approx(expected(0))


def test_create_expression_dc_custom():
    """Creates a dc_custom bc and checks create_expression returns
    the correct expression
    """
    # build
    def func(T, prm1, prm2):
        return 2 * T + prm1 * prm2

    T = fenics.Expression("2 + x[0] + t", degree=1, t=0)
    expressions = [T]
    # run
    my_BC = festim.CustomDirichlet(
        surfaces=[1, 0], function=func, prm1=1 + 2 * festim.t, prm2=2
    )
    my_BC.create_expression(T)
    expressions += my_BC.sub_expressions

    # test
    expected = 2 * (2 + festim.x + festim.t) + (1 + 2 * festim.t) * 2
    expected = fenics.Expression(sp.printing.ccode(expected), t=0, degree=1)
    for t in range(10):
        expected.t = t
        for expr in expressions:
            expr.t = t
        for x in range(5):
            assert expected(x) == my_BC.expression(x)


def test_create_form_flux_custom():
    """Creates a flux_custom bc and checks
    create_form returns
    the correct form
    """
    # build
    def func(T, c, prm1, prm2):
        return 2 * T + c + prm1 * prm2

    expr_prm1 = 1 + 2 * festim.t + festim.x
    expr_prm2 = 2
    expr_T = 2 + festim.x + festim.t
    expr_c = festim.x * festim.x

    T = fenics.Expression(sp.printing.ccode(expr_T), degree=1, t=0)
    solute = fenics.Expression(sp.printing.ccode(expr_c), degree=1, t=0)
    expressions = [T, solute]

    # run
    my_BC = festim.CustomFlux(
        surfaces=[1, 0], field="T", function=func, prm1=expr_prm1, prm2=expr_prm2
    )
    my_BC.create_form(T, solute)
    value_BC = my_BC.form

    # test
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, "P", 1)
    expected_expr = 2 * expr_T + expr_c + expr_prm1 * expr_prm2
    expected_expr = fenics.Expression(sp.printing.ccode(expected_expr), t=0, degree=1)
    for t in range(10):

        expected_expr.t = t
        for expr in my_BC.sub_expressions + expressions:
            expr.t = t
        expected = fenics.project(expected_expr, V)
        computed = fenics.project(value_BC, V)
        for x in [0, 0.5, 1]:
            assert computed(x) == pytest.approx(expected(x))


def test_convective_flux():
    expr_T = 2 + festim.x
    T = fenics.Expression(sp.printing.ccode(expr_T), degree=1, t=0)

    my_BC = festim.ConvectiveFlux(surfaces=[0], h_coeff=expr_T, T_ext=expr_T)
    my_BC.create_form(T, None)


def test_recomb_flux():
    expr = 2 + festim.x
    T = fenics.Expression(sp.printing.ccode(expr), degree=1, t=0)
    c = fenics.Expression(sp.printing.ccode(expr), degree=1, t=0)

    my_BC = festim.RecombinationFlux(surfaces=[0], Kr_0=expr, E_Kr=expr, order=2)
    my_BC.create_form(T, c)


def test_string_for_field_in_dirichletbc():
    """Test catching issue #462"""
    # build
    mesh = fenics.UnitSquareMesh(4, 4)

    surface_marker = fenics.MeshFunction("size_t", mesh, 1, 0)

    V = fenics.VectorFunctionSpace(mesh, "P", 1, 2)
    bc = festim.DirichletBC(surfaces=[0, 1], value=1, field="solute")

    # test
    bc.create_dirichletbc(V, fenics.Constant(1), surface_marker)
